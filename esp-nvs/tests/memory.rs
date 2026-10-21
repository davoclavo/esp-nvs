//! Memory benchmarks: the stack and heap `Nvs` uses while it loads a partition and serves reads.
//!
//! Run them in release, which is what ends up on a device, and with output, to see the numbers:
//!
//! ```text
//! cargo test --release -p esp-nvs --test memory -- --nocapture --test-threads=1
//! ```
//!
//! The figures are measured on the host, so they are not those of an Xtensa or RISC-V build, but
//! they move together with them: a buffer copied on the stack here is copied there as well. The
//! benchmarks are ignored in debug builds and their bounds only checked in release, a debug build
//! spills far more to the stack than any device firmware would.

use std::alloc::{
    GlobalAlloc,
    Layout,
    System,
};
use std::cell::Cell;

use esp_nvs::mem_flash::MemFlash;
use esp_nvs::{
    FLASH_SECTOR_SIZE,
    Key,
    Nvs,
};

// ---------------------------------------------------------------------------------------------
// Heap: a global allocator that counts, per thread, what is allocated

struct CountingAllocator;

#[derive(Clone, Copy, Default)]
struct HeapCounters {
    // Signed: memory allocated on another thread and freed on this one counts negative.
    current: isize,
    peak: isize,
    allocations: usize,
}

thread_local! {
    static HEAP: Cell<HeapCounters> = const {
        Cell::new(HeapCounters {
            current: 0,
            peak: 0,
            allocations: 0,
        })
    };
}

fn track(f: impl FnOnce(&mut HeapCounters)) {
    // `try_with` because the allocator also runs while the thread local is being torn down.
    let _ = HEAP.try_with(|cell| {
        let mut counters = cell.get();
        f(&mut counters);
        cell.set(counters);
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        track(|c| {
            c.current += layout.size() as isize;
            c.peak = c.peak.max(c.current);
            c.allocations += 1;
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        track(|c| c.current -= layout.size() as isize);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        track(|c| {
            c.current = c.current - layout.size() as isize + new_size as isize;
            c.peak = c.peak.max(c.current);
            c.allocations += 1;
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// ---------------------------------------------------------------------------------------------
// Stack: paint the unused stack below the caller with a pattern, run the code, and look for the
// deepest byte that no longer holds the pattern.

const PAINTED_STACK: usize = 256 * 1024;
const PAINT: u64 = 0xA5A5_A5A5_A5A5_A5A5;

/// The address of a local in a frame below the caller's, which is free again once this returns.
#[inline(never)]
fn below_caller_frame() -> usize {
    let marker = 0u64;
    core::hint::black_box(core::ptr::addr_of!(marker)) as usize & !7
}

/// Returns how deep below the caller's frame `f` reached into the stack.
///
/// Writing to and reading from memory below the stack pointer is outside of what Rust promises
/// anything about, which is why this lives in a test and why every access is volatile. It is the
/// usual stack painting a firmware does, and only ever touches the stack of a thread spawned with
/// room to spare for it.
#[inline(never)]
fn stack_used_by<R>(f: impl FnOnce() -> R) -> (R, usize) {
    // Painting starts below a frame that has already returned rather than below a local of this
    // one: this frame holds `f`, and in a debug build it is large enough that painting a fixed
    // distance below one of its locals would overwrite the rest of them.
    let top = below_caller_frame();
    let paint_end = top;
    let paint_start = top - PAINTED_STACK;

    let mut addr = paint_start;
    while addr < paint_end {
        unsafe { core::ptr::write_volatile(addr as *mut u64, PAINT) };
        addr += 8;
    }

    let result = core::hint::black_box(f());

    let mut addr = paint_start;
    while addr < paint_end && unsafe { core::ptr::read_volatile(addr as *const u64) } == PAINT {
        addr += 8;
    }
    assert!(addr > paint_start, "the painted stack overflowed, raise PAINTED_STACK");

    (result, top - addr)
}

#[derive(Debug)]
struct Usage {
    stack: usize,
    heap_peak: isize,
    heap_retained: isize,
    allocations: usize,
}

/// Measures `f` on a thread of its own, so that neither the test harness nor other tests running
/// in parallel show up in the figures.
fn measure<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> (R, Usage) {
    std::thread::Builder::new()
        .stack_size(4 * PAINTED_STACK)
        .spawn(move || {
            let before = HEAP.with(|cell| {
                let mut counters = cell.get();
                counters.peak = counters.current;
                counters.allocations = 0;
                cell.set(counters);
                counters
            });

            let (result, stack) = stack_used_by(f);

            let after = HEAP.with(Cell::get);
            let usage = Usage {
                stack,
                heap_peak: after.peak - before.current,
                heap_retained: after.current - before.current,
                allocations: after.allocations,
            };
            (result, usage)
        })
        .unwrap()
        .join()
        .unwrap()
}

fn report(name: &str, usage: &Usage) {
    println!(
        "{name:<34} stack: {:>6} B  heap peak: {:>7} B  heap retained: {:>6} B  allocations: {:>5}",
        usage.stack, usage.heap_peak, usage.heap_retained, usage.allocations
    );
}

/// Checks a bound in release builds, see the module documentation. Debug builds skip it, for when
/// the benchmarks are run there with `--include-ignored`.
fn assert_at_most(name: &str, what: &str, measured: usize, bound: usize) {
    if cfg!(debug_assertions) {
        return;
    }
    assert!(
        measured <= bound,
        "{name}: {what} is {measured} B, more than the {bound} B it is expected to stay within"
    );
}

// ---------------------------------------------------------------------------------------------
// Partitions to load

/// 64 kB, a partition of the size people commonly give NVS.
const PAGES: usize = 16;
/// 512 kB, a large partition.
const LARGE_PAGES: usize = 128;

fn empty_partition(pages: usize) -> Vec<u8> {
    MemFlash::new(pages).into_inner()
}

/// A partition in typical use: namespaces with a mix of primitives, strings and blobs, several
/// values written more than once. Each namespace takes about 250 of the 126 entries a page has, so
/// four of them fill half of a 64 kB partition, and 32 half of a 512 kB one.
fn populated_partition(pages: usize, namespaces: usize) -> Vec<u8> {
    let mut flash = MemFlash::new(pages);
    {
        let mut nvs = Nvs::new(0, pages * FLASH_SECTOR_SIZE, &mut flash).unwrap();
        for ns in 0..namespaces {
            let namespace = Key::from_str(&format!("ns{ns}"));
            for i in 0..80u32 {
                nvs.set(&namespace, &Key::from_str(&format!("u32_{i}")), i).unwrap();
            }
            for i in 0..20u32 {
                nvs.set(&namespace, &Key::from_str(&format!("u32_{i}")), i + 1000)
                    .unwrap();
            }
            for i in 0..10 {
                let value = format!("string value number {i} in namespace {ns}");
                nvs.set(&namespace, &Key::from_str(&format!("str_{i}")), value.as_str())
                    .unwrap();
            }
            let blob: Vec<u8> = (0..3000).map(|i| i as u8).collect();
            nvs.set(&namespace, &Key::from_str("blob"), blob.as_slice()).unwrap();
        }
    }
    flash.into_inner()
}

/// Loads `image` and returns the instance, so that what it keeps shows up as retained heap.
fn load(image: Vec<u8>) -> Nvs<MemFlash> {
    let size = image.len();
    Nvs::new(0, size, MemFlash::from_bytes(image)).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Benchmarks

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "only meaningful in release, see the module documentation"
)]
fn init_empty_partition() {
    let image = empty_partition(PAGES);
    let (_nvs, usage) = measure(move || load(image));
    report("init, empty 64 kB partition", &usage);

    // One sector read into a buffer, and little else. Copies of that buffer took it to 14 kB.
    assert_at_most("init_empty_partition", "stack", usage.stack, 7 * 1024);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "only meaningful in release, see the module documentation"
)]
fn init_populated_partition() {
    let image = populated_partition(PAGES, 4);
    let (_nvs, usage) = measure(move || load(image));
    report("init, populated 64 kB partition", &usage);

    assert_at_most("init_populated_partition", "stack", usage.stack, 7 * 1024);
    // Duplicate detection held a map of lists, one per item on the partition, which peaked at
    // 44 kB here for an instance that keeps 3 kB.
    assert_at_most(
        "init_populated_partition",
        "heap peak",
        usage.heap_peak as usize,
        8 * 1024,
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "only meaningful in release, see the module documentation"
)]
fn init_empty_large_partition() {
    let image = empty_partition(LARGE_PAGES);
    let (_nvs, usage) = measure(move || load(image));
    report("init, empty 512 kB partition", &usage);

    assert_at_most("init_empty_large_partition", "stack", usage.stack, 7 * 1024);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "only meaningful in release, see the module documentation"
)]
fn init_populated_large_partition() {
    let image = populated_partition(LARGE_PAGES, 32);
    let (_nvs, usage) = measure(move || load(image));
    report("init, populated 512 kB partition", &usage);

    assert_at_most("init_populated_large_partition", "stack", usage.stack, 7 * 1024);
    // 351 kB before duplicate detection went from a map of lists to a flat list.
    assert_at_most(
        "init_populated_large_partition",
        "heap peak",
        usage.heap_peak as usize,
        64 * 1024,
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "only meaningful in release, see the module documentation"
)]
fn read_primitive() {
    let mut nvs = load(populated_partition(PAGES, 4));
    let (_nvs, usage) = measure(move || {
        let value: u32 = nvs.get(&Key::from_str("ns2"), &Key::from_str("u32_70")).unwrap();
        assert_eq!(value, 70);
        nvs
    });
    report("get u32", &usage);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "only meaningful in release, see the module documentation"
)]
fn read_string() {
    let mut nvs = load(populated_partition(PAGES, 4));
    let (_nvs, usage) = measure(move || {
        let value: String = nvs.get(&Key::from_str("ns1"), &Key::from_str("str_3")).unwrap();
        assert_eq!(value, "string value number 3 in namespace 1");
        nvs
    });
    report("get string", &usage);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "only meaningful in release, see the module documentation"
)]
fn read_blob() {
    const SIZE: usize = 20_000;
    let mut nvs = load(empty_partition(PAGES));
    let blob: Vec<u8> = (0..SIZE).map(|i| (i * 7) as u8).collect();
    nvs.set(&Key::from_str("ns"), &Key::from_str("blob"), blob.as_slice())
        .unwrap();

    let (_nvs, usage) = measure(move || {
        let value: Vec<u8> = nvs.get(&Key::from_str("ns"), &Key::from_str("blob")).unwrap();
        assert_eq!(value, blob);
        // Handed back, so that freeing the blob, which was allocated outside, does not count.
        (nvs, blob)
    });
    report("get 20 kB blob", &usage);

    // The blob itself has to be on the heap, the question is what else is.
    assert_at_most("read_blob", "heap peak", usage.heap_peak as usize, SIZE + 1024);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "only meaningful in release, see the module documentation"
)]
fn write_with_defragmentation() {
    const ROUNDS: u32 = 2000;
    let mut nvs = load(populated_partition(PAGES, 4));
    let (_nvs, usage) = measure(move || {
        // Overwriting long enough for the partition to have to reclaim pages several times over.
        for round in 0..ROUNDS {
            nvs.set(&Key::from_str("ns0"), &Key::from_str("counter"), round)
                .unwrap();
            if round % 10 == 0 {
                // An odd length, so the string ends in a partial write.
                let value = format!("round {round:04}");
                nvs.set(&Key::from_str("ns0"), &Key::from_str("label"), value.as_str())
                    .unwrap();
            }
        }
        nvs
    });
    report("2000 writes with defragmentation", &usage);
}
