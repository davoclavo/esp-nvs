//! NVS initialization and sector loading.
//!
//! This module contains the logic for reading flash sectors and initializing
//! the in-memory page structures during [`Nvs`](crate::Nvs) startup.

use alloc::vec;
use core::ops::Not;

#[cfg(feature = "defmt")]
use defmt::trace;

use crate::Nvs;
use crate::blob::{
    BlobIndex,
    BlobIndexEntryBlobIndexData,
};
use crate::error::Error;
use crate::page::{
    ItemHashListEntry,
    LoadPageResult,
    ThinPage,
    ThinPageState,
};
use crate::platform::Platform;
#[cfg(feature = "debug-logs")]
use crate::raw::slice_with_nullbytes_to_str;
use crate::raw::{
    ENTRIES_PER_PAGE,
    ENTRY_STATE_BITMAP_SIZE,
    EntryMapState,
    FLASH_SECTOR_SIZE,
    ITEM_SIZE,
    Item,
    ItemType,
    PAGE_HEADER_SIZE,
    PageHeaderRaw,
};
use crate::types::{
    ChunkIndex,
    NamespaceIndex,
    VersionOffset,
};

impl<T> Nvs<T>
where
    T: Platform,
{
    pub(crate) fn load_sectors(&mut self) -> Result<(), Error> {
        #[cfg(feature = "defmt")]
        trace!("load_sectors");

        #[cfg(feature = "debug-logs")]
        println!("internal: load_sectors");

        let mut blob_index = self.scan_sectors()?;

        // Finishing an interrupted defragmentation erases pages the scan above has already
        // accounted for. The blob index in particular would count each chunk on both the source
        // and its partial copy, find twice the data its index claims, and delete a blob that is
        // perfectly intact. So start over from what is on flash once the recovery is done.
        // ESP-IDF never leaves more than one page `Freeing`, but nothing stops flash from saying
        // otherwise, so all of them are finished.
        let mut recovered = false;
        while self.continue_free_page()? {
            recovered = true;
        }
        if recovered {
            self.pages.clear();
            self.free_pages.clear();
            self.namespaces.clear();
            self.legacy_blobs.clear();
            blob_index = self.scan_sectors()?;
        }

        // Settle the blobs first, so at most one version of each is left, then check for duplicate
        // entries and mark older ones as erased. This handles cases where deletion failed after a
        // successful write.
        self.cleanup_dirty_blobs(blob_index)?;

        self.cleanup_duplicate_entries()?;

        self.migrate_legacy_blobs()?;

        Ok(())
    }

    /// Rewrites every blob in the single-page format of ESP-IDF before v4.0 as a multi-page blob,
    /// the only format this crate writes.
    ///
    /// That is an ordinary overwrite: the new blob is written first and the legacy item erased
    /// after, so a power loss in between leaves both, which the next boot resolves in favour of the
    /// newer one, and migrates again what is still legacy.
    ///
    /// It is best effort. A partition without room for the copy keeps the legacy blob, which is
    /// still read, until there is room; so does one that does not read back, rather than being
    /// turned into a blob with the wrong content.
    fn migrate_legacy_blobs(&mut self) -> Result<(), Error> {
        let legacy = core::mem::take(&mut self.legacy_blobs);
        for (namespace_index, key) in legacy {
            let Some(namespace) = self
                .namespaces
                .iter()
                .find(|(_, index)| **index == namespace_index)
                .map(|(namespace, _)| *namespace)
            else {
                continue;
            };
            // The scan may have found an item that the cleanups after it have since replaced.
            match self.load_item(namespace_index, ChunkIndex::Any, &key) {
                Ok((_, _, item)) if item.type_ == ItemType::Blob => {}
                Err(Error::FlashError) => return Err(Error::FlashError),
                _ => continue,
            }
            let data = match self.get_blob(&namespace, &key) {
                Ok(data) => data,
                Err(Error::FlashError) => return Err(Error::FlashError),
                Err(_) => continue,
            };
            match self.set_blob(&namespace, key, &data) {
                Ok(()) => {}
                Err(Error::FlashError) => return Err(Error::FlashError),
                // No room for this one, and so for none of the others either.
                Err(_) => return Ok(()),
            }
        }

        Ok(())
    }

    /// Loads every sector into `self.pages` and `self.free_pages`, and the namespaces into
    /// `self.namespaces`, returning what was found about blobs.
    fn scan_sectors(&mut self) -> Result<BlobIndex, Error> {
        let mut blob_index = BlobIndex::new();
        let sectors = self.sectors as usize;
        for sector_idx in 0..sectors {
            let sector_addr = self.base_address + sector_idx * FLASH_SECTOR_SIZE;
            match self.load_sector(sector_addr, &mut blob_index)? {
                LoadPageResult::Empty(page) => self.free_pages.push(page),
                LoadPageResult::Used(page) => self.pages.push(page),
            };
        }

        #[cfg(feature = "debug-logs")]
        println!("internal: load_sectors: blob_index: {:?}", blob_index);

        self.ensure_active_page_order()?;

        Ok(blob_index)
    }

    /// Loads the sector at `sector_address`. The namespaces found on it go straight into
    /// `self.namespaces` and what it holds of blobs into `blob_index`, which collects them across
    /// all sectors.
    pub(crate) fn load_sector(
        &mut self,
        sector_address: usize,
        blob_index: &mut BlobIndex,
    ) -> Result<LoadPageResult, Error> {
        #[cfg(feature = "defmt")]
        trace!("load_sector: @{:#08x}", sector_address);

        #[cfg(feature = "debug-logs")]
        println!("  raw: load page: 0x{sector_address:04X}");

        let mut buf = [0u8; FLASH_SECTOR_SIZE];
        self.hal
            .read(sector_address as _, &mut buf)
            .map_err(|_| Error::FlashError)?;

        // The sector is split into its parts once, rather than each being sliced out of `buf` where
        // it is used. Everything below is read out of `buf` in place: reinterpreting the whole
        // sector as a `RawPage` took two more copies of it, one to sanitize the item types and the
        // transmuted page itself, which tripled what loading a sector costs on the stack.
        const { assert!(PAGE_HEADER_SIZE + ENTRY_STATE_BITMAP_SIZE + ENTRIES_PER_PAGE * ITEM_SIZE == FLASH_SECTOR_SIZE) };
        let (header_bytes, rest) = buf
            .split_first_chunk::<PAGE_HEADER_SIZE>()
            .expect("a sector holds a page header");
        let (entry_state_bitmap, entries) = rest
            .split_first_chunk::<ENTRY_STATE_BITMAP_SIZE>()
            .expect("a sector holds an entry state bitmap");
        let (entries, _) = entries.as_chunks::<ITEM_SIZE>();

        if *header_bytes == [0xFFu8; PAGE_HEADER_SIZE] {
            #[cfg(feature = "debug-logs")]
            println!("  raw: load page: 0x{sector_address:04X} -> uninitialized");

            // A blank header does not make a blank page: an erase the power cut short, or a header
            // write that never happened, leaves data behind it. Handed out as uninitialized, the
            // page would be initialized without an erase and new items programmed over the old
            // bytes. Corrupt pages are erased before they are used.
            let mut page = ThinPage::uninitialized(sector_address);
            if buf.iter().all(|it| *it == 0xFF).not() {
                page.header.state = ThinPageState::Corrupt;
            }
            return Ok(LoadPageResult::Empty(page));
        }

        // Safety: the header is plain old data. It is either returned as a CORRUPT/INVALID/EMPTY
        // page directly or its CRC is checked below.
        let header = unsafe { PageHeaderRaw { raw: *header_bytes }.page_header };

        #[cfg(feature = "debug-logs")]
        {
            let state = crate::raw::PageState::from(header.state);
            println!("  raw: load page: 0x{sector_address:04X} -> {state}");
        }

        let mut page = ThinPage {
            address: sector_address,
            header: header.into(),
            entry_state_bitmap: *entry_state_bitmap,
            erased_entry_count: 0,
            used_entry_count: 0,
            item_hash_list: vec![],
        };

        match page.header.state {
            ThinPageState::Corrupt | ThinPageState::Invalid => {
                return Ok(LoadPageResult::Empty(page));
            }
            ThinPageState::Uninitialized => {
                // validate that the page is truly empty
                if buf.iter().all(|it| *it == 0xFF).not() {
                    page.header.state = ThinPageState::Corrupt;
                };

                return Ok(LoadPageResult::Empty(page));
            }
            ThinPageState::Freeing => (),
            ThinPageState::Active => (),
            ThinPageState::Full => (),
        }

        if header.crc != header.calculate_crc32(T::crc32) {
            page.header.state = ThinPageState::Corrupt;
            return Ok(LoadPageResult::Empty(page));
        };

        // Grown one push at a time, the list would end up with up to twice the room it needs, and
        // a page that takes no more writes never uses that room. The active page is left to grow,
        // as its list does with every write.
        if page.header.state != ThinPageState::Active {
            page.item_hash_list.reserve_exact(page.hashed_item_bound(entries));
        }

        // This iterator desugaring is necessary to be able to skip entries, e.g. a BLOB or STR
        // entries are followed by entries containing their raw value.
        //
        // Each entry is copied out of `buf` on its own, which is also where its type byte is
        // sanitized, see `Item::from_raw`. `buf` itself keeps the bytes as they are on flash.
        let mut item_iter = entries.iter().map(|raw| Item::from_raw(*raw)).zip(u8::MIN..u8::MAX);
        'item_iter: while let Some((item, item_index)) = item_iter.next() {
            let state = page.get_entry_state(item_index);

            // `span` is an unvalidated u8 straight from flash, and everything below uses it to walk
            // entries, mark ranges of the entry map and add to the page's u8 entry counters. All of
            // that assumes the span's entries are on this page. A span of zero, or one reaching
            // past the last entry, is corrupt and has to be recognised before it is
            // trusted: summing it into `used_entry_count` overflowed, which is a panic
            // in debug and a wrong count in release, and it happens during `Nvs::new`,
            // so a single bad byte took the partition down at startup before anything
            // could be read.
            let span_fits_page = item.span >= 1 && item_index as usize + item.span as usize <= ENTRIES_PER_PAGE;

            match state {
                EntryMapState::Illegal => {
                    page.erased_entry_count += 1;
                    continue 'item_iter;
                }
                EntryMapState::Erased => {
                    page.erased_entry_count += 1;
                    continue 'item_iter;
                }
                EntryMapState::Empty => {
                    if entries[item_index as usize].iter().all(|it| *it == 0xFF) {
                        continue 'item_iter;
                    }

                    // Not blank, so something was written here but the map was not updated yet.
                    // Either it is a complete item and recovered below, or it
                    // is what a torn write left behind. That has to be marked
                    // erased and counted like one: left as it is, it
                    // is where the next item would be written, programmed on top of the leftovers,
                    // and both the new value and the old one it replaces would be lost.
                    let calculated_crc = item.calculate_crc32(T::crc32);
                    if item.crc != calculated_crc || !span_fits_page {
                        page.set_entry_state(&mut self.hal, item_index as _, EntryMapState::Erased)?;
                        page.erased_entry_count += 1;
                        continue 'item_iter;
                    }

                    match item.type_ {
                        ItemType::U8
                        | ItemType::I8
                        | ItemType::U16
                        | ItemType::I16
                        | ItemType::U32
                        | ItemType::I32
                        | ItemType::U64
                        | ItemType::I64
                        | ItemType::BlobIndex
                            if item.span == 1 =>
                        {
                            #[cfg(feature = "debug-logs")]
                            println!("encountered valid but empty scalar item at {item_index}");
                            page.set_entry_state(&mut self.hal, item_index as _, EntryMapState::Written)?;
                            page.used_entry_count += 1;
                        }
                        ItemType::Sized | ItemType::BlobData | ItemType::Blob => {
                            #[cfg(feature = "debug-logs")]
                            println!("encountered valid but EMPTY variable sized item at {item_index}");
                            let data_is_valid = match page.load_referenced_data(&mut self.hal, item_index, &item) {
                                Ok(data) => T::crc32(u32::MAX, &data) == unsafe { item.data.sized.crc },
                                Err(Error::CorruptedData) => false,
                                Err(e) => return Err(e),
                            };
                            if !data_is_valid {
                                page.set_entry_state_range(
                                    &mut self.hal,
                                    item_index..item_index + item.span,
                                    EntryMapState::Erased,
                                )?;
                                page.erased_entry_count += item.span;
                                // The whole span is counted above, so its payload entries must
                                // not be visited again: they now read as erased and would each
                                // be counted a second time, pushing the next free entry past
                                // the end of the page.
                                if item.span >= 2 {
                                    item_iter.nth((item.span - 2) as usize);
                                }
                                continue 'item_iter;
                            }
                            page.set_entry_state_range(
                                &mut self.hal,
                                item_index..item_index + item.span,
                                EntryMapState::Written,
                            )?;
                            page.used_entry_count += item.span;
                        }
                        _ => {
                            page.set_entry_state(&mut self.hal, item_index as _, EntryMapState::Erased)?;
                            page.erased_entry_count += 1;
                            continue 'item_iter;
                        }
                    }
                }
                EntryMapState::Written => {
                    let calculated_crc = item.calculate_crc32(T::crc32);
                    if item.crc != calculated_crc || !span_fits_page {
                        #[cfg(feature = "debug-logs")]
                        println!(
                            "CRC mismatch for item '{}', marking as erased",
                            slice_with_nullbytes_to_str(&item.key.0)
                        );
                        // The span was read from a header whose CRC just failed, or one that cannot
                        // fit the page, so it says nothing trustworthy about how many entries this
                        // item covers. Erasing a range on its word takes out whatever happens to
                        // follow, valid items included. Erase the header alone: the entries behind
                        // it are scanned like any other and reach this same
                        // check one at a time, which ends in the same place
                        // for a genuinely half written item without reaching
                        // past it.
                        //
                        // The entry has to be marked erased in the entry map, not only counted as
                        // such. Those entries behind it are payload, and their "span" byte is as
                        // often as not out of range, so this branch is
                        // where most of them end up. Left `Written`,
                        // `copy_items` later loads each one, fails its CRC and aborts the
                        // defragmentation with the source page stuck in `Freeing`; every `Nvs::new`
                        // after that resumes the copy and fails the same way.
                        page.set_entry_state_range(&mut self.hal, item_index..(item_index + 1), EntryMapState::Erased)?;
                        page.erased_entry_count += 1;
                        continue 'item_iter;
                    }

                    // ESP-IDF marks a header written before it writes the payload, so a header
                    // whose payload entries are not all written is a write the
                    // power cut short there - or here, if the single entry map
                    // write of this crate was torn. ESP-IDF erases such an item
                    // on load, and so does this: kept, it is a newer duplicate that reads as
                    // corrupt, and the duplicate cleanup would erase the older, intact value for
                    // it.
                    let payload_is_written = (item_index + 1..item_index + item.span)
                        .all(|index| page.get_entry_state(index) == EntryMapState::Written);
                    if !payload_is_written {
                        page.set_entry_state_range(
                            &mut self.hal,
                            item_index..item_index + item.span,
                            EntryMapState::Erased,
                        )?;
                        page.erased_entry_count += item.span;
                        if item.span >= 2 {
                            item_iter.nth((item.span - 2) as usize);
                        }
                        continue 'item_iter;
                    }

                    page.used_entry_count += item.span;
                }
            }

            // Continue for valid WRITTEN and formerly EMPTY entries
            #[cfg(feature = "debug-logs")]
            println!("item: {:?}", item);

            if item.namespace_index == 0 {
                self.namespaces.insert(item.key, unsafe { item.data.raw[0] });
                continue 'item_iter;
            }

            if item.type_ == ItemType::BlobIndex || item.type_ == ItemType::BlobData {
                let chunk_start = if item.type_ == ItemType::BlobIndex {
                    unsafe { VersionOffset::from(item.data.blob_index.chunk_start) }
                } else {
                    VersionOffset::from(item.chunk_index)
                };

                let key = (NamespaceIndex(item.namespace_index), chunk_start, item.key);
                let (index, observed) = blob_index.entry(key).or_default();
                if item.type_ == ItemType::BlobIndex {
                    *index = Some(BlobIndexEntryBlobIndexData {
                        item_index,
                        page_sequence: page.header.sequence,
                        size: unsafe { item.data.blob_index.size },
                        chunk_count: unsafe { item.data.blob_index.chunk_count },
                    });
                } else {
                    observed.chunk_count += 1;
                    observed.data_size += unsafe { item.data.sized.size } as u64;
                }
            }

            if item.type_ == ItemType::Blob {
                self.legacy_blobs.push((item.namespace_index, item.key));
            }

            page.item_hash_list.push(ItemHashListEntry {
                hash: item.calculate_hash(T::crc32),
                index: item_index,
            });

            // skip following items containing raw data
            if item.span >= 2 {
                item_iter.nth((item.span - 2) as usize);
            }
        }

        // The next free entry is taken to be the number of used and erased entries, which assumes
        // every entry before it is in use. A blank entry left EMPTY in the middle of the page
        // breaks that: the next free entry is then one already programmed, and every write
        // after is programmed on top of it, reporting success but readable neither itself
        // nor the item it landed on. Writes only ever append, so nothing will be written
        // into such a gap; mark it erased like the leftovers of a torn write, so the counts
        // match the entries' positions again. ESP-IDF takes the first EMPTY entry as the
        // next free one instead, and would write into the gap and then over what follows
        // it, so this keeps the page safe for it too.
        //
        // Non-blank EMPTY entries were settled by the scan above, so only blank ones are left, and
        // the scan did not count those. A power loss in here leaves the rest of the gap EMPTY, to
        // be marked on the next boot, or an entry torn between the two states, which the scan
        // erases and counts as it does any other.
        let last_used = (0..ENTRIES_PER_PAGE as u8)
            .rev()
            .find(|&index| page.get_entry_state(index) != EntryMapState::Empty);
        if let Some(last_used) = last_used {
            for index in 0..last_used {
                if page.get_entry_state(index) == EntryMapState::Empty {
                    #[cfg(feature = "debug-logs")]
                    println!("empty gap at {index} before used entry {last_used}, marking as erased");
                    page.set_entry_state(&mut self.hal, index as _, EntryMapState::Erased)?;
                    page.erased_entry_count += 1;
                }
            }
        }

        #[cfg(feature = "debug-logs")]
        println!("PGE {page:?}");

        Ok(LoadPageResult::Used(page))
    }
}
