//! Blob handling types and data structures.
//!
//! This module contains types used for tracking and managing multi-chunk
//! blob storage in NVS.

use alloc::collections::BTreeMap;

use crate::Key;
use crate::types::{
    NamespaceIndex,
    VersionOffset,
};

/// Key for the blob index: (namespace, version_offset, key).
pub(crate) type BlobIndexKey = (NamespaceIndex, VersionOffset, Key);

/// Value for the blob index: (optional blob index data, observed chunk data).
pub(crate) type BlobIndexValue = (Option<BlobIndexEntryBlobIndexData>, BlobObservedData);

/// The value will only have multiple entries if we are interrupted while writing an updated blob.
/// Since we clean up on init, there are at most two.
pub(crate) type BlobIndex = BTreeMap<BlobIndexKey, BlobIndexValue>;

/// The blob data chunks observed across all pages, summed up. Summed wide, so that a corrupt count
/// cannot overflow.
#[derive(Default)]
#[cfg_attr(feature = "debug-logs", derive(Debug))]
pub(crate) struct BlobObservedData {
    pub(crate) chunk_count: u32,
    pub(crate) data_size: u64,
}

/// Blob index entry metadata from the BlobIndex item.
#[cfg_attr(feature = "debug-logs", derive(Debug))]
pub(crate) struct BlobIndexEntryBlobIndexData {
    pub(crate) item_index: u8,
    pub(crate) page_sequence: u32,
    pub(crate) size: u32,
    pub(crate) chunk_count: u8,
}
