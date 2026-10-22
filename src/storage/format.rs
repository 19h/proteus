//! Binary format for efficient retina storage.
//!
//! The format is designed for fast random access and minimal storage overhead.
//!
//! ## Format Layout
//!
//! ```text
//! +------------------+
//! | Header (64 bytes)|
//! +------------------+
//! | Metadata (v4/v5) |
//! +------------------+
//! | Word Index Table |
//! | (variable)       |
//! +------------------+
//! | Fingerprint Data |
//! | (variable)       |
//! +------------------+
//! | String Pool      |
//! | (variable)       |
//! +------------------+
//! ```
//!
//! ### Header (64 bytes)
//! - Magic number (4 bytes): "PRET"
//! - Version (2 bytes)
//! - Flags (2 bytes)
//! - Grid dimension (4 bytes)
//! - Number of words (4 bytes)
//! - Index table offset (8 bytes)
//! - Fingerprint data offset (8 bytes)
//! - String pool offset (8 bytes)
//! - Reserved (24 bytes)
//!
//! ### Word Index Table
//! - Array of (string_offset: u32, string_len: u16, fp_offset: u32, fp_len: u16)
//!
//! ### Fingerprint Data
//! - Compressed RoaringBitmap data for each word
//!
//! ### String Pool
//! - Concatenated word strings (UTF-8)

use crate::error::{ProteusError, Result};
use crate::fingerprint::{Sdr, WordFingerprint};
use crate::index::InvertedIndex;
use bincode::Options;
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Magic number for Proteus retina files.
const MAGIC: &[u8; 4] = b"PRET";

/// Current format version.
const VERSION: u16 = 4;

/// Header size in bytes.
const HEADER_SIZE: usize = 64;

/// Explicit upper bound on decoded v5 metadata, including source annotations.
const MAX_SEMANTIC_METADATA_BYTES: u64 = 1024 * 1024 * 1024;

/// Retina file header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetinaHeader {
    /// Grid dimension.
    pub dimension: u32,
    /// Number of words.
    pub num_words: u32,
    /// Offset to the index table.
    pub index_offset: u64,
    /// Offset to fingerprint data.
    pub fingerprint_offset: u64,
    /// Offset to string pool.
    pub string_pool_offset: u64,
    /// Offset of an optional persisted inverted index.
    pub inverted_index_offset: u64,
    /// Format version.
    pub version: u16,
    /// Flags.
    pub flags: u16,
}

impl RetinaHeader {
    /// Creates a new header.
    pub fn new(dimension: u32, num_words: u32) -> Self {
        Self {
            dimension,
            num_words,
            index_offset: HEADER_SIZE as u64,
            fingerprint_offset: 0, // Set during write
            string_pool_offset: 0, // Set during write
            inverted_index_offset: 0,
            version: VERSION,
            flags: 0,
        }
    }

    /// Header with the memory-mappable posting-index flag.
    pub fn new_with_index(dimension: u32, num_words: u32) -> Self {
        let mut header = Self::new(dimension, num_words);
        header.flags = 2;
        header
    }
    /// Whether a legacy bincode index is present.
    pub fn has_inverted_index(&self) -> bool {
        self.flags & 1 != 0
    }
    /// Whether a memory-mappable posting index is present.
    pub fn has_mmap_index(&self) -> bool {
        self.flags & 2 != 0
    }
    /// Current v4/v5 metadata, or the earlier local version 2 metadata layout.
    fn has_metadata(&self) -> bool {
        self.version >= 4 || (self.version == 2 && self.flags == 0 && self.index_offset > 64)
    }

    /// Writes the header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_SIZE];

        // Magic number
        bytes[0..4].copy_from_slice(MAGIC);

        // Version
        bytes[4..6].copy_from_slice(&self.version.to_le_bytes());

        // Flags
        bytes[6..8].copy_from_slice(&self.flags.to_le_bytes());

        // Dimension
        bytes[8..12].copy_from_slice(&self.dimension.to_le_bytes());

        // Number of words
        bytes[12..16].copy_from_slice(&self.num_words.to_le_bytes());

        // Index offset
        bytes[16..24].copy_from_slice(&self.index_offset.to_le_bytes());

        // Fingerprint offset
        bytes[24..32].copy_from_slice(&self.fingerprint_offset.to_le_bytes());

        // String pool offset
        bytes[32..40].copy_from_slice(&self.string_pool_offset.to_le_bytes());

        bytes[40..48].copy_from_slice(&self.inverted_index_offset.to_le_bytes());
        // Reserved (bytes 48-63)
        bytes
    }

    /// Reads a header from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(ProteusError::InvalidRetinaFormat(
                "Header too short".to_string(),
            ));
        }

        // Check magic number
        if &bytes[0..4] != MAGIC {
            return Err(ProteusError::InvalidRetinaFormat(
                "Invalid magic number".to_string(),
            ));
        }

        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        let dimension = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let num_words = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        let index_offset = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let fingerprint_offset = u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let string_pool_offset = u64::from_le_bytes([
            bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
        ]);

        let inverted_index_offset = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
        if !matches!(version, 1..=5)
            || flags > 2
            || (version == 1 && flags != 0)
            || (flags == 1 && version != 2)
            || (flags == 2 && version < 3)
            || ((flags == 0) != (inverted_index_offset == 0))
        {
            return Err(invalid("Unsupported retina version or flags"));
        }
        if dimension == 0 || dimension.checked_mul(dimension).is_none() {
            return Err(invalid("Invalid grid dimension"));
        }
        Ok(Self {
            dimension,
            num_words,
            index_offset,
            fingerprint_offset,
            string_pool_offset,
            inverted_index_offset,
            version,
            flags,
        })
    }
}

/// Index entry for a word.
#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    /// Offset into the string pool.
    string_offset: u32,
    /// Length of the word string.
    string_len: u16,
    /// Offset into fingerprint data.
    fp_offset: u32,
    /// Length of fingerprint data.
    fp_len: u16,
}

impl IndexEntry {
    const SIZE: usize = 12;

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.string_offset.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.string_len.to_le_bytes());
        bytes[6..10].copy_from_slice(&self.fp_offset.to_le_bytes());
        bytes[10..12].copy_from_slice(&self.fp_len.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            string_offset: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            string_len: u16::from_le_bytes([bytes[4], bytes[5]]),
            fp_offset: u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]),
            fp_len: u16::from_le_bytes([bytes[10], bytes[11]]),
        }
    }
}

/// Persisted parameters and word metadata. Version 1 files have no metadata.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RetinaMetadata {
    /// Fingerprint density and text aggregation policy.
    pub fingerprint_config: crate::FingerprintConfig,
    /// Tokenization policy used at inference.
    pub text_config: crate::TextConfig,
    /// Resolved training parameters and diagnostics, if known.
    pub training_report: Option<crate::training::TrainingReport>,
    /// Provenance for a non-spatial angular retina.
    #[serde(default)]
    pub angular_report: Option<crate::training::angular::AngularReport>,
    /// Version 5 lexical senses, relation scores and attributed evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_model: Option<crate::semantic::SemanticModel>,
    /// Frequencies and POS tags, keyed in canonical word order.
    pub word_metadata: std::collections::BTreeMap<String, (f64, Vec<String>)>,
}

impl RetinaMetadata {
    fn validate(&self, header: &RetinaHeader) -> Result<()> {
        self.fingerprint_config.validate()?;
        if (header.version == 5) != self.semantic_model.is_some() {
            return Err(invalid("Semantic representation requires a version 5 file"));
        }
        if let Some(model) = &self.semantic_model {
            model.validate()?;
            if self.training_report.is_some()
                || self.angular_report.is_some()
                || model.report.config.dimension != header.dimension as usize
                || model.report.config.fingerprint != self.fingerprint_config
                || model.report.config.text != self.text_config
                || model.entries.len() != header.num_words as usize
                || model.entries.keys().ne(self.word_metadata.keys())
            {
                return Err(invalid("Semantic metadata does not match retina"));
            }
        }
        if self
            .word_metadata
            .values()
            .any(|(f, _)| !f.is_finite() || *f < 0.0 || *f > 1.0)
        {
            return Err(invalid("Invalid word frequency"));
        }
        if let Some(report) = &self.angular_report {
            report.config.validate()?;
            if self.training_report.is_some()
                || report.config.dimension != header.dimension as usize
                || report.config.text != self.text_config
                || report.config.fingerprint != self.fingerprint_config
                || report.vocabulary != header.num_words as usize
                || report.snippets == 0
                || report.features == 0
                || report.vector_dimension == 0
                || report.vector_dimension > report.features
                || report.vector_dimension > report.config.embedding_dimension
                || report.active_bits
                    != self
                        .fingerprint_config
                        .active_bit_limit(header.dimension * header.dimension)
                || report.duplicate_fingerprints > report.vocabulary
            {
                return Err(invalid("Angular metadata does not match retina"));
            }
        }
        if let Some(report) = &self.training_report {
            report.config.validate()?;
            if report.config.som.dimension != header.dimension as usize
                || report.config.fingerprint != self.fingerprint_config
                || report.config.text != self.text_config
                || report.vocabulary != header.num_words as usize
                || report.mapped_snippets > report.snippets
                || report.occupied_cells > (header.dimension * header.dimension) as usize
                || !report.quantization_error.is_finite()
                || report.quantization_error < 0.0
            {
                return Err(invalid("Training metadata does not match retina"));
            }
            if let Some(diagnostics) = &report.quality {
                if diagnostics.single_context_words > report.vocabulary
                    || diagnostics.duplicate_fingerprints > report.vocabulary
                    || !diagnostics.mean_active_bits.is_finite()
                    || diagnostics.mean_active_bits < 0.0
                    || diagnostics.mean_active_bits > (header.dimension * header.dimension) as f64
                    || !diagnostics.maximum_cell_fraction.is_finite()
                    || !(0.0..=1.0).contains(&diagnostics.maximum_cell_fraction)
                    || !diagnostics.effective_occupied_cells.is_finite()
                    || diagnostics.effective_occupied_cells < 1.0
                    || diagnostics.effective_occupied_cells > report.occupied_cells as f64 + 1e-8
                {
                    return Err(invalid("Invalid training quality diagnostics"));
                }
            }
        }
        Ok(())
    }
}

fn invalid(message: &str) -> ProteusError {
    ProteusError::InvalidRetinaFormat(message.into())
}

fn checked_slice(data: &[u8], start: u64, len: u64) -> Result<&[u8]> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid("Offset overflow"))?;
    let start = usize::try_from(start).map_err(|_| invalid("Offset out of range"))?;
    let end = usize::try_from(end).map_err(|_| invalid("Offset out of range"))?;
    data.get(start..end)
        .ok_or_else(|| invalid("Section outside file"))
}

fn read_posting(
    data: &[u8],
    header: &RetinaHeader,
    pos: u32,
) -> Result<crate::roaring::RoaringBitmap> {
    let grid = header.dimension * header.dimension;
    if pos >= grid {
        return Err(invalid("Posting position outside grid"));
    }
    let table = checked_slice(data, header.inverted_index_offset, u64::from(grid) * 8)?;
    let row = &table[pos as usize * 8..pos as usize * 8 + 8];
    let offset = u32::from_le_bytes(row[..4].try_into().unwrap());
    let len = u32::from_le_bytes(row[4..].try_into().unwrap());
    if len == 0 {
        return Ok(crate::roaring::RoaringBitmap::new());
    }
    let bytes = checked_slice(
        data,
        header.inverted_index_offset + u64::from(grid) * 8 + u64::from(offset),
        u64::from(len),
    )?;
    let mut reader = std::io::Cursor::new(bytes);
    let bitmap = crate::roaring::RoaringBitmap::deserialize_from(&mut reader)?;
    if reader.position() != bytes.len() as u64
        || bitmap.max().is_some_and(|id| id >= header.num_words)
    {
        return Err(invalid("Invalid posting list"));
    }
    Ok(bitmap)
}

struct Parsed<'a> {
    header: RetinaHeader,
    entries: Vec<IndexEntry>,
    words: Vec<String>,
    metadata: RetinaMetadata,
    data: &'a [u8],
    inverted_index: Option<InvertedIndex>,
}

impl<'a> Parsed<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        let header = RetinaHeader::from_bytes(data)?;
        let index_len = u64::from(header.num_words) * IndexEntry::SIZE as u64;
        if header.index_offset < HEADER_SIZE as u64
            || header.index_offset.checked_add(index_len) != Some(header.fingerprint_offset)
            || header.fingerprint_offset > header.string_pool_offset
        {
            return Err(invalid("Invalid section order or index size"));
        }
        let index = checked_slice(data, header.index_offset, index_len)?;
        checked_slice(
            data,
            header.fingerprint_offset,
            header.string_pool_offset - header.fingerprint_offset,
        )?;
        let string_end = if header.flags != 0 {
            header.inverted_index_offset
        } else {
            data.len() as u64
        };
        let string_len = string_end
            .checked_sub(header.string_pool_offset)
            .ok_or_else(|| invalid("Invalid string/index section order"))?;
        checked_slice(data, header.string_pool_offset, string_len)?;
        if header.flags != 0 {
            checked_slice(
                data,
                string_end,
                (data.len() as u64)
                    .checked_sub(string_end)
                    .ok_or_else(|| invalid("Index outside file"))?,
            )?;
        }
        let metadata = if header.has_metadata() {
            let bytes = checked_slice(
                data,
                HEADER_SIZE as u64,
                header.index_offset - HEADER_SIZE as u64,
            )?;
            let decoded;
            let bytes = if header.version == 5 {
                decoded = decode_semantic_metadata(bytes)?;
                decoded.as_slice()
            } else {
                bytes
            };
            let m: RetinaMetadata = serde_json::from_slice(bytes)
                .map_err(|e| invalid(&format!("Invalid metadata: {e}")))?;
            m.validate(&header)?;
            m
        } else {
            if header.index_offset != HEADER_SIZE as u64 {
                return Err(invalid("Invalid legacy index offset"));
            }
            RetinaMetadata::default()
        };
        let entries: Vec<_> = index
            .chunks_exact(IndexEntry::SIZE)
            .map(IndexEntry::from_bytes)
            .collect();
        let mut words = Vec::with_capacity(entries.len());
        let mut seen = std::collections::HashSet::new();
        for entry in &entries {
            if u64::from(entry.string_offset) + u64::from(entry.string_len) > string_len {
                return Err(invalid("Word outside string pool"));
            }
            let word = checked_slice(
                data,
                header.string_pool_offset + u64::from(entry.string_offset),
                u64::from(entry.string_len),
            )?;
            let word = std::str::from_utf8(word)
                .map_err(|_| invalid("Invalid UTF-8 word"))?
                .to_owned();
            if word.is_empty() || !seen.insert(word.clone()) {
                return Err(invalid("Empty or duplicate word"));
            }
            let fp_end = u64::from(entry.fp_offset) + u64::from(entry.fp_len);
            if fp_end > header.string_pool_offset - header.fingerprint_offset {
                return Err(invalid("Fingerprint outside data section"));
            }
            words.push(word);
        }
        if header.has_metadata()
            && (metadata.word_metadata.len() != words.len()
                || words
                    .iter()
                    .any(|w| !metadata.word_metadata.contains_key(w)))
        {
            return Err(invalid("Word metadata and index differ"));
        }
        let mut parsed = Self {
            header,
            entries,
            words,
            metadata,
            data,
            inverted_index: None,
        };
        // Validate bitmap structure and universe before exposing either reader API.
        for i in 0..parsed.entries.len() {
            parsed.fingerprint(i)?;
        }
        if parsed.header.flags != 0 {
            let mut fingerprints = HashMap::new();
            for (i, word) in parsed.words.iter().enumerate() {
                fingerprints.insert(
                    word.clone(),
                    WordFingerprint::new(
                        word.clone(),
                        &parsed.fingerprint(i)?.to_positions(),
                        parsed.header.dimension * parsed.header.dimension,
                    ),
                );
            }
            let persisted = if parsed.header.has_inverted_index() {
                let bytes = &data[parsed.header.inverted_index_offset as usize..];
                bincode::DefaultOptions::new()
                    .with_fixint_encoding()
                    .with_limit(bytes.len() as u64)
                    .reject_trailing_bytes()
                    .deserialize::<InvertedIndex>(bytes)?
            } else {
                let mut postings = Vec::new();
                for pos in 0..parsed.header.dimension * parsed.header.dimension {
                    postings.push(read_posting(data, &parsed.header, pos)?);
                }
                InvertedIndex::from_postings(parsed.words.clone(), postings)
            };
            if !persisted.matches_fingerprints(
                &fingerprints,
                parsed.header.dimension * parsed.header.dimension,
            ) {
                return Err(invalid("Persisted index does not match fingerprints"));
            }
            parsed.inverted_index = Some(persisted);
        }
        Ok(parsed)
    }

    fn fingerprint(&self, i: usize) -> Result<Sdr> {
        let entry = &self.entries[i];
        let bytes = checked_slice(
            self.data,
            self.header.fingerprint_offset + u64::from(entry.fp_offset),
            u64::from(entry.fp_len),
        )?;
        let mut reader = std::io::Cursor::new(bytes);
        let bitmap = crate::roaring::RoaringBitmap::deserialize_from(&mut reader)
            .map_err(|e| invalid(&format!("Invalid bitmap: {e}")))?;
        if reader.position() != bytes.len() as u64
            || bitmap
                .max()
                .is_some_and(|p| p >= self.header.dimension * self.header.dimension)
        {
            return Err(invalid(
                "Bitmap has trailing bytes or out-of-grid positions",
            ));
        }
        Ok(Sdr::from_bitmap(
            bitmap,
            self.header.dimension * self.header.dimension,
        ))
    }
}

/// Versioned, validated binary storage. New files preserve model metadata.
pub struct RetinaFormat;

impl RetinaFormat {
    /// Write a version 4 retina with default inference configuration.
    pub fn write<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, WordFingerprint>,
        dimension: u32,
    ) -> Result<()> {
        Self::write_with_metadata(path, fingerprints, dimension, RetinaMetadata::default())
    }

    /// Write atomically in canonical word order, preserving frequencies and configuration.
    pub fn write_with_metadata<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, WordFingerprint>,
        dimension: u32,
        metadata: RetinaMetadata,
    ) -> Result<()> {
        Self::write_with_metadata_and_index(path, fingerprints, dimension, metadata, None)
    }

    /// Persist a canonical mmap index along with default inference metadata.
    pub fn write_with_index<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, WordFingerprint>,
        dimension: u32,
        index: &InvertedIndex,
    ) -> Result<()> {
        Self::write_with_metadata_and_index(
            path,
            fingerprints,
            dimension,
            RetinaMetadata::default(),
            Some(index),
        )
    }

    /// Atomically persist metadata and an optional validated, canonical posting index.
    pub fn write_with_metadata_and_index<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, WordFingerprint>,
        dimension: u32,
        mut metadata: RetinaMetadata,
        index: Option<&InvertedIndex>,
    ) -> Result<()> {
        if dimension == 0 || dimension.checked_mul(dimension).is_none() {
            return Err(invalid("Invalid grid dimension"));
        }
        metadata.fingerprint_config.validate()?;
        let num_words = u32::try_from(fingerprints.len()).map_err(|_| invalid("Too many words"))?;
        let mut words: Vec<_> = fingerprints.keys().collect();
        words.sort();
        metadata.word_metadata.clear();
        let mut entries = Vec::with_capacity(words.len());
        let mut strings = Vec::new();
        let mut bitmaps = Vec::new();
        for word in words {
            let wf = &fingerprints[word];
            if word.is_empty()
                || &wf.word != word
                || wf.fingerprint.size() != dimension * dimension
                || wf.fingerprint.iter().any(|p| p >= dimension * dimension)
                || !wf.frequency.is_finite()
                || !(0.0..=1.0).contains(&wf.frequency)
            {
                return Err(invalid("Invalid word fingerprint or frequency"));
            }
            let mut bytes = Vec::new();
            wf.fingerprint.bitmap().serialize_into(&mut bytes)?;
            entries.push(IndexEntry {
                string_offset: u32::try_from(strings.len())
                    .map_err(|_| invalid("String pool exceeds 4 GiB"))?,
                string_len: u16::try_from(word.len())
                    .map_err(|_| invalid("Word exceeds 65535 bytes"))?,
                fp_offset: u32::try_from(bitmaps.len())
                    .map_err(|_| invalid("Bitmap section exceeds 4 GiB"))?,
                fp_len: u16::try_from(bytes.len())
                    .map_err(|_| invalid("Bitmap exceeds 65535 bytes"))?,
            });
            strings.extend_from_slice(word.as_bytes());
            bitmaps.extend_from_slice(&bytes);
            metadata
                .word_metadata
                .insert(word.clone(), (wf.frequency, wf.pos_tags.clone()));
        }
        let mut meta = serde_json::to_vec(&metadata).map_err(|e| invalid(&e.to_string()))?;
        let mut header = RetinaHeader::new(dimension, num_words);
        if metadata.semantic_model.is_some() {
            header.version = 5;
            if meta.len() as u64 > MAX_SEMANTIC_METADATA_BYTES {
                return Err(invalid("Semantic metadata exceeds the 1 GiB decoded limit"));
            }
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&meta)?;
            let compressed = encoder.finish()?;
            let mut envelope = Vec::with_capacity(compressed.len() + 8);
            envelope.extend_from_slice(&(meta.len() as u64).to_le_bytes());
            envelope.extend_from_slice(&compressed);
            meta = envelope;
        }
        header.index_offset = HEADER_SIZE as u64 + meta.len() as u64;
        header.fingerprint_offset =
            header.index_offset + entries.len() as u64 * IndexEntry::SIZE as u64;
        header.string_pool_offset = header.fingerprint_offset + bitmaps.len() as u64;
        metadata.validate(&header)?;
        let mut index_table = Vec::new();
        let mut index_data = Vec::new();
        if let Some(index) = index {
            if !index.matches_fingerprints(fingerprints, dimension * dimension) {
                return Err(invalid("Index does not match fingerprints"));
            }
            header.flags = 2;
            header.inverted_index_offset = header.string_pool_offset + strings.len() as u64;
            for posting in index.posting_lists_iter() {
                let offset = u32::try_from(index_data.len())
                    .map_err(|_| invalid("Posting index exceeds 4 GiB"))?;
                let mut bytes = Vec::new();
                if !posting.is_empty() {
                    posting.serialize_into(&mut bytes)?;
                }
                let len =
                    u32::try_from(bytes.len()).map_err(|_| invalid("Posting list too large"))?;
                index_table.extend_from_slice(&offset.to_le_bytes());
                index_table.extend_from_slice(&len.to_le_bytes());
                index_data.extend_from_slice(&bytes);
            }
        }
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
        tmp.write_all(&header.to_bytes())?;
        tmp.write_all(&meta)?;
        for entry in entries {
            tmp.write_all(&entry.to_bytes())?;
        }
        tmp.write_all(&bitmaps)?;
        tmp.write_all(&strings)?;
        tmp.write_all(&index_table)?;
        tmp.write_all(&index_data)?;
        tmp.as_file().sync_all()?;
        tmp.persist(path).map_err(|e| ProteusError::Io(e.error))?;
        Ok(())
    }

    /// Read version 1 or 2 fingerprints. Version 1 has unknown frequencies (zero).
    pub fn read<P: AsRef<Path>>(
        path: P,
    ) -> Result<(RetinaHeader, HashMap<String, WordFingerprint>)> {
        let (header, fingerprints, _) = Self::read_with_metadata(path)?;
        Ok((header, fingerprints))
    }

    /// Read all fingerprints and the saved inference/training parameters.
    pub fn read_with_metadata<P: AsRef<Path>>(
        path: P,
    ) -> Result<(
        RetinaHeader,
        HashMap<String, WordFingerprint>,
        RetinaMetadata,
    )> {
        let (header, fingerprints, metadata, _) = Self::read_full(path)?;
        Ok((header, fingerprints, metadata))
    }

    /// Read a persisted index, if present.
    pub fn read_with_index<P: AsRef<Path>>(
        path: P,
    ) -> Result<(
        RetinaHeader,
        HashMap<String, WordFingerprint>,
        Option<InvertedIndex>,
    )> {
        let (header, fingerprints, _, index) = Self::read_full(path)?;
        Ok((header, fingerprints, index))
    }

    /// Read all model state and validate that a persisted index matches the bitmaps.
    pub fn read_full<P: AsRef<Path>>(
        path: P,
    ) -> Result<(
        RetinaHeader,
        HashMap<String, WordFingerprint>,
        RetinaMetadata,
        Option<InvertedIndex>,
    )> {
        let data = std::fs::read(path)?;
        let parsed = Parsed::new(&data)?;
        let mut fingerprints = HashMap::with_capacity(parsed.entries.len());
        for (i, word) in parsed.words.iter().enumerate() {
            let (frequency, pos_tags) = parsed
                .metadata
                .word_metadata
                .get(word)
                .cloned()
                .unwrap_or_default();
            fingerprints.insert(
                word.clone(),
                WordFingerprint {
                    word: word.clone(),
                    fingerprint: parsed.fingerprint(i)?,
                    frequency,
                    pos_tags,
                },
            );
        }
        Ok((
            parsed.header,
            fingerprints,
            parsed.metadata,
            parsed.inverted_index,
        ))
    }

    /// Open a validated memory map.
    pub fn mmap<P: AsRef<Path>>(path: P) -> Result<MmappedRetina> {
        MmappedRetina::open(path)
    }
}

fn decode_semantic_metadata(bytes: &[u8]) -> Result<Vec<u8>> {
    let length_bytes = bytes
        .get(..8)
        .ok_or_else(|| invalid("Truncated semantic metadata envelope"))?;
    let decoded_length = u64::from_le_bytes(
        length_bytes
            .try_into()
            .map_err(|_| invalid("Invalid metadata length"))?,
    );
    if decoded_length == 0 || decoded_length > MAX_SEMANTIC_METADATA_BYTES {
        return Err(invalid(
            "Invalid or excessive decoded semantic metadata size",
        ));
    }
    let mut decoder = flate2::read::ZlibDecoder::new(&bytes[8..]);
    let mut decoded = Vec::new();
    decoder
        .by_ref()
        .take(decoded_length + 1)
        .read_to_end(&mut decoded)
        .map_err(|error| invalid(&format!("Invalid compressed semantic metadata: {error}")))?;
    if decoded.len() as u64 != decoded_length || decoder.total_in() != (bytes.len() - 8) as u64 {
        return Err(invalid(
            "Semantic metadata length or compressed stream boundary mismatch",
        ));
    }
    Ok(decoded)
}

/// A memory-mapped retina; bitmap values are decoded on demand after validation.
pub struct MmappedRetina {
    mmap: Mmap,
    header: RetinaHeader,
    entries: Vec<IndexEntry>,
    words: Vec<String>,
    word_index: HashMap<String, usize>,
    text_config: crate::TextConfig,
    legacy_index: Option<InvertedIndex>,
    semantic_model: Option<crate::semantic::SemanticModel>,
}

impl MmappedRetina {
    /// Validate the file before exposing indexed or word-based access.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the backing file must not be modified while mapped, as with all
        // memmap2 users. Atomic saves replace the file rather than modifying it.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let parsed = Parsed::new(&mmap)?;
        let word_index = parsed
            .words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
        let text_config = parsed.metadata.text_config;
        let semantic_model = parsed.metadata.semantic_model;
        let legacy_index = if parsed.header.has_inverted_index() {
            parsed.inverted_index
        } else {
            None
        };
        let header = parsed.header;
        let entries = parsed.entries;
        let words = parsed.words;
        Ok(Self {
            mmap,
            header,
            entries,
            words,
            word_index,
            text_config,
            legacy_index,
            semantic_model,
        })
    }
    /// Parsed header.
    pub fn header(&self) -> &RetinaHeader {
        &self.header
    }
    /// Vocabulary size.
    pub fn len(&self) -> usize {
        self.words.len()
    }
    /// Whether there are no words.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
    /// Exact, normalized-key lookup.
    pub fn contains(&self, word: &str) -> bool {
        self.word_index.contains_key(word)
    }
    /// Decode a fingerprint by word.
    pub fn get(&self, word: &str) -> Option<Sdr> {
        self.get_by_index(*self.word_index.get(word)?)
    }
    /// Decode a fingerprint by index.
    pub fn get_by_index(&self, idx: usize) -> Option<Sdr> {
        let entry = self.entries.get(idx)?;
        let bytes = checked_slice(
            &self.mmap,
            self.header.fingerprint_offset + u64::from(entry.fp_offset),
            u64::from(entry.fp_len),
        )
        .ok()?;
        let bitmap = crate::roaring::RoaringBitmap::deserialize_from(bytes).ok()?;
        Some(Sdr::from_bitmap(
            bitmap,
            self.header.dimension * self.header.dimension,
        ))
    }
    /// Read a word by index.
    pub fn get_word(&self, idx: usize) -> Option<String> {
        self.words.get(idx).cloned()
    }
    /// Iterate word keys.
    pub fn words(&self) -> impl Iterator<Item = &String> {
        self.word_index.keys()
    }
    /// Number of vocabulary entries.
    pub fn num_words(&self) -> usize {
        self.len()
    }
    /// Grid side length.
    pub fn dimension(&self) -> u32 {
        self.header.dimension
    }
    /// Whether the file includes a persisted posting index.
    pub fn has_inverted_index(&self) -> bool {
        self.header.flags != 0
    }
    /// Exact model-specific word ranking with deterministic ties. Legacy models
    /// use all posting candidates; grounded models score the retained graph.
    pub fn find_similar(&self, word: &str, k: usize) -> Option<Vec<(String, f64)>> {
        if let Some(model) = &self.semantic_model {
            return model
                .find_similar(
                    crate::semantic::WordQuery::new(word),
                    crate::semantic::Relation::Similarity,
                    k,
                )
                .ok()
                .map(|results| {
                    results
                        .into_iter()
                        .map(|result| (result.word, result.score))
                        .collect()
                });
        }
        if !self.has_inverted_index() {
            return None;
        }
        let tokens = crate::Tokenizer::new(self.text_config.clone()).tokenize_to_strings(word);
        if tokens.len() != 1 {
            return None;
        }
        let word = &tokens[0];
        let target = self.get(word)?;
        let mut candidates = std::collections::BTreeSet::new();
        for pos in target.iter() {
            if let Some(index) = &self.legacy_index {
                for w in index.words_at_position(pos) {
                    candidates.insert(*self.word_index.get(&w)?);
                }
            } else {
                for id in read_posting(&self.mmap, &self.header, pos).ok()?.iter() {
                    candidates.insert(id as usize);
                }
            }
        }
        candidates.remove(self.word_index.get(word)?);
        if candidates.len() < k {
            candidates.extend((0..self.len()).filter(|&i| self.words[i] != *word));
        }
        let mut results: Vec<_> = candidates
            .into_iter()
            .filter_map(|i| {
                self.get_by_index(i)
                    .map(|fp| (self.words[i].clone(), target.cosine_similarity(&fp)))
            })
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        results.truncate(k);
        Some(results)
    }

    /// Decoded semantic metadata; bitmap payloads remain mapped separately.
    pub fn semantic_model(&self) -> Option<&crate::semantic::SemanticModel> {
        self.semantic_model.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_fingerprints() -> HashMap<String, WordFingerprint> {
        let mut fps = HashMap::new();
        fps.insert(
            "hello".to_string(),
            WordFingerprint::new("hello".to_string(), &[1, 5, 10, 100], 16384),
        );
        fps.insert(
            "world".to_string(),
            WordFingerprint::new("world".to_string(), &[2, 6, 20, 200], 16384),
        );
        fps.insert(
            "test".to_string(),
            WordFingerprint::new("test".to_string(), &[3, 7, 30, 300], 16384),
        );
        fps
    }

    #[test]
    fn test_header_roundtrip() {
        let header = RetinaHeader::new(128, 1000);
        let bytes = header.to_bytes();
        let recovered = RetinaHeader::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.dimension, 128);
        assert_eq!(recovered.num_words, 1000);
        assert_eq!(recovered.version, VERSION);
    }

    #[test]
    fn test_write_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.retina");

        let fps = create_test_fingerprints();
        RetinaFormat::write(&path, &fps, 128).unwrap();

        let (header, recovered) = RetinaFormat::read(&path).unwrap();

        assert_eq!(header.dimension, 128);
        assert_eq!(header.num_words, 3);
        assert_eq!(recovered.len(), 3);

        let hello = recovered.get("hello").unwrap();
        assert!(hello.fingerprint.contains(1));
        assert!(hello.fingerprint.contains(100));
    }

    #[test]
    fn test_mmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.retina");

        let fps = create_test_fingerprints();
        RetinaFormat::write(&path, &fps, 128).unwrap();

        let mmap = RetinaFormat::mmap(&path).unwrap();

        assert_eq!(mmap.len(), 3);
        assert!(mmap.contains("hello"));
        assert!(mmap.contains("world"));

        let hello = mmap.get("hello").unwrap();
        assert!(hello.contains(1));
        assert!(hello.contains(100));
    }
}
