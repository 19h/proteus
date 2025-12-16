//! Inverted index and semantic lookup for fast similarity search.
//!
//! This module provides multiple indexing strategies:
//!
//! - **Inverted Index**: Traditional posting-list based lookup for exact matches
//! - **HNSW**: Hierarchical Navigable Small World graph for O(log n) ANN search
//! - **Product Quantization**: Compressed vector storage with fast distance computation
//! - **LSH**: Locality-Sensitive Hashing for sublinear approximate search
//! - **Semantic Lookup**: Cortical.io-style position-based semantic search

mod inverted;
mod mmap;
pub mod topology;
pub mod semantic_lookup;
pub mod hnsw;
pub mod pq;
pub mod lsh;

pub use inverted::{InvertedIndex, IndexStats};
pub use mmap::{MmappedIndexWriter, MmappedInvertedIndex};
pub use topology::ToroidalGrid;
pub use semantic_lookup::{
    SemanticLookupEngine, SemanticLookupConfig, LookupResult,
    LookupFactors, PositionStats, SemanticCluster,
};
pub use hnsw::{HnswIndex, HnswConfig, HnswStats};
pub use pq::{ProductQuantizer, PqConfig, OptimizedProductQuantizer};
pub use lsh::{
    SimHash, MinHash, BitSamplingLsh, LshIndex, LshHasher, LshStats,
    create_minhash_index, create_bitsampling_index,
};
