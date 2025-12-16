//! Inverted index and semantic lookup for fast similarity search.

mod inverted;
mod mmap;
pub mod topology;
pub mod semantic_lookup;

pub use inverted::{InvertedIndex, IndexStats};
pub use mmap::{MmappedIndexWriter, MmappedInvertedIndex};
pub use topology::ToroidalGrid;
pub use semantic_lookup::{
    SemanticLookupEngine, SemanticLookupConfig, LookupResult,
    LookupFactors, PositionStats, SemanticCluster,
};
