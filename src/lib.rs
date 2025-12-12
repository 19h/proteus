//! # Proteus - Semantic Folding Engine
//!
//! Proteus is a Rust implementation of Semantic Folding, a technique for
//! representing natural language as Sparse Distributed Representations (SDRs).
//!
//! ## Overview
//!
//! Semantic Folding represents words and texts as binary fingerprints on a
//! 2D semantic space. Each position in the space represents a learned semantic
//! context, and words are represented by the set of contexts they appear in.
//!
//! ## Key Features
//!
//! - **Self-Organizing Map (SOM)** for learning semantic space topology
//! - **Sparse Distributed Representations (SDRs)** for efficient storage and operations
//! - **Fast similarity search** via inverted index
//! - **Multiple similarity measures**: Cosine, Jaccard, Overlap
//! - **Efficient binary format** for persistence
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use proteus::{Config, Retina, TextFingerprinter};
//!
//! // Load a pre-trained retina
//! let retina = Retina::load("english.retina")?;
//!
//! // Compute text fingerprint
//! let fp = retina.fingerprint_text("Hello, world!");
//!
//! // Find similar words
//! let similar = retina.find_similar_words("king", 10)?;
//!
//! // Compute text similarity
//! let sim = retina.text_similarity("cat", "dog");
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`text`] - Text processing and tokenization
//! - [`som`] - Self-Organizing Map implementation
//! - [`fingerprint`] - SDR and fingerprint generation
//! - [`storage`] - Binary format and persistence
//! - [`index`] - Inverted index for similarity search
//! - [`similarity`] - Similarity measures
//!
//! ## Training a New Retina
//!
//! ```rust,ignore
//! use proteus::{Config, SomTrainer, Som, WordFingerprinter, Retina};
//!
//! // Create configuration
//! let config = Config::default();
//!
//! // Initialize SOM
//! let mut som = Som::new(&config.som);
//!
//! // Train on corpus
//! let mut trainer = SomTrainer::new(config.som.clone());
//! let word_to_bmus = trainer.train(&mut som, &training_contexts)?;
//!
//! // Generate fingerprints
//! let mut fingerprinter = WordFingerprinter::new(config.fingerprint.clone(), som.total_neurons() as u32);
//! fingerprinter.create_fingerprints(&word_to_bmus, None);
//!
//! // Create and save retina
//! let retina = Retina::with_index(fingerprinter.into_fingerprints(), 128, config.fingerprint);
//! retina.save("my_retina.retina")?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::needless_return)]
#![feature(portable_simd)]

pub mod accel;
pub mod config;
pub mod error;
pub mod fingerprint;
pub mod index;
pub mod roaring;
pub mod segmentation;
pub mod similarity;
pub mod som;
pub mod storage;
pub mod text;
pub mod wtpsplit;

// Re-export commonly used types
pub use config::{Config, FingerprintConfig, SomConfig, StorageConfig, TextConfig};
pub use error::{ProteusError, Result};
pub use fingerprint::{Sdr, TextFingerprinter, WordFingerprint, WordFingerprinter};
pub use index::InvertedIndex;
pub use similarity::{CosineSimilarity, JaccardSimilarity, OverlapSimilarity, SimilarityMeasure, SimilarityType};
pub use som::{Neuron, Som, SomTrainer, TrainingContext};
pub use storage::{Retina, RetinaFormat, RetinaHeader};
pub use text::{Normalizer, SentenceSegmenter, Token, Tokenizer};
pub use segmentation::{SegmentationConfig, SegmentationResult, SemanticSegment, SemanticSegmenter};

// GPU acceleration (requires "gpu" feature)
pub use accel::GpuAccelerator;

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default grid dimension.
pub const DEFAULT_DIMENSION: usize = 128;

/// Default grid size (dimension^2).
pub const DEFAULT_GRID_SIZE: usize = DEFAULT_DIMENSION * DEFAULT_DIMENSION;

/// Default sparsity (2%).
pub const DEFAULT_SPARSITY: f64 = 0.02;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_DIMENSION, 128);
        assert_eq!(DEFAULT_GRID_SIZE, 16384);
        assert!((DEFAULT_SPARSITY - 0.02).abs() < 1e-10);
    }
}
