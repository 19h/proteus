//! Error types for the Proteus Semantic Folding engine.

use std::path::PathBuf;
use thiserror::Error;

/// The main error type for Proteus operations.
#[derive(Error, Debug)]
pub enum ProteusError {
    /// Error during text processing.
    #[error("Text processing error: {0}")]
    TextProcessing(String),

    /// Error during SOM training or inference.
    #[error("SOM error: {0}")]
    Som(String),

    /// Error during fingerprint generation.
    #[error("Fingerprint error: {0}")]
    Fingerprint(String),

    /// Error during storage operations.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Error during similarity computation.
    #[error("Similarity error: {0}")]
    Similarity(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid configuration.
    #[error("Configuration error: {0}")]
    Config(String),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// Invalid retina format.
    #[error("Invalid retina format: {0}")]
    InvalidRetinaFormat(String),

    /// Word not found in vocabulary.
    #[error("Word not found in vocabulary: {0}")]
    WordNotFound(String),

    /// Index out of bounds.
    #[error("Index out of bounds: {index} >= {max}")]
    IndexOutOfBounds {
        /// The index that was out of bounds.
        index: usize,
        /// The maximum allowed index.
        max: usize,
    },

    /// Empty input.
    #[error("Empty input: {0}")]
    EmptyInput(String),

    /// Training error.
    #[error("Training error: {0}")]
    Training(String),
}

/// Result type alias for Proteus operations.
pub type Result<T> = std::result::Result<T, ProteusError>;

impl From<bincode::Error> for ProteusError {
    fn from(err: bincode::Error) -> Self {
        ProteusError::Serialization(err.to_string())
    }
}
