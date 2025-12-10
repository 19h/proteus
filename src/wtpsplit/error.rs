//! Error types for wtpsplit

use thiserror::Error;

/// Main error type for wtpsplit operations
#[derive(Error, Debug)]
pub enum Error {
    /// ONNX Runtime error
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Hub download error
    #[error("Hub download error: {0}")]
    HubDownload(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Inference error
    #[error("Inference error: {0}")]
    Inference(String),
}
