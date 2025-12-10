//! Vendored wtpsplit - Universal sentence and paragraph segmentation.
//!
//! This module provides sentence boundary detection using ONNX models,
//! vendored from the wtpsplit-rs library.
//!
//! ## Features
//!
//! - **SaT (Segment any Text)**: Modern subword-based models using XLM-RoBERTa
//! - **WtP (Where's the Point)**: Legacy character-based models (deprecated)
//! - ONNX Runtime for efficient inference
//! - Automatic model downloading from HuggingFace Hub
//!
//! ## Example
//!
//! ```no_run
//! use proteus::wtpsplit::SaT;
//!
//! let mut sat = SaT::new("sat-3l-sm", None)?;
//! let sentences = sat.split("This is a test. Another sentence here.", None)?;
//! for sentence in sentences {
//!     println!("{}", sentence);
//! }
//! # Ok::<(), proteus::wtpsplit::Error>(())
//! ```

pub mod config;
pub mod constants;
pub mod error;
pub mod extract;
pub mod hub;
pub mod model;
pub mod sat;
pub mod utils;
pub mod wtp;

pub use config::ModelConfig;
pub use error::Error;
pub use extract::Weighting;
pub use sat::{SaT, SaTOptions};
#[allow(deprecated)]
pub use wtp::{WtP, WtPOptions};

/// Result type alias for wtpsplit operations
pub type Result<T> = std::result::Result<T, Error>;
