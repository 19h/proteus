//! Fingerprint generation module for SDR representations.
//!
//! This module provides:
//!
//! - **SDR**: Sparse Distributed Representations (binary fingerprints)
//! - **Word/Text Fingerprinting**: Generate SDRs from text
//! - **HDC**: Hyperdimensional Computing operations for vector symbolic architectures

mod sdr;
mod text;
mod word;
pub mod hdc;

pub use sdr::Sdr;
pub use text::TextFingerprinter;
pub use word::{WordFingerprint, WordFingerprinter};
pub use hdc::{
    Hypervector, ItemMemory, SequenceEncoder, NgramEncoder, AnalogySolver,
};
