//! Semantic text segmentation.
//!
//! This module provides semantic segmentation of text into coherent sections
//! based on topic shifts detected via fingerprint similarity analysis.
//!
//! The algorithm is based on TextTiling (Hearst, 1997) adapted for
//! Sparse Distributed Representations (SDRs).

mod semantic;

pub use semantic::{
    SegmentationConfig, SegmentationResult, SemanticSegment, SemanticSegmenter,
};
