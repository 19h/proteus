//! Self-Organizing Map (SOM) module for semantic space learning.
//!
//! This module provides multiple SOM training strategies:
//!
//! - **Online Training**: Traditional sequential updates (training.rs)
//! - **SIMD-Accelerated**: Architecture-aware vectorized operations (simd.rs)
//! - **Mini-Batch Training**: Modern batch updates with Adam optimizer (batch.rs)

mod map;
mod neuron;
pub mod simd;
pub mod training;
pub mod batch;

pub use map::Som;
pub use neuron::Neuron;
pub use training::{SomTrainer, TrainingContext, WordEmbeddings, DEFAULT_EMBEDDING_DIM};
pub use batch::{BatchSomTrainer, BatchSomConfig, TrainingMetrics, GrowingSom};
