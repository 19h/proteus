//! Self-Organizing Map (SOM) module for semantic space learning.

mod map;
mod neuron;
pub mod simd;
pub mod training;

pub use map::Som;
pub use neuron::Neuron;
pub use training::{SomTrainer, TrainingContext, WordEmbeddings, DEFAULT_EMBEDDING_DIM};
