//! Storage module for efficient binary format and persistence.

mod format;
mod retina;

pub use format::{MmappedRetina, RetinaFormat, RetinaHeader};
pub use retina::Retina;
