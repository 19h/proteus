//! Storage module for efficient binary format and persistence.

mod format;
mod retina;

pub use format::{RetinaFormat, RetinaHeader};
pub use retina::Retina;
