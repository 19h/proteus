//! Hardware acceleration for compute-intensive operations.
//!
//! This module provides GPU and optimized CPU implementations for
//! operations that benefit from hardware acceleration:
//!
//! - **Batched BMU search**: GPU-accelerated distance matrix computation
//! - **Matrix operations**: Optimized matmul for distance calculations
//!
//! # Architecture
//!
//! - GPU backend uses wgpu (Metal on macOS, Vulkan on Linux/Windows)
//! - Automatic fallback to CPU if GPU unavailable
//! - M-series Macs benefit from unified memory (no PCIe copy overhead)
//!
//! # Performance
//!
//! GPU acceleration is beneficial for:
//! - Batch sizes >= 1000 samples
//! - Large SOM grids (>= 128x128 neurons)
//!
//! For smaller workloads, CPU SIMD (in `som::simd`) is faster due to
//! GPU dispatch and transfer overhead.
//!
//! # Usage
//!
//! ```rust,no_run
//! use proteus::accel::GpuAccelerator;
//!
//! if GpuAccelerator::is_available() {
//!     let gpu = GpuAccelerator::new().expect("GPU init failed");
//!
//!     // Compute distances for 10k inputs against 16k neurons (128x128 grid)
//!     let distances = gpu.batched_distances(
//!         &inputs,   // [10000 × 100] flattened
//!         &weights,  // [16384 × 100] flattened
//!         10000,     // batch_size
//!         16384,     // num_neurons
//!         100,       // input_dim
//!     ).unwrap();
//!
//!     // Find BMU for each input
//!     let bmus = GpuAccelerator::find_bmu_indices(&distances, 10000, 16384);
//! }
//! ```

#[cfg(feature = "gpu")]
mod shaders;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::{GpuAccelerator, GpuError};

// Re-export a CPU-only stub when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
mod stub {
    /// Stub accelerator when GPU feature is disabled
    pub struct GpuAccelerator;

    impl GpuAccelerator {
        /// Check if GPU acceleration is available.
        /// Always returns false when the `gpu` feature is disabled.
        pub fn is_available() -> bool {
            false
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub use stub::GpuAccelerator;
