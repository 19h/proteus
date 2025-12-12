//! GPU-accelerated operations for compute-intensive tasks.
//!
//! Provides GPU acceleration via wgpu (Metal on macOS, Vulkan on Linux/Windows).
//! Optimized for batched operations to amortize PCIe transfer overhead.
//!
//! # Performance Characteristics
//!
//! - M-series Macs benefit from unified memory (zero-copy transfers)
//! - Batched matmul achieves 10-50x speedup for large workloads
//! - Minimum batch size: 1000 samples (below this, CPU SIMD is faster)

use super::shaders;
use std::sync::Arc;

/// Errors that can occur during GPU operations.
#[derive(Debug, Clone)]
pub enum GpuError {
    /// No GPU adapter found
    NoAdapter,
    /// Failed to create device
    DeviceCreation(String),
    /// Buffer mapping failed
    BufferMapping(String),
    /// Dimension mismatch
    DimensionMismatch {
        /// Expected dimension size
        expected: usize,
        /// Actual dimension size
        actual: usize,
    },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "No GPU adapter found"),
            GpuError::DeviceCreation(msg) => write!(f, "Device creation failed: {}", msg),
            GpuError::BufferMapping(msg) => write!(f, "Buffer mapping failed: {}", msg),
            GpuError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU accelerator for batched BMU search and distance computations.
///
/// This struct manages GPU resources and provides methods for
/// batched distance matrix computation, which is the core operation
/// in BMU (Best Matching Unit) search for SOM training.
pub struct GpuAccelerator {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Maximum storage buffer binding size (typically 2GB on Vulkan, larger on Metal)
    max_buffer_binding_size: u64,
    /// Cached weights buffer (stays on GPU across batches)
    cached_weights: Option<CachedWeights>,
}

/// Cached neuron weights and norms on GPU.
struct CachedWeights {
    weights_buffer: wgpu::Buffer,
    weight_norms_buffer: wgpu::Buffer,
    num_neurons: usize,
    input_dim: usize,
}

impl GpuAccelerator {
    /// Check if GPU acceleration is available.
    pub fn is_available() -> bool {
        pollster::block_on(Self::is_available_async())
    }

    /// Check if GPU acceleration is available (async).
    async fn is_available_async() -> bool {
        let instance = wgpu::Instance::default();
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .is_some()
    }

    /// Create a new GPU accelerator.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    /// Create a new GPU accelerator (async).
    async fn new_async() -> Result<Self, GpuError> {
        // Prefer Vulkan on Linux (OpenGL/EGL has 2GB buffer limit)
        // Metal on macOS, DX12 on Windows
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        // Get the adapter's actual limits - use the hardware's full capabilities
        // M-series Macs: unified memory allows huge buffers (36GB+ on M2 Max)
        // NVIDIA GPUs: typically have large VRAM (8-24GB+)
        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Proteus GPU Accelerator"),
                    required_features: wgpu::Features::empty(),
                    // Request the adapter's full limits for all buffer-related settings
                    required_limits: adapter_limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e: wgpu::RequestDeviceError| GpuError::DeviceCreation(e.to_string()))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            max_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size as u64,
            cached_weights: None,
        })
    }

    /// Returns the maximum recommended batch size for fused BMU search.
    ///
    /// With the fused kernel, we don't materialize the full distance matrix,
    /// so the limiting factor is the input buffer size, not output.
    ///
    /// Input buffer: batch_size × input_dim × sizeof(f32)
    ///
    /// Note: wgpu's `max_storage_buffer_binding_size` is a u32, capping it at ~4GB
    /// even on hardware that supports more. See https://github.com/gfx-rs/wgpu/issues/2337
    pub fn recommended_batch_size(&self, num_neurons: usize) -> usize {
        // With fused kernel, we're limited by input buffer not output
        // Input buffer: batch_size × input_dim × sizeof(f32)
        // Assume input_dim ~ 100 (typical for SOM), use 80% of max
        let assumed_input_dim = 100usize;
        let max_input_bytes = (self.max_buffer_binding_size as f64 * 0.8) as u64;
        let bytes_per_sample = (assumed_input_dim * std::mem::size_of::<f32>()) as u64;
        let max_batch = max_input_bytes / bytes_per_sample;

        // Also consider that each thread processes one sample and iterates over all neurons
        // Too large batches can cause GPU timeouts, so cap at a reasonable level
        // 500k samples × 16k neurons × 100 dims = 800 billion FLOPs per batch
        let _ = num_neurons; // Acknowledge the parameter (used for old distance-matrix approach)

        // Clamp to reasonable range
        // Lower bound: 10k (below this, GPU overhead dominates)
        // Upper bound: 500k (avoid GPU timeouts)
        (max_batch as usize).clamp(10_000, 500_000)
    }

    /// Returns the maximum buffer binding size in bytes.
    ///
    /// Note: This is capped at ~4GB due to wgpu using u32 for this limit,
    /// even on hardware that supports larger buffers.
    pub fn max_buffer_binding_size(&self) -> u64 {
        self.max_buffer_binding_size
    }

    /// Cache neuron weights on GPU for faster repeated BMU searches.
    ///
    /// Call this once before processing batches. The weights and their
    /// precomputed norms will stay on GPU memory, avoiding repeated uploads.
    pub fn cache_weights(&mut self, weights: &[f32], num_neurons: usize, input_dim: usize) -> Result<(), GpuError> {
        if weights.len() != num_neurons * input_dim {
            return Err(GpuError::DimensionMismatch {
                expected: num_neurons * input_dim,
                actual: weights.len(),
            });
        }

        // Create weights buffer
        let weights_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cached Weights Buffer"),
            size: (weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&weights_buffer, 0, bytemuck::cast_slice(weights));

        // Compute weight norms
        let weight_norms = pollster::block_on(self.compute_row_norms_async(weights, num_neurons, input_dim))?;

        // Create weight norms buffer
        let weight_norms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cached Weight Norms Buffer"),
            size: (weight_norms.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&weight_norms_buffer, 0, bytemuck::cast_slice(&weight_norms));

        self.cached_weights = Some(CachedWeights {
            weights_buffer,
            weight_norms_buffer,
            num_neurons,
            input_dim,
        });

        Ok(())
    }

    /// Clear cached weights from GPU memory.
    pub fn clear_cached_weights(&mut self) {
        self.cached_weights = None;
    }

    /// Apply batched SOM weight updates on GPU.
    ///
    /// This function aggregates all weight updates and applies them in a single GPU pass.
    /// Updates are pre-aggregated by neuron to handle the case where multiple samples
    /// update the same neuron.
    ///
    /// # Arguments
    ///
    /// * `weights` - Mutable reference to neuron weights (num_neurons × weight_dim), row-major
    /// * `inputs` - The input vectors that triggered updates
    /// * `updates` - List of (neuron_idx, input_idx, influence) tuples
    /// * `num_neurons` - Number of neurons
    /// * `weight_dim` - Dimension of weight vectors
    ///
    /// The update formula is:
    ///   weights[neuron] = (1 - total_influence) * weights[neuron] + sum(influence_i * input_i)
    pub fn apply_weight_updates(
        &self,
        weights: &mut [f32],
        inputs: &[f32],
        updates: &[(usize, usize, f32)], // (neuron_idx, input_idx, influence)
        num_neurons: usize,
        weight_dim: usize,
    ) -> Result<(), GpuError> {
        if updates.is_empty() {
            return Ok(());
        }

        // Pre-aggregate updates by neuron on CPU
        // For each neuron: accumulate total_influence and weighted_input_sum
        let mut neuron_influences = vec![0.0f32; num_neurons];
        let mut neuron_weighted_inputs = vec![0.0f32; num_neurons * weight_dim];

        for &(neuron_idx, input_idx, influence) in updates {
            neuron_influences[neuron_idx] += influence;
            let input_start = input_idx * weight_dim;
            let weighted_start = neuron_idx * weight_dim;
            for d in 0..weight_dim {
                neuron_weighted_inputs[weighted_start + d] += influence * inputs[input_start + d];
            }
        }

        pollster::block_on(self.apply_weight_updates_async(
            weights,
            &neuron_influences,
            &neuron_weighted_inputs,
            num_neurons,
            weight_dim,
        ))
    }

    /// Apply aggregated weight updates on GPU (async).
    async fn apply_weight_updates_async(
        &self,
        weights: &mut [f32],
        neuron_influences: &[f32],
        neuron_weighted_inputs: &[f32],
        num_neurons: usize,
        weight_dim: usize,
    ) -> Result<(), GpuError> {
        // Create buffers
        let weights_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weights Buffer"),
            size: (weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let influences_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Influences Buffer"),
            size: (neuron_influences.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let weighted_inputs_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weighted Inputs Buffer"),
            size: (neuron_weighted_inputs.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            num_neurons: u32,
            weight_dim: u32,
        }

        let dims = Dimensions {
            num_neurons: num_neurons as u32,
            weight_dim: weight_dim as u32,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Update Dimensions"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        self.queue.write_buffer(&weights_buffer, 0, bytemuck::cast_slice(weights));
        self.queue.write_buffer(&influences_buffer, 0, bytemuck::cast_slice(neuron_influences));
        self.queue.write_buffer(&weighted_inputs_buffer, 0, bytemuck::cast_slice(neuron_weighted_inputs));
        self.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create shader
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Weight Update Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::AGGREGATED_WEIGHT_UPDATE_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Weight Update Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Weight Update Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: influences_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weighted_inputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Weight Update Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Weight Update Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Dispatch - workgroup size is 16x16, so we need ceil(num_neurons/16) x ceil(weight_dim/16) workgroups
        let workgroups_x = (num_neurons as u32).div_ceil(16);
        let workgroups_y = (weight_dim as u32).div_ceil(16);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Weight Update Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Weight Update Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy updated weights back to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weight Staging Buffer"),
            size: (weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &weights_buffer,
            0,
            &staging_buffer,
            0,
            (weights.len() * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read back updated weights
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .receive()
            .await
            .ok_or_else(|| GpuError::BufferMapping("Channel receive failed".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        {
            let data = buffer_slice.get_mapped_range();
            let updated_weights: &[f32] = bytemuck::cast_slice(&data);
            weights.copy_from_slice(updated_weights);
        }

        staging_buffer.unmap();

        Ok(())
    }

    /// Find BMU indices using cached weights (fastest path).
    ///
    /// Requires `cache_weights` to be called first.
    /// Only uploads input data; weights stay on GPU.
    pub fn find_bmus_cached(&self, inputs: &[f32], batch_size: usize) -> Result<Vec<usize>, GpuError> {
        let cached = self.cached_weights.as_ref().ok_or_else(|| {
            GpuError::DeviceCreation("Weights not cached. Call cache_weights first.".to_string())
        })?;

        if inputs.len() != batch_size * cached.input_dim {
            return Err(GpuError::DimensionMismatch {
                expected: batch_size * cached.input_dim,
                actual: inputs.len(),
            });
        }

        pollster::block_on(self.find_bmus_cached_async(inputs, batch_size, cached))
    }

    /// Find BMUs using cached weights (async).
    async fn find_bmus_cached_async(
        &self,
        inputs: &[f32],
        batch_size: usize,
        cached: &CachedWeights,
    ) -> Result<Vec<usize>, GpuError> {
        let input_dim = cached.input_dim;
        let num_neurons = cached.num_neurons;

        // Compute input norms
        let input_norms = self.compute_row_norms_async(inputs, batch_size, input_dim).await?;

        // Create input buffers
        let inputs_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Inputs Buffer"),
            size: (inputs.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_norms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Norms Buffer"),
            size: (input_norms.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BMU Indices Buffer"),
            size: (batch_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            batch_size: u32,
            num_neurons: u32,
            input_dim: u32,
            _padding: u32,
        }

        let dims = Dimensions {
            batch_size: batch_size as u32,
            num_neurons: num_neurons as u32,
            input_dim: input_dim as u32,
            _padding: 0,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        self.queue.write_buffer(&inputs_buffer, 0, bytemuck::cast_slice(inputs));
        self.queue.write_buffer(&input_norms_buffer, 0, bytemuck::cast_slice(&input_norms));
        self.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create shader and pipeline
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fused BMU Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::FUSED_BMU_SHADER.into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fused BMU Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused BMU Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cached.weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_norms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cached.weight_norms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused BMU Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fused BMU Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (batch_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Fused BMU Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fused BMU Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Workgroup size is 256 (matches FUSED_BMU_SHADER @workgroup_size)
            compute_pass.dispatch_workgroups((batch_size as u32).div_ceil(256), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .receive()
            .await
            .ok_or_else(|| GpuError::BufferMapping("Channel receive failed".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let result: Vec<usize> = {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&data).iter().map(|&x| x as usize).collect()
        };

        staging_buffer.unmap();

        Ok(result)
    }

    /// Compute batched squared Euclidean distances for BMU search.
    ///
    /// Given a batch of input vectors and neuron weights, computes all pairwise
    /// squared distances efficiently on the GPU.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Flattened input vectors (batch_size × input_dim), row-major
    /// * `weights` - Flattened neuron weights (num_neurons × input_dim), row-major
    /// * `batch_size` - Number of input vectors
    /// * `num_neurons` - Number of neurons in the SOM grid
    /// * `input_dim` - Dimension of each vector
    ///
    /// # Returns
    ///
    /// Flattened distance matrix (batch_size × num_neurons), row-major.
    /// `distances[i * num_neurons + j]` = squared distance from input i to neuron j.
    ///
    /// # Performance
    ///
    /// This is the key operation for GPU BMU search. Instead of computing:
    /// ```text
    /// for each input:
    ///     for each neuron:
    ///         distance = sum((input - neuron)^2)  // input_dim operations
    /// ```
    ///
    /// We use matrix operations:
    /// ```text
    /// ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
    /// distances = input_norms + neuron_norms.T - 2 * (inputs @ weights.T)
    /// ```
    ///
    /// This turns O(batch × neurons × dim) scalar ops into:
    /// - 2 norm computations (parallelizable)
    /// - 1 matrix multiplication (GPU-optimized)
    pub fn batched_distances(
        &self,
        inputs: &[f32],
        weights: &[f32],
        batch_size: usize,
        num_neurons: usize,
        input_dim: usize,
    ) -> Result<Vec<f32>, GpuError> {
        // Validate dimensions
        if inputs.len() != batch_size * input_dim {
            return Err(GpuError::DimensionMismatch {
                expected: batch_size * input_dim,
                actual: inputs.len(),
            });
        }
        if weights.len() != num_neurons * input_dim {
            return Err(GpuError::DimensionMismatch {
                expected: num_neurons * input_dim,
                actual: weights.len(),
            });
        }

        pollster::block_on(self.batched_distances_async(
            inputs, weights, batch_size, num_neurons, input_dim,
        ))
    }

    /// Find BMU indices for a batch of inputs (fully on GPU).
    ///
    /// This is more efficient than `batched_distances` + `find_bmu_indices` because:
    /// 1. It avoids materializing the full distance matrix (saves memory bandwidth)
    /// 2. Distance computation and argmin are fused in one GPU pass
    /// 3. Only the BMU indices (u32 per sample) are transferred back, not distances
    ///
    /// # Arguments
    ///
    /// * `inputs` - Flattened input vectors (batch_size × input_dim), row-major
    /// * `weights` - Flattened neuron weights (num_neurons × input_dim), row-major
    /// * `batch_size` - Number of input vectors
    /// * `num_neurons` - Number of neurons in the SOM grid
    /// * `input_dim` - Dimension of each vector
    ///
    /// # Returns
    ///
    /// Vector of BMU indices, one per input sample.
    pub fn find_bmus_gpu(
        &self,
        inputs: &[f32],
        weights: &[f32],
        batch_size: usize,
        num_neurons: usize,
        input_dim: usize,
    ) -> Result<Vec<usize>, GpuError> {
        // Validate dimensions
        if inputs.len() != batch_size * input_dim {
            return Err(GpuError::DimensionMismatch {
                expected: batch_size * input_dim,
                actual: inputs.len(),
            });
        }
        if weights.len() != num_neurons * input_dim {
            return Err(GpuError::DimensionMismatch {
                expected: num_neurons * input_dim,
                actual: weights.len(),
            });
        }

        pollster::block_on(self.find_bmus_gpu_async(
            inputs, weights, batch_size, num_neurons, input_dim,
        ))
    }

    /// Find BMU indices on GPU (async implementation).
    async fn find_bmus_gpu_async(
        &self,
        inputs: &[f32],
        weights: &[f32],
        batch_size: usize,
        num_neurons: usize,
        input_dim: usize,
    ) -> Result<Vec<usize>, GpuError> {
        // Step 1: Compute squared norms for inputs and weights
        let input_norms = self.compute_row_norms_async(inputs, batch_size, input_dim).await?;
        let weight_norms = self.compute_row_norms_async(weights, num_neurons, input_dim).await?;

        // Step 2: Run fused BMU shader
        let bmu_indices = self.fused_bmu_search_async(
            inputs, weights, &input_norms, &weight_norms,
            batch_size, num_neurons, input_dim,
        ).await?;

        Ok(bmu_indices.into_iter().map(|x| x as usize).collect())
    }

    /// Fused BMU search - finds argmin distance for each input in one GPU pass.
    async fn fused_bmu_search_async(
        &self,
        inputs: &[f32],
        weights: &[f32],
        input_norms: &[f32],
        weight_norms: &[f32],
        batch_size: usize,
        num_neurons: usize,
        input_dim: usize,
    ) -> Result<Vec<u32>, GpuError> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fused BMU Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::FUSED_BMU_SHADER.into()),
        });

        // Create buffers
        let inputs_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Inputs Buffer"),
            size: (inputs.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let weights_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weights Buffer"),
            size: (weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_norms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Norms Buffer"),
            size: (input_norms.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let weight_norms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weight Norms Buffer"),
            size: (weight_norms.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BMU Indices Buffer"),
            size: (batch_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            batch_size: u32,
            num_neurons: u32,
            input_dim: u32,
            _padding: u32,
        }

        let dims = Dimensions {
            batch_size: batch_size as u32,
            num_neurons: num_neurons as u32,
            input_dim: input_dim as u32,
            _padding: 0,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        self.queue.write_buffer(&inputs_buffer, 0, bytemuck::cast_slice(inputs));
        self.queue.write_buffer(&weights_buffer, 0, bytemuck::cast_slice(weights));
        self.queue.write_buffer(&input_norms_buffer, 0, bytemuck::cast_slice(input_norms));
        self.queue.write_buffer(&weight_norms_buffer, 0, bytemuck::cast_slice(weight_norms));
        self.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fused BMU Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused BMU Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_norms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weight_norms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused BMU Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fused BMU Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (batch_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Fused BMU Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fused BMU Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Workgroup size is 256 (matches FUSED_BMU_SHADER @workgroup_size)
            compute_pass.dispatch_workgroups((batch_size as u32).div_ceil(256), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (batch_size * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .receive()
            .await
            .ok_or_else(|| GpuError::BufferMapping("Channel receive failed".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let result = {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&data).to_vec()
        };

        staging_buffer.unmap();

        Ok(result)
    }

    /// Compute batched squared distances (async implementation).
    async fn batched_distances_async(
        &self,
        inputs: &[f32],
        weights: &[f32],
        batch_size: usize,
        num_neurons: usize,
        input_dim: usize,
    ) -> Result<Vec<f32>, GpuError> {
        // Step 1: Compute squared norms for inputs and weights
        let input_norms = self.compute_row_norms_async(inputs, batch_size, input_dim).await?;
        let weight_norms = self.compute_row_norms_async(weights, num_neurons, input_dim).await?;

        // Step 2: Compute dot products: inputs @ weights.T
        // This gives us a (batch_size × num_neurons) matrix
        let dot_products = self
            .matmul_transposed_async(inputs, weights, batch_size, input_dim, num_neurons)
            .await?;

        // Step 3: Combine: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
        let mut distances = vec![0.0f32; batch_size * num_neurons];
        for i in 0..batch_size {
            for j in 0..num_neurons {
                let idx = i * num_neurons + j;
                distances[idx] = input_norms[i] + weight_norms[j] - 2.0 * dot_products[idx];
                // Clamp to avoid negative values due to floating point errors
                if distances[idx] < 0.0 {
                    distances[idx] = 0.0;
                }
            }
        }

        Ok(distances)
    }

    /// Compute squared L2 norms for each row of a matrix.
    async fn compute_row_norms_async(
        &self,
        data: &[f32],
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Vec<f32>, GpuError> {
        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Row Norms Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::ROW_NORMS_SHADER.into()),
        });

        // Create buffers
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Row Norms Input"),
            size: (data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Row Norms Output"),
            size: (num_rows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dimensions uniform
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            num_rows: u32,
            num_cols: u32,
        }

        let dims = Dimensions {
            num_rows: num_rows as u32,
            num_cols: num_cols as u32,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        self.queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(data));
        self.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Row Norms Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Row Norms Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Row Norms Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Row Norms Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (num_rows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Row Norms Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Row Norms Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((num_rows as u32).div_ceil(256), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (num_rows * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .receive()
            .await
            .ok_or_else(|| GpuError::BufferMapping("Channel receive failed".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let result = {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };

        staging_buffer.unmap();

        Ok(result)
    }

    /// Matrix multiplication with transposed B: C = A @ B.T
    ///
    /// Computes C[i,j] = sum_k(A[i,k] * B[j,k])
    ///
    /// This is equivalent to A @ B.T where:
    /// - A is (m × k)
    /// - B is (n × k)
    /// - C is (m × n)
    async fn matmul_transposed_async(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, GpuError> {
        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Transposed Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::MATMUL_TRANSPOSED_SHADER.into()),
        });

        // Create buffers
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix A"),
            size: (a.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix B"),
            size: (b.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C"),
            size: ((m * n) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dimensions uniform
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            m: u32,
            k: u32,
            n: u32,
            _padding: u32,
        }

        let dims = Dimensions {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _padding: 0,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        self.queue.write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        self.queue.write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));
        self.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matmul Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matmul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matmul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: ((m * n) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matmul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (16×16 threads per workgroup)
            let num_workgroups_x = (m as u32).div_ceil(16);
            let num_workgroups_y = (n as u32).div_ceil(16);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &c_buffer,
            0,
            &staging_buffer,
            0,
            ((m * n) * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .receive()
            .await
            .ok_or_else(|| GpuError::BufferMapping("Channel receive failed".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let result = {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };

        staging_buffer.unmap();

        Ok(result)
    }

    /// Find BMU indices for a batch of inputs.
    ///
    /// Given distances (batch_size × num_neurons), returns the index of the
    /// neuron with minimum distance for each input.
    ///
    /// # Arguments
    ///
    /// * `distances` - Flattened distance matrix (batch_size × num_neurons)
    /// * `batch_size` - Number of inputs
    /// * `num_neurons` - Number of neurons
    ///
    /// # Returns
    ///
    /// Vector of BMU indices, one per input.
    pub fn find_bmu_indices(
        distances: &[f32],
        batch_size: usize,
        num_neurons: usize,
    ) -> Vec<usize> {
        let mut bmus = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let row_start = i * num_neurons;
            let row_end = row_start + num_neurons;
            let row = &distances[row_start..row_end];

            let (min_idx, _min_dist) = row
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &f32::MAX));

            bmus.push(min_idx);
        }

        bmus
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available() {
        // Just test that is_available doesn't panic
        let _available = GpuAccelerator::is_available();
    }

    #[test]
    fn test_find_bmu_indices() {
        // Test the CPU BMU index finder
        let distances = vec![
            1.0, 0.5, 2.0,  // Input 0: min at index 1
            0.1, 0.2, 0.3,  // Input 1: min at index 0
            9.0, 8.0, 7.0,  // Input 2: min at index 2
        ];

        let bmus = GpuAccelerator::find_bmu_indices(&distances, 3, 3);
        assert_eq!(bmus, vec![1, 0, 2]);
    }

    #[test]
    fn test_batched_distances() {
        if !GpuAccelerator::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let gpu = GpuAccelerator::new().expect("Failed to create GPU accelerator");

        // Simple test: 2 inputs, 3 neurons, 4 dimensions
        let inputs = vec![
            1.0, 0.0, 0.0, 0.0,  // Input 0: [1,0,0,0]
            0.0, 1.0, 0.0, 0.0,  // Input 1: [0,1,0,0]
        ];

        let weights = vec![
            1.0, 0.0, 0.0, 0.0,  // Neuron 0: [1,0,0,0] - same as input 0
            0.0, 1.0, 0.0, 0.0,  // Neuron 1: [0,1,0,0] - same as input 1
            0.0, 0.0, 1.0, 0.0,  // Neuron 2: [0,0,1,0]
        ];

        let distances = gpu
            .batched_distances(&inputs, &weights, 2, 3, 4)
            .expect("batched_distances failed");

        // Expected distances:
        // Input 0 to Neuron 0: 0 (identical)
        // Input 0 to Neuron 1: 2 (1^2 + 1^2 = 2)
        // Input 0 to Neuron 2: 2 (1^2 + 1^2 = 2)
        // Input 1 to Neuron 0: 2
        // Input 1 to Neuron 1: 0 (identical)
        // Input 1 to Neuron 2: 2

        assert_eq!(distances.len(), 6);
        assert!(distances[0].abs() < 0.01, "Input 0 to Neuron 0 should be ~0");
        assert!((distances[1] - 2.0).abs() < 0.01, "Input 0 to Neuron 1 should be ~2");
        assert!((distances[3] - 2.0).abs() < 0.01, "Input 1 to Neuron 0 should be ~2");
        assert!(distances[4].abs() < 0.01, "Input 1 to Neuron 1 should be ~0");

        // Test BMU finding
        let bmus = GpuAccelerator::find_bmu_indices(&distances, 2, 3);
        assert_eq!(bmus[0], 0, "Input 0 BMU should be neuron 0");
        assert_eq!(bmus[1], 1, "Input 1 BMU should be neuron 1");
    }

    #[test]
    fn test_dimension_mismatch() {
        if !GpuAccelerator::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let gpu = GpuAccelerator::new().expect("Failed to create GPU accelerator");

        // Wrong input size
        let inputs = vec![1.0, 2.0, 3.0]; // 3 elements
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements

        let result = gpu.batched_distances(&inputs, &weights, 2, 2, 2);
        assert!(result.is_err(), "Should fail with dimension mismatch");
    }
}
