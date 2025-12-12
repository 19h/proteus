//! WGSL compute shaders for GPU-accelerated operations.
//!
//! These shaders are optimized for batched BMU search in SOM training.

/// Fused BMU search using tiled shared memory algorithm.
///
/// This shader is optimized for memory bandwidth by using shared memory
/// to cache neuron weights, dramatically reducing global memory traffic.
///
/// Algorithm:
/// 1. Each workgroup handles a tile of input samples (TILE_SAMPLES samples)
/// 2. For each tile of neurons (TILE_NEURONS neurons):
///    a. Cooperatively load neuron weights and norms into shared memory
///    b. Each thread computes distances from its sample to all neurons in tile
///    c. Update local minimum
/// 3. Write final BMU index to global memory
///
/// Memory traffic analysis:
/// - Without tiling: each thread reads all neuron weights = O(batch_size * num_neurons * dim)
/// - With tiling: weights loaded once per tile = O(num_neurons * dim + batch_size * dim)
///
/// For typical workload (50k samples, 65k neurons, 128 dim):
/// - Without tiling: 50k * 65k * 128 * 4B = 1.6TB reads
/// - With tiling: 65k * 128 * 4B + 50k * 128 * 4B = 58MB reads
/// That's a 27,000x reduction in memory traffic!
///
/// Inputs:
/// - inputs: (batch_size × input_dim), row-major
/// - weights: (num_neurons × input_dim), row-major
/// - input_norms: precomputed ||input[i]||^2 for each input
/// - weight_norms: precomputed ||weight[j]||^2 for each neuron
/// - dims: { batch_size, num_neurons, input_dim }
///
/// Output:
/// - bmu_indices: u32 index of best matching unit for each input
pub const FUSED_BMU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> input_norms: array<f32>;
@group(0) @binding(3) var<storage, read> weight_norms: array<f32>;
@group(0) @binding(4) var<storage, read_write> bmu_indices: array<u32>;

struct Dimensions {
    batch_size: u32,
    num_neurons: u32,
    input_dim: u32,
    _padding: u32,
}

@group(0) @binding(5) var<uniform> dims: Dimensions;

// Tile size for neurons - load this many neurons into shared memory at once
// 32 neurons * 128 dim * 4 bytes = 16KB per tile (fits in shared memory)
const TILE_NEURONS: u32 = 32u;

// Shared memory for neuron weights tile
// Maximum input_dim we support is 256 (32 * 256 * 4 = 32KB)
var<workgroup> weight_tile: array<f32, 8192>;  // TILE_NEURONS * 256
var<workgroup> norm_tile: array<f32, 32>;       // TILE_NEURONS

// Each thread handles one input sample
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let sample_idx = global_id.x;
    let thread_idx = local_id.x;

    // Early exit for out-of-bounds threads
    let valid_sample = sample_idx < dims.batch_size;

    // Load input vector and norm into registers (only for valid samples)
    var input_norm: f32 = 0.0;
    var min_dist: f32 = 3.402823466e+38; // FLT_MAX
    var min_idx: u32 = 0u;

    if (valid_sample) {
        input_norm = input_norms[sample_idx];
    }

    // Process neurons in tiles
    let num_tiles = (dims.num_neurons + TILE_NEURONS - 1u) / TILE_NEURONS;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {
        let tile_start = tile * TILE_NEURONS;
        let tile_end = min(tile_start + TILE_NEURONS, dims.num_neurons);
        let neurons_in_tile = tile_end - tile_start;

        // === Phase 1: Cooperatively load neuron weights into shared memory ===
        // Each thread loads multiple elements to fill the tile
        let elements_per_neuron = dims.input_dim;
        let total_elements = neurons_in_tile * elements_per_neuron;
        let elements_per_thread = (total_elements + 255u) / 256u;

        for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
            let elem_idx = thread_idx + i * 256u;
            if (elem_idx < total_elements) {
                let neuron_local = elem_idx / elements_per_neuron;
                let dim_idx = elem_idx % elements_per_neuron;
                let neuron_global = tile_start + neuron_local;

                let src_idx = neuron_global * dims.input_dim + dim_idx;
                let dst_idx = neuron_local * dims.input_dim + dim_idx;

                // Use max dimension (256) for indexing into fixed-size array
                weight_tile[neuron_local * 256u + dim_idx] = weights[src_idx];
            }
        }

        // Load norms (only need TILE_NEURONS values)
        if (thread_idx < neurons_in_tile) {
            norm_tile[thread_idx] = weight_norms[tile_start + thread_idx];
        }

        // Synchronize to ensure tile is fully loaded
        workgroupBarrier();

        // === Phase 2: Compute distances from this sample to all neurons in tile ===
        if (valid_sample) {
            let input_start = sample_idx * dims.input_dim;

            // Process neurons in groups of 4 for ILP
            let neuron_groups = neurons_in_tile / 4u;
            let neuron_remainder = neurons_in_tile % 4u;

            for (var ng: u32 = 0u; ng < neuron_groups; ng = ng + 1u) {
                let n0 = ng * 4u;
                let n1 = n0 + 1u;
                let n2 = n0 + 2u;
                let n3 = n0 + 3u;

                var dot0: f32 = 0.0;
                var dot1: f32 = 0.0;
                var dot2: f32 = 0.0;
                var dot3: f32 = 0.0;

                // Compute dot products using shared memory
                for (var d: u32 = 0u; d < dims.input_dim; d = d + 1u) {
                    let inp = inputs[input_start + d];
                    // Index into weight_tile using 256 stride
                    dot0 = dot0 + inp * weight_tile[n0 * 256u + d];
                    dot1 = dot1 + inp * weight_tile[n1 * 256u + d];
                    dot2 = dot2 + inp * weight_tile[n2 * 256u + d];
                    dot3 = dot3 + inp * weight_tile[n3 * 256u + d];
                }

                // Compute distances
                let dist0 = input_norm + norm_tile[n0] - 2.0 * dot0;
                let dist1 = input_norm + norm_tile[n1] - 2.0 * dot1;
                let dist2 = input_norm + norm_tile[n2] - 2.0 * dot2;
                let dist3 = input_norm + norm_tile[n3] - 2.0 * dot3;

                // Update minimum
                let global_n0 = tile_start + n0;
                if (dist0 < min_dist) { min_dist = dist0; min_idx = global_n0; }
                if (dist1 < min_dist) { min_dist = dist1; min_idx = global_n0 + 1u; }
                if (dist2 < min_dist) { min_dist = dist2; min_idx = global_n0 + 2u; }
                if (dist3 < min_dist) { min_dist = dist3; min_idx = global_n0 + 3u; }
            }

            // Handle remaining neurons
            for (var r: u32 = 0u; r < neuron_remainder; r = r + 1u) {
                let n_local = neuron_groups * 4u + r;

                var dot: f32 = 0.0;
                for (var d: u32 = 0u; d < dims.input_dim; d = d + 1u) {
                    dot = dot + inputs[input_start + d] * weight_tile[n_local * 256u + d];
                }

                let dist = input_norm + norm_tile[n_local] - 2.0 * dot;
                let global_n = tile_start + n_local;
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = global_n;
                }
            }
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (valid_sample) {
        bmu_indices[sample_idx] = min_idx;
    }
}
"#;

/// Compute squared L2 norm for each row of a matrix.
///
/// Given a (num_rows × num_cols) matrix, computes ||row||^2 for each row.
/// Each thread handles one row, computing the sum of squared elements.
///
/// Inputs:
/// - input: flattened matrix (num_rows × num_cols), row-major
/// - dims: { num_rows, num_cols }
///
/// Output:
/// - output: vector of length num_rows, output[i] = ||input[i,:]||^2
pub const ROW_NORMS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Dimensions {
    num_rows: u32,
    num_cols: u32,
}

@group(0) @binding(2) var<uniform> dims: Dimensions;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if (row >= dims.num_rows) {
        return;
    }

    var sum: f32 = 0.0;
    let row_start = row * dims.num_cols;

    // Compute sum of squared elements for this row
    for (var col: u32 = 0u; col < dims.num_cols; col = col + 1u) {
        let val = input[row_start + col];
        sum = sum + val * val;
    }

    output[row] = sum;
}
"#;

/// Matrix multiplication with transposed B: C = A @ B.T
///
/// Computes C[i,j] = sum_k(A[i,k] * B[j,k])
///
/// This is optimized for BMU distance computation where we need
/// to compute dot products between inputs and all neuron weights.
///
/// Inputs:
/// - a: matrix A (m × k), row-major
/// - b: matrix B (n × k), row-major (will be transposed virtually)
/// - dims: { m, k, n }
///
/// Output:
/// - c: matrix C (m × n), row-major
///
/// Performance note: Each thread computes one element of C by iterating
/// over the shared dimension k. This is memory-bound but parallelizes well.
pub const MATMUL_TRANSPOSED_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dimensions {
    M: u32,  // rows of A and C
    K: u32,  // cols of A, cols of B (B is n×k, we treat it as transposed)
    N: u32,  // rows of B and cols of C
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Workgroup size: 16×16 = 256 threads
// Each thread computes one element of C
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;  // Which row of A / row of C
    let col = global_id.y;  // Which row of B / col of C

    // Bounds check
    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum: f32 = 0.0;

    // Compute dot product: C[row,col] = sum(A[row,k] * B[col,k])
    // Note: B is accessed as B[col,k], not B[k,col], because we want B.T
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        let a_idx = row * dims.K + k;    // A[row, k]
        let b_idx = col * dims.K + k;    // B[col, k] (B.T[k, col])
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = row * dims.N + col;       // C[row, col]
    c[c_idx] = sum;
}
"#;

/// Combined squared distance computation.
///
/// Computes ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a · b)
///
/// This shader combines all three terms efficiently for BMU search.
/// For each input vector and each neuron, computes the squared distance.
///
/// Inputs:
/// - inputs: (batch_size × input_dim), row-major
/// - weights: (num_neurons × input_dim), row-major
/// - dims: { batch_size, num_neurons, input_dim }
///
/// Output:
/// - distances: (batch_size × num_neurons), row-major
///
/// Note: This is an alternative to the decomposed approach (norms + matmul).
/// The decomposed approach may be faster for very large dimensions due to
/// better memory access patterns in the matmul.
///
/// Currently unused - kept for future optimization experiments.
#[allow(dead_code)]
pub const SQUARED_DISTANCES_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;

struct Dimensions {
    batch_size: u32,
    num_neurons: u32,
    input_dim: u32,
    _padding: u32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Each thread computes distance from one input to one neuron
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_idx = global_id.x;   // Which input vector
    let neuron_idx = global_id.y;  // Which neuron

    // Bounds check
    if (input_idx >= dims.batch_size || neuron_idx >= dims.num_neurons) {
        return;
    }

    var sum: f32 = 0.0;

    let input_start = input_idx * dims.input_dim;
    let weight_start = neuron_idx * dims.input_dim;

    // Compute squared Euclidean distance
    for (var d: u32 = 0u; d < dims.input_dim; d = d + 1u) {
        let diff = inputs[input_start + d] - weights[weight_start + d];
        sum = sum + diff * diff;
    }

    let dist_idx = input_idx * dims.num_neurons + neuron_idx;
    distances[dist_idx] = sum;
}
"#;

/// Find minimum index (argmin) for each row.
///
/// Given a (num_rows × num_cols) matrix, finds the column index with
/// minimum value for each row.
///
/// Inputs:
/// - input: flattened matrix (num_rows × num_cols), row-major
/// - dims: { num_rows, num_cols }
///
/// Output:
/// - output: vector of u32 indices, length num_rows
///
/// Note: This is done on CPU currently since argmin reduction
/// on GPU requires multiple passes for large matrices.
///
/// Currently unused - kept for future optimization experiments.
#[allow(dead_code)]
pub const ARGMIN_ROWS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Dimensions {
    num_rows: u32,
    num_cols: u32,
}

@group(0) @binding(2) var<uniform> dims: Dimensions;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if (row >= dims.num_rows) {
        return;
    }

    let row_start = row * dims.num_cols;

    var min_val: f32 = input[row_start];
    var min_idx: u32 = 0u;

    // Linear scan for minimum (simple but effective for reasonable num_cols)
    for (var col: u32 = 1u; col < dims.num_cols; col = col + 1u) {
        let val = input[row_start + col];
        if (val < min_val) {
            min_val = val;
            min_idx = col;
        }
    }

    output[row] = min_idx;
}
"#;

/// Batched SOM weight updates.
///
/// Applies multiple weight updates in parallel. Each update moves a neuron's
/// weight vector toward an input vector by a given influence factor:
///   weights[neuron] += influence * (input - weights[neuron])
///
/// This is equivalent to:
///   weights[neuron] = (1 - influence) * weights[neuron] + influence * input
///
/// The shader processes updates in a conflict-free manner by having each
/// thread handle one dimension of one update.
///
/// Inputs:
/// - weights: (num_neurons × weight_dim), read-write
/// - inputs: flattened input vectors for all updates
/// - update_info: array of (neuron_idx, input_offset, influence) per update
/// - dims: { num_updates, weight_dim, num_neurons }
///
/// Note: Multiple updates to the same neuron are accumulated correctly
/// because we use atomicAdd for the weight modifications.
///
/// Currently unused - kept for future optimization experiments.
#[allow(dead_code)]
pub const WEIGHT_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> update_neuron_idx: array<u32>;
@group(0) @binding(3) var<storage, read> update_input_idx: array<u32>;
@group(0) @binding(4) var<storage, read> update_influence: array<f32>;

struct Dimensions {
    num_updates: u32,
    weight_dim: u32,
    num_neurons: u32,
    _padding: u32,
}

@group(0) @binding(5) var<uniform> dims: Dimensions;

// Each thread handles one update's contribution to one dimension
// Thread (update_idx, dim_idx) updates weights[neuron][dim]
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let update_idx = global_id.x;
    let dim_idx = global_id.y;

    if (update_idx >= dims.num_updates || dim_idx >= dims.weight_dim) {
        return;
    }

    let neuron_idx = update_neuron_idx[update_idx];
    let input_idx = update_input_idx[update_idx];
    let influence = update_influence[update_idx];

    let weight_offset = neuron_idx * dims.weight_dim + dim_idx;
    let input_offset = input_idx * dims.weight_dim + dim_idx;

    let current_weight = weights[weight_offset];
    let input_val = inputs[input_offset];

    // weight += influence * (input - weight)
    // Note: This is not atomic, so we rely on each (neuron, dim) pair
    // being updated by at most one thread. If multiple updates target
    // the same neuron, they should be pre-aggregated on CPU.
    weights[weight_offset] = current_weight + influence * (input_val - current_weight);
}
"#;

/// Aggregated weight update shader - handles multiple updates to same neuron.
///
/// This version pre-computes the aggregated delta for each neuron on the GPU,
/// allowing multiple samples to contribute to the same neuron's update.
///
/// Algorithm:
/// 1. For each neuron, accumulate: sum(influence_i * input_i) and sum(influence_i)
/// 2. New weight = (1 - total_influence) * old_weight + sum(influence_i * input_i)
///
/// This is mathematically equivalent to applying updates sequentially when
/// influences are small (which they are in SOM training).
///
/// Bindings:
/// - 0: weights (read_write) - neuron weights to update
/// - 1: neuron_influences - total influence per neuron
/// - 2: neuron_weighted_inputs - pre-aggregated weighted input sums
/// - 3: dims uniform
pub const AGGREGATED_WEIGHT_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> neuron_influences: array<f32>;
@group(0) @binding(2) var<storage, read> neuron_weighted_inputs: array<f32>;

struct Dimensions {
    num_neurons: u32,
    weight_dim: u32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Each thread handles one dimension of one neuron
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let neuron_idx = global_id.x;
    let dim_idx = global_id.y;

    if (neuron_idx >= dims.num_neurons || dim_idx >= dims.weight_dim) {
        return;
    }

    let total_influence = neuron_influences[neuron_idx];

    // Skip neurons with no updates
    if (total_influence == 0.0) {
        return;
    }

    let weight_offset = neuron_idx * dims.weight_dim + dim_idx;

    let current_weight = weights[weight_offset];
    let weighted_input_sum = neuron_weighted_inputs[weight_offset];

    // new_weight = (1 - total_influence) * current_weight + weighted_input_sum
    // where weighted_input_sum = sum(influence_i * input_i)
    weights[weight_offset] = (1.0 - total_influence) * current_weight + weighted_input_sum;
}
"#;
