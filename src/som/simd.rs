//! SIMD-optimized operations for SOM training.
//!
//! Uses f32 for better SIMD throughput.
//! Explicit portable SIMD for guaranteed vectorization.
//!
//! Architecture-specific optimizations:
//! - ARM (NEON): f32x4 (native 128-bit registers)
//! - x86_64 (AVX): f32x8 (native 256-bit registers)

use rayon::prelude::*;
use std::simd::num::SimdFloat;

// Use architecture-appropriate vector width
#[cfg(target_arch = "aarch64")]
use std::simd::f32x4 as f32xN;
#[cfg(target_arch = "aarch64")]
const LANE_COUNT: usize = 4;

#[cfg(target_arch = "x86_64")]
use std::simd::f32x8 as f32xN;
#[cfg(target_arch = "x86_64")]
const LANE_COUNT: usize = 8;

// Fallback for other architectures
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
use std::simd::f32x4 as f32xN;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const LANE_COUNT: usize = 4;

/// Compute squared Euclidean distance between two f32 slices using explicit SIMD.
/// Automatically uses optimal vector width for the target architecture.
#[inline]
pub fn distance_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Process 4 vectors per iteration for better ILP
    let stride = LANE_COUNT * 4;
    let chunks = a.len() / stride;
    let mut sum0 = f32xN::splat(0.0);
    let mut sum1 = f32xN::splat(0.0);
    let mut sum2 = f32xN::splat(0.0);
    let mut sum3 = f32xN::splat(0.0);

    for i in 0..chunks {
        let base = i * stride;

        let va0 = f32xN::from_slice(&a[base..]);
        let vb0 = f32xN::from_slice(&b[base..]);
        let diff0 = va0 - vb0;
        sum0 += diff0 * diff0;

        let va1 = f32xN::from_slice(&a[base + LANE_COUNT..]);
        let vb1 = f32xN::from_slice(&b[base + LANE_COUNT..]);
        let diff1 = va1 - vb1;
        sum1 += diff1 * diff1;

        let va2 = f32xN::from_slice(&a[base + LANE_COUNT * 2..]);
        let vb2 = f32xN::from_slice(&b[base + LANE_COUNT * 2..]);
        let diff2 = va2 - vb2;
        sum2 += diff2 * diff2;

        let va3 = f32xN::from_slice(&a[base + LANE_COUNT * 3..]);
        let vb3 = f32xN::from_slice(&b[base + LANE_COUNT * 3..]);
        let diff3 = va3 - vb3;
        sum3 += diff3 * diff3;
    }

    // Combine accumulators
    let sum_vec = sum0 + sum1 + sum2 + sum3;
    let mut sum = sum_vec.reduce_sum();

    // Handle remainder
    let base = chunks * stride;
    for i in base..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }

    sum
}

/// Compute dot product using explicit SIMD.
/// Automatically uses optimal vector width for the target architecture.
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let stride = LANE_COUNT * 4;
    let chunks = a.len() / stride;
    let mut sum0 = f32xN::splat(0.0);
    let mut sum1 = f32xN::splat(0.0);
    let mut sum2 = f32xN::splat(0.0);
    let mut sum3 = f32xN::splat(0.0);

    for i in 0..chunks {
        let base = i * stride;

        let va0 = f32xN::from_slice(&a[base..]);
        let vb0 = f32xN::from_slice(&b[base..]);
        sum0 += va0 * vb0;

        let va1 = f32xN::from_slice(&a[base + LANE_COUNT..]);
        let vb1 = f32xN::from_slice(&b[base + LANE_COUNT..]);
        sum1 += va1 * vb1;

        let va2 = f32xN::from_slice(&a[base + LANE_COUNT * 2..]);
        let vb2 = f32xN::from_slice(&b[base + LANE_COUNT * 2..]);
        sum2 += va2 * vb2;

        let va3 = f32xN::from_slice(&a[base + LANE_COUNT * 3..]);
        let vb3 = f32xN::from_slice(&b[base + LANE_COUNT * 3..]);
        sum3 += va3 * vb3;
    }

    let sum_vec = sum0 + sum1 + sum2 + sum3;
    let mut sum = sum_vec.reduce_sum();

    let base = chunks * stride;
    for i in base..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

/// Find BMU (Best Matching Unit) index for a single input vector.
/// Returns the index of the neuron with minimum squared distance.
#[inline]
pub fn find_bmu_f32(weights: &[f32], input: &[f32], num_neurons: usize, weight_dim: usize) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    // Process 4 neurons at a time to reduce loop overhead and improve ILP
    let chunks = num_neurons / 4;

    for chunk in 0..chunks {
        let base = chunk * 4;

        let dist0 = distance_squared_f32(&weights[base * weight_dim..(base + 1) * weight_dim], input);
        let dist1 = distance_squared_f32(&weights[(base + 1) * weight_dim..(base + 2) * weight_dim], input);
        let dist2 = distance_squared_f32(&weights[(base + 2) * weight_dim..(base + 3) * weight_dim], input);
        let dist3 = distance_squared_f32(&weights[(base + 3) * weight_dim..(base + 4) * weight_dim], input);

        if dist0 < best_dist { best_dist = dist0; best_idx = base; }
        if dist1 < best_dist { best_dist = dist1; best_idx = base + 1; }
        if dist2 < best_dist { best_dist = dist2; best_idx = base + 2; }
        if dist3 < best_dist { best_dist = dist3; best_idx = base + 3; }
    }

    // Handle remaining neurons
    for i in (chunks * 4)..num_neurons {
        let offset = i * weight_dim;
        let dist = distance_squared_f32(&weights[offset..offset + weight_dim], input);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    best_idx
}

/// Find BMUs for all inputs in parallel.
/// Returns vector of (sample_index, bmu_index).
pub fn find_all_bmus_parallel(
    weights: &[f32],
    inputs: &[(usize, Vec<f32>)], // (original_idx, input_vec)
    num_neurons: usize,
    weight_dim: usize,
) -> Vec<(usize, usize)> {
    inputs
        .par_iter()
        .map(|(idx, input)| {
            let bmu = find_bmu_f32(weights, input, num_neurons, weight_dim);
            (*idx, bmu)
        })
        .collect()
}

/// Find BMU using hierarchical search for large grids.
/// First searches a coarse subgrid, then refines around the best match.
/// Much faster for large grids (e.g., 128x128) with minimal accuracy loss.
#[inline]
pub fn find_bmu_hierarchical(
    weights: &[f32],
    input: &[f32],
    grid_dim: usize,
    weight_dim: usize,
) -> usize {
    let num_neurons = grid_dim * grid_dim;

    // For small grids, just do exhaustive search
    if grid_dim <= 32 {
        return find_bmu_f32(weights, input, num_neurons, weight_dim);
    }

    // Coarse search: sample every 4th neuron in each dimension
    let step = 4;
    let mut best_coarse_idx = 0;
    let mut best_coarse_dist = f32::MAX;

    for row in (0..grid_dim).step_by(step) {
        for col in (0..grid_dim).step_by(step) {
            let idx = row * grid_dim + col;
            let offset = idx * weight_dim;
            let dist = distance_squared_f32(&weights[offset..offset + weight_dim], input);
            if dist < best_coarse_dist {
                best_coarse_dist = dist;
                best_coarse_idx = idx;
            }
        }
    }

    // Fine search: check all neurons within radius of best coarse match
    let coarse_row = (best_coarse_idx / grid_dim) as isize;
    let coarse_col = (best_coarse_idx % grid_dim) as isize;
    let search_radius: isize = (step as isize) + 1;

    let mut best_idx = best_coarse_idx;
    let mut best_dist = best_coarse_dist;

    for dr in -search_radius..=search_radius {
        for dc in -search_radius..=search_radius {
            let row = coarse_row + dr;
            let col = coarse_col + dc;

            if row >= 0 && row < grid_dim as isize && col >= 0 && col < grid_dim as isize {
                let idx = (row as usize) * grid_dim + (col as usize);
                let offset = idx * weight_dim;
                let dist = distance_squared_f32(&weights[offset..offset + weight_dim], input);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
        }
    }

    best_idx
}

/// Find BMUs for all inputs in parallel using hierarchical search.
/// Faster for large grids with acceptable accuracy trade-off.
pub fn find_all_bmus_parallel_fast(
    weights: &[f32],
    inputs: &[(usize, Vec<f32>)],
    grid_dim: usize,
    weight_dim: usize,
) -> Vec<(usize, usize)> {
    inputs
        .par_iter()
        .map(|(idx, input)| {
            let bmu = find_bmu_hierarchical(weights, input, grid_dim, weight_dim);
            (*idx, bmu)
        })
        .collect()
}

/// Update neuron weights towards input with given influence.
/// Optimized for autovectorization.
#[inline]
pub fn update_weights_f32(weights: &mut [f32], input: &[f32], influence: f32) {
    debug_assert_eq!(weights.len(), input.len());

    // Process in chunks for better vectorization
    let chunks = weights.len() / 4;
    let remainder = weights.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        weights[base] += influence * (input[base] - weights[base]);
        weights[base + 1] += influence * (input[base + 1] - weights[base + 1]);
        weights[base + 2] += influence * (input[base + 2] - weights[base + 2]);
        weights[base + 3] += influence * (input[base + 3] - weights[base + 3]);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        weights[base + i] += influence * (input[base + i] - weights[base + i]);
    }
}

/// Normalize a vector to unit length.
/// SIMD-optimized with loop unrolling.
#[inline]
pub fn normalize_f32(vec: &mut [f32]) {
    let norm_sq = dot_product_f32(vec, vec);
    if norm_sq > 1e-10 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        scale_f32(vec, inv_norm);
    }
}

/// Compute dot product of two f32 slices.
/// Alias for dot_product_simd for backwards compatibility.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    dot_product_simd(a, b)
}

/// Scale a vector by a scalar.
/// SIMD-optimized with 8-wide unrolling.
#[inline]
pub fn scale_f32(vec: &mut [f32], scalar: f32) {
    let chunks = vec.len() / 8;
    let remainder = vec.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        vec[base] *= scalar;
        vec[base + 1] *= scalar;
        vec[base + 2] *= scalar;
        vec[base + 3] *= scalar;
        vec[base + 4] *= scalar;
        vec[base + 5] *= scalar;
        vec[base + 6] *= scalar;
        vec[base + 7] *= scalar;
    }

    let base = chunks * 8;
    for i in 0..remainder {
        vec[base + i] *= scalar;
    }
}

/// Add src vector to dst vector (dst += src).
/// SIMD-optimized with 8-wide unrolling.
#[inline]
pub fn add_vectors_f32(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());

    let chunks = dst.len() / 8;
    let remainder = dst.len() % 8;

    for i in 0..chunks {
        let base = i * 8;
        dst[base] += src[base];
        dst[base + 1] += src[base + 1];
        dst[base + 2] += src[base + 2];
        dst[base + 3] += src[base + 3];
        dst[base + 4] += src[base + 4];
        dst[base + 5] += src[base + 5];
        dst[base + 6] += src[base + 6];
        dst[base + 7] += src[base + 7];
    }

    let base = chunks * 8;
    for i in 0..remainder {
        dst[base + i] += src[base + i];
    }
}

/// Pre-computed neighborhood weights for a given radius.
/// Returns a flat array of (dr, dc, weight) tuples for non-zero weights.
pub fn precompute_neighborhood(radius: f32, threshold: f32) -> Vec<(i32, i32, f32)> {
    let radius_int = (radius * 1.5).ceil() as i32;
    let sigma_sq = radius * radius;
    let mut neighbors = Vec::new();

    for dr in -radius_int..=radius_int {
        for dc in -radius_int..=radius_int {
            let dist_sq = (dr * dr + dc * dc) as f32;
            if dist_sq <= radius * radius * 4.0 {
                let weight = (-dist_sq / (2.0 * sigma_sq)).exp();
                if weight > threshold {
                    neighbors.push((dr, dc, weight));
                }
            }
        }
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_squared() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dist = distance_squared_f32(&a, &b);
        let expected: f32 = (1..=8).map(|x| (x * x) as f32).sum();
        assert!((dist - expected).abs() < 1e-5);
    }

    #[test]
    fn test_find_bmu() {
        let weights = vec![
            1.0f32, 0.0, 0.0,  // neuron 0
            0.0, 1.0, 0.0,     // neuron 1
            0.0, 0.0, 1.0,     // neuron 2
        ];
        let input = vec![0.9f32, 0.1, 0.0];
        let bmu = find_bmu_f32(&weights, &input, 3, 3);
        assert_eq!(bmu, 0);
    }

    #[test]
    fn test_update_weights() {
        let mut weights = vec![0.0f32, 0.0, 0.0, 0.0];
        let input = vec![1.0f32, 1.0, 1.0, 1.0];
        update_weights_f32(&mut weights, &input, 0.5);
        assert!((weights[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let dot = dot_product_f32(&a, &b);
        let expected: f32 = (1..=8).sum::<i32>() as f32;
        assert!((dot - expected).abs() < 1e-5);
    }

    #[test]
    fn test_scale() {
        let mut vec = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        scale_f32(&mut vec, 2.0);
        assert!((vec[0] - 2.0).abs() < 1e-5);
        assert!((vec[7] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_add_vectors() {
        let mut dst = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let src = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        add_vectors_f32(&mut dst, &src);
        assert!((dst[0] - 2.0).abs() < 1e-5);
        assert!((dst[7] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize() {
        let mut vec = vec![3.0f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        normalize_f32(&mut vec);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
