//! SIMD-optimized operations for SOM training.
//!
//! Uses f32 for better SIMD throughput (8 floats per AVX vs 4 doubles).

use rayon::prelude::*;

/// Compute squared Euclidean distance between two f32 slices.
/// Optimized for autovectorization.
#[inline]
pub fn distance_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Process in chunks of 8 for AVX alignment
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;

    let mut sum = 0.0f32;

    // Main loop - should autovectorize
    for i in 0..chunks {
        let base = i * 8;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        let d4 = a[base + 4] - b[base + 4];
        let d5 = a[base + 5] - b[base + 5];
        let d6 = a[base + 6] - b[base + 6];
        let d7 = a[base + 7] - b[base + 7];

        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 +
               d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
    }

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum
}

/// Find BMU (Best Matching Unit) index for a single input vector.
/// Returns the index of the neuron with minimum distance.
#[inline]
pub fn find_bmu_f32(weights: &[f32], input: &[f32], num_neurons: usize, weight_dim: usize) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    for i in 0..num_neurons {
        let offset = i * weight_dim;
        let neuron_weights = &weights[offset..offset + weight_dim];
        let dist = distance_squared_f32(neuron_weights, input);

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
/// SIMD-optimized with 8-wide unrolling for AVX.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        sum += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3]
            + a[base + 4] * b[base + 4]
            + a[base + 5] * b[base + 5]
            + a[base + 6] * b[base + 6]
            + a[base + 7] * b[base + 7];
    }

    let base = chunks * 8;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
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
