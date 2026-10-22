//! Spatial evidence aggregation, separated from SOM optimization.

use super::FoldingConfig;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct KernelOffset {
    row_delta: isize,
    column_delta: isize,
    weight: f64,
}

/// Reuses the corpus occupancy correction for every word on a trained map.
pub(super) struct SpatialFolder {
    dimension: usize,
    toroidal: bool,
    kernel: Vec<KernelOffset>,
    source_normalizers: Vec<f64>,
    smoothed_occupancy: Vec<f64>,
    background_exponent: f64,
}

impl SpatialFolder {
    pub(super) fn new(
        occupancy: &[usize],
        dimension: usize,
        toroidal: bool,
        config: &FoldingConfig,
    ) -> Self {
        let kernel = if config.sigma == 0.0 {
            Vec::new()
        } else {
            gaussian_kernel(dimension, toroidal, config.sigma)
        };
        let mut source_normalizers = vec![0.0; occupancy.len()];
        let mut smoothed_occupancy = vec![0.0; occupancy.len()];

        // Normalize each source separately so planar edges preserve evidence mass.
        for (source, normalizer) in source_normalizers.iter_mut().enumerate() {
            for &offset in &kernel {
                if offset_cell(source, offset, dimension, toroidal).is_some() {
                    *normalizer += offset.weight;
                }
            }
        }

        for (source, &count) in occupancy.iter().enumerate() {
            if count == 0 {
                continue;
            }
            for &offset in &kernel {
                if let Some(target) = offset_cell(source, offset, dimension, toroidal) {
                    smoothed_occupancy[target] +=
                        count as f64 * offset.weight / source_normalizers[source];
                }
            }
        }

        Self {
            dimension,
            toroidal,
            kernel,
            source_normalizers,
            smoothed_occupancy,
            background_exponent: config.background_exponent,
        }
    }

    pub(super) fn fold(&self, word_incidence: &BTreeMap<usize, usize>, limit: usize) -> Vec<u32> {
        if self.kernel.is_empty() {
            return strongest_observed_cells(word_incidence, limit);
        }

        let mut smoothed_incidence = BTreeMap::<usize, f64>::new();
        for (&source, &count) in word_incidence {
            for &offset in &self.kernel {
                if let Some(target) = offset_cell(source, offset, self.dimension, self.toroidal) {
                    *smoothed_incidence.entry(target).or_default() +=
                        count as f64 * offset.weight / self.source_normalizers[source];
                }
            }
        }

        let mut scored_cells: Vec<_> = smoothed_incidence
            .into_iter()
            .map(|(cell, incidence)| {
                let correction = self.smoothed_occupancy[cell].powf(self.background_exponent);
                (cell, incidence / correction)
            })
            .filter(|(_, score)| score.is_finite() && *score > 0.0)
            .collect();
        scored_cells
            .sort_unstable_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        scored_cells
            .into_iter()
            .take(limit)
            .map(|(cell, _)| cell as u32)
            .collect()
    }
}

fn strongest_observed_cells(incidence: &BTreeMap<usize, usize>, limit: usize) -> Vec<u32> {
    let mut observed_cells: Vec<_> = incidence.iter().collect();
    observed_cells.sort_unstable_by(|left, right| right.1.cmp(left.1).then(left.0.cmp(right.0)));
    observed_cells
        .into_iter()
        .take(limit)
        .map(|(&cell, _)| cell as u32)
        .collect()
}

fn gaussian_kernel(dimension: usize, toroidal: bool, sigma: f64) -> Vec<KernelOffset> {
    let radius = (3.0 * sigma).ceil().min((dimension - 1) as f64) as isize;
    let axis_offsets: Vec<isize> = if toroidal {
        // Each toroidal destination occurs once, including the even-grid antipode.
        (0..dimension)
            .map(|coordinate| {
                if coordinate <= dimension / 2 {
                    coordinate as isize
                } else {
                    coordinate as isize - dimension as isize
                }
            })
            .filter(|delta| delta.abs() <= radius)
            .collect()
    } else {
        (-radius..=radius).collect()
    };

    let mut kernel = Vec::new();
    for &row_delta in &axis_offsets {
        for &column_delta in &axis_offsets {
            let squared_scaled_distance =
                (row_delta as f64 / sigma).powi(2) + (column_delta as f64 / sigma).powi(2);
            let weight = (-0.5 * squared_scaled_distance).exp();
            if weight > 0.0 {
                kernel.push(KernelOffset {
                    row_delta,
                    column_delta,
                    weight,
                });
            }
        }
    }
    kernel
}

fn offset_cell(
    source: usize,
    offset: KernelOffset,
    dimension: usize,
    toroidal: bool,
) -> Option<usize> {
    let row = (source / dimension) as isize + offset.row_delta;
    let column = (source % dimension) as isize + offset.column_delta;
    let axis_range = 0..dimension as isize;

    if toroidal {
        let wrapped_row = row.rem_euclid(dimension as isize) as usize;
        let wrapped_column = column.rem_euclid(dimension as isize) as usize;
        Some(wrapped_row * dimension + wrapped_column)
    } else if axis_range.contains(&row) && axis_range.contains(&column) {
        Some(row as usize * dimension + column as usize)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_matches_independent_dense_equations_on_plane_and_torus() {
        let dimension = 4;
        let cell_count = dimension * dimension;
        let occupancy = vec![4, 0, 1, 0, 0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0];
        let word_incidence = BTreeMap::from([(0, 2), (5, 1), (8, 1)]);
        let expected_mass = occupancy.iter().sum::<usize>() as f64;

        for toroidal in [true, false] {
            // The kernel spans the entire map, including both toroidal antipodes.
            let config = FoldingConfig {
                sigma: 2.0,
                background_exponent: 0.5,
            };
            let folder = SpatialFolder::new(&occupancy, dimension, toroidal, &config);
            let mut dense_kernel = vec![vec![0.0; cell_count]; cell_count];

            // Independent oracle: enumerate source/destination pairs directly.
            for (source, row) in dense_kernel.iter_mut().enumerate() {
                for (target, weight) in row.iter_mut().enumerate() {
                    let mut row_distance = (source / dimension).abs_diff(target / dimension);
                    let mut column_distance = (source % dimension).abs_diff(target % dimension);
                    if toroidal {
                        row_distance = row_distance.min(dimension - row_distance);
                        column_distance = column_distance.min(dimension - column_distance);
                    }
                    let squared_distance = row_distance.pow(2) + column_distance.pow(2);
                    *weight = (-(squared_distance as f64) / (2.0 * config.sigma.powi(2))).exp();
                }
                let normalizer: f64 = row.iter().sum();
                for weight in row {
                    *weight /= normalizer;
                }
            }

            let mut expected_scores = Vec::new();
            for target in 0..cell_count {
                let background: f64 = (0..cell_count)
                    .map(|source| occupancy[source] as f64 * dense_kernel[source][target])
                    .sum();
                let evidence: f64 = word_incidence
                    .iter()
                    .map(|(&source, &count)| count as f64 * dense_kernel[source][target])
                    .sum();
                assert!((folder.smoothed_occupancy[target] - background).abs() < 1e-12);
                expected_scores.push((target as u32, evidence / background.sqrt()));
            }
            expected_scores
                .sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
            let expected_positions: Vec<_> = expected_scores
                .into_iter()
                .take(9)
                .map(|(position, _)| position)
                .collect();

            assert_eq!(folder.fold(&word_incidence, 9), expected_positions);
            assert!((folder.smoothed_occupancy.iter().sum::<f64>() - expected_mass).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_sigma_preserves_observed_cells_and_smoothing_has_bounded_support() {
        let mut occupancy = vec![0; 121];
        occupancy[0] = 2;
        let word_incidence = BTreeMap::from([(0, 2)]);
        let legacy = SpatialFolder::new(&occupancy, 11, true, &FoldingConfig::incidence());
        let toroidal = SpatialFolder::new(&occupancy, 11, true, &FoldingConfig::gaussian());
        let planar = SpatialFolder::new(&occupancy, 11, false, &FoldingConfig::gaussian());

        assert_eq!(legacy.fold(&word_incidence, 121), vec![0]);
        assert_eq!(toroidal.fold(&word_incidence, 121).len(), 49);
        assert_eq!(toroidal.fold(&word_incidence, 1), vec![0]);
        assert!(toroidal.fold(&word_incidence, 121).contains(&120));
        assert_eq!(planar.fold(&word_incidence, 121).len(), 16);
        assert!(!planar.fold(&word_incidence, 121).contains(&120));
    }

    #[test]
    fn single_cell_map_preserves_mass_without_duplicate_wrapped_offsets() {
        let folder = SpatialFolder::new(&[2], 1, true, &FoldingConfig::gaussian());
        let word_incidence = BTreeMap::from([(0, 2)]);

        assert_eq!(folder.fold(&word_incidence, 100), vec![0]);
        assert_eq!(folder.smoothed_occupancy, vec![2.0]);
    }
}
