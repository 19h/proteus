//! Low-rank word-context projection by orthogonal subspace iteration.
//!
//! This computes coordinates in an approximate right singular subspace. It does
//! not compute singular values or claim an exact truncated SVD.

use rayon::prelude::*;

pub(super) type SparseRow = Vec<(usize, f32)>;

/// Return row coordinates A * Q, with Q an approximate dominant right subspace.
/// Independent output columns may run in parallel; each sum has fixed row order.
pub(super) fn project(
    matrix: &[SparseRow],
    random_projection: &[Vec<f32>],
    iterations: usize,
) -> Vec<Vec<f32>> {
    let feature_count = random_projection.len();
    let dimension = random_projection[0].len();
    let mut basis: Vec<Vec<f32>> = (0..dimension)
        .map(|column| random_projection.iter().map(|row| row[column]).collect())
        .collect();
    orthonormalize(&mut basis);

    for _ in 0..iterations {
        let mut left_basis = multiply(matrix, &basis);
        orthonormalize(&mut left_basis);
        basis = multiply_transpose(matrix, &left_basis, feature_count);
        orthonormalize(&mut basis);
    }

    let columns = multiply(matrix, &basis);
    (0..matrix.len())
        .map(|row| columns.iter().map(|column| column[row]).collect())
        .collect()
}

fn multiply(matrix: &[SparseRow], columns: &[Vec<f32>]) -> Vec<Vec<f32>> {
    columns
        .par_iter()
        .map(|column| {
            matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&(feature, weight)| weight as f64 * column[feature] as f64)
                        .sum::<f64>() as f32
                })
                .collect()
        })
        .collect()
}

fn multiply_transpose(
    matrix: &[SparseRow],
    columns: &[Vec<f32>],
    feature_count: usize,
) -> Vec<Vec<f32>> {
    columns
        .par_iter()
        .map(|column| {
            let mut output = vec![0.0_f64; feature_count];
            for (row, &coordinate) in matrix.iter().zip(column) {
                for &(feature, weight) in row {
                    output[feature] += weight as f64 * coordinate as f64;
                }
            }
            output.into_iter().map(|value| value as f32).collect()
        })
        .collect()
}

fn orthonormalize(columns: &mut [Vec<f32>]) {
    for column_index in 0..columns.len() {
        let (previous_columns, remaining_columns) = columns.split_at_mut(column_index);
        let current_column = &mut remaining_columns[0];
        let initial_norm = squared_norm(current_column).sqrt();

        // Reorthogonalization limits accumulated loss of orthogonality in f32.
        for _ in 0..2 {
            for basis_column in previous_columns.iter() {
                let projection = current_column
                    .iter()
                    .zip(basis_column)
                    .map(|(&value, &basis)| value as f64 * basis as f64)
                    .sum::<f64>();
                for (value, &basis) in current_column.iter_mut().zip(basis_column) {
                    *value = (*value as f64 - projection * basis as f64) as f32;
                }
            }
        }

        let residual_norm = squared_norm(current_column).sqrt();
        if residual_norm <= initial_norm * 1e-6 || residual_norm == 0.0 {
            // A rank-deficient input supplies no evidence for this coordinate.
            current_column.fill(0.0);
        } else {
            for value in current_column {
                *value = (*value as f64 / residual_norm) as f32;
            }
        }
    }
}

fn squared_norm(vector: &[f32]) -> f64 {
    vector.iter().map(|&value| (value as f64).powi(2)).sum()
}

/// Replace row coordinates by an orthonormal basis of their column space.
/// Pairwise row products then describe the subspace projector, removing
/// variation in singular-direction scale before per-word normalization.
pub(super) fn whiten_rows(rows: &mut [Vec<f32>]) {
    let Some(first_row) = rows.first() else {
        return;
    };
    let dimension = first_row.len();
    let mut columns: Vec<Vec<f32>> = (0..dimension)
        .map(|column| rows.iter().map(|row| row[column]).collect())
        .collect();
    orthonormalize(&mut columns);
    for (row_index, row) in rows.iter_mut().enumerate() {
        for (column_index, coordinate) in row.iter_mut().enumerate() {
            *coordinate = columns[column_index][row_index];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_rank_projection_preserves_row_inner_products() {
        let matrix = vec![vec![(0, 3.0), (1, 1.0)], vec![(0, 1.0), (1, 2.0)]];
        let starting_projection = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
        let coordinates = project(&matrix, &starting_projection, 2);
        let inner_product = |left: &[f32], right: &[f32]| {
            left.iter()
                .zip(right)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>()
        };

        assert!((inner_product(&coordinates[0], &coordinates[0]) - 10.0).abs() < 2e-6);
        assert!((inner_product(&coordinates[0], &coordinates[1]) - 5.0).abs() < 2e-6);
        assert!((inner_product(&coordinates[1], &coordinates[1]) - 5.0).abs() < 2e-6);
    }

    #[test]
    fn power_iteration_finds_the_dominant_diagonal_subspace() {
        let matrix = vec![vec![(0, 8.0)], vec![(1, 2.0)], vec![(2, 1.0)]];
        let starting_projection = vec![vec![1.0], vec![1.0], vec![1.0]];
        let coordinates = project(&matrix, &starting_projection, 4);

        assert!((coordinates[0][0].abs() - 8.0).abs() < 1e-6);
        assert!(coordinates[1][0].abs() < 1e-4);
        assert!(coordinates[2][0].abs() < 1e-6);
    }

    #[test]
    fn rank_deficiency_and_empty_rows_do_not_create_coordinates() {
        let matrix = vec![vec![(0, 1.0)], vec![(0, 2.0)], Vec::new()];
        let starting_projection = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
        let coordinates = project(&matrix, &starting_projection, 2);

        assert_eq!(coordinates[2], vec![0.0, 0.0]);
        assert!(coordinates.iter().flatten().all(|value| value.is_finite()));
        assert!(coordinates.iter().all(|row| row[1] == 0.0));
    }
}

#[cfg(test)]
mod whitening_tests {
    use super::*;

    #[test]
    fn whitened_row_products_match_the_analytic_subspace_projector() {
        // A(A^T A)^-1 A^T for this matrix has diagonal 2/3,
        // off-diagonal (0,1) = -1/3, and (0,2) = (1,2) = 1/3.
        let mut rows = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        whiten_rows(&mut rows);
        let expected = [[2.0, -1.0, 1.0], [-1.0, 2.0, 1.0], [1.0, 1.0, 2.0]];
        for (left_index, left) in rows.iter().enumerate() {
            for (right_index, right) in rows.iter().enumerate() {
                let product: f64 = left
                    .iter()
                    .zip(right)
                    .map(|(&left, &right)| left as f64 * right as f64)
                    .sum();
                assert!((product - expected[left_index][right_index] / 3.0).abs() < 2e-7);
            }
        }
    }
}
