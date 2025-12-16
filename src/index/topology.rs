//! Toroidal grid topology for SOM-based semantic space.
//!
//! The Self-Organizing Map creates a 2D manifold where semantic similarity
//! correlates with spatial proximity. This module provides operations on
//! this toroidal (wrapped) topology.

/// A position on the 2D toroidal grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPosition {
    /// Row coordinate (0 to dimension-1).
    pub row: u32,
    /// Column coordinate (0 to dimension-1).
    pub col: u32,
}

impl GridPosition {
    /// Create a new grid position.
    pub fn new(row: u32, col: u32) -> Self {
        Self { row, col }
    }

    /// Convert from linear index to 2D position.
    #[inline]
    pub fn from_linear(index: u32, dimension: u32) -> Self {
        Self {
            row: index / dimension,
            col: index % dimension,
        }
    }

    /// Convert to linear index.
    #[inline]
    pub fn to_linear(&self, dimension: u32) -> u32 {
        self.row * dimension + self.col
    }
}

/// Toroidal grid operations for SOM topology.
///
/// The grid wraps around at edges (torus topology), which:
/// 1. Eliminates edge effects in the SOM
/// 2. Creates a continuous manifold without boundaries
/// 3. Matches the topology used during SOM training
#[derive(Debug, Clone)]
pub struct ToroidalGrid {
    dimension: u32,
    grid_size: u32,
}

impl ToroidalGrid {
    /// Create a new toroidal grid with the given dimension.
    pub fn new(dimension: u32) -> Self {
        Self {
            dimension,
            grid_size: dimension * dimension,
        }
    }

    /// Get the grid dimension (one side of the square grid).
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Get the total grid size (dimension^2).
    pub fn grid_size(&self) -> u32 {
        self.grid_size
    }

    /// Convert linear index to 2D position.
    #[inline]
    pub fn to_2d(&self, index: u32) -> GridPosition {
        GridPosition::from_linear(index, self.dimension)
    }

    /// Convert 2D position to linear index.
    #[inline]
    pub fn to_linear(&self, pos: GridPosition) -> u32 {
        pos.to_linear(self.dimension)
    }

    /// Compute wrapped (toroidal) Euclidean distance between two positions.
    ///
    /// On a torus, we consider paths that wrap around edges. The distance
    /// is the minimum of all possible paths.
    #[inline]
    pub fn wrapped_distance(&self, p1: u32, p2: u32) -> f64 {
        let pos1 = self.to_2d(p1);
        let pos2 = self.to_2d(p2);
        self.wrapped_distance_2d(pos1, pos2)
    }

    /// Compute wrapped distance between two 2D positions.
    #[inline]
    pub fn wrapped_distance_2d(&self, p1: GridPosition, p2: GridPosition) -> f64 {
        let dim = self.dimension as i32;

        // Compute row distance (minimum of direct and wrapped)
        let dr1 = (p1.row as i32 - p2.row as i32).abs();
        let dr2 = dim - dr1;
        let dr = dr1.min(dr2) as f64;

        // Compute column distance (minimum of direct and wrapped)
        let dc1 = (p1.col as i32 - p2.col as i32).abs();
        let dc2 = dim - dc1;
        let dc = dc1.min(dc2) as f64;

        (dr * dr + dc * dc).sqrt()
    }

    /// Compute wrapped Manhattan distance.
    #[inline]
    pub fn wrapped_manhattan(&self, p1: u32, p2: u32) -> u32 {
        let pos1 = self.to_2d(p1);
        let pos2 = self.to_2d(p2);

        let dim = self.dimension as i32;

        let dr1 = (pos1.row as i32 - pos2.row as i32).abs();
        let dr = dr1.min(dim - dr1) as u32;

        let dc1 = (pos1.col as i32 - pos2.col as i32).abs();
        let dc = dc1.min(dim - dc1) as u32;

        dr + dc
    }

    /// Get all positions within a given Euclidean radius (toroidal).
    ///
    /// This is the neighborhood expansion used by Cortical.io with radius 1.8.
    pub fn neighborhood(&self, center: u32, radius: f64) -> Vec<u32> {
        let center_pos = self.to_2d(center);
        let radius_ceil = radius.ceil() as i32;
        let dim = self.dimension as i32;

        let mut neighbors = Vec::new();

        // Search in a square around the center (accounting for wrapping)
        for dr in -radius_ceil..=radius_ceil {
            for dc in -radius_ceil..=radius_ceil {
                // Wrap coordinates
                let row = ((center_pos.row as i32 + dr) % dim + dim) % dim;
                let col = ((center_pos.col as i32 + dc) % dim + dim) % dim;

                let neighbor = GridPosition::new(row as u32, col as u32);

                if self.wrapped_distance_2d(center_pos, neighbor) < radius {
                    neighbors.push(self.to_linear(neighbor));
                }
            }
        }

        neighbors
    }

    /// Get positions within radius with their distances (for weighted operations).
    pub fn neighborhood_weighted(&self, center: u32, radius: f64) -> Vec<(u32, f64)> {
        let center_pos = self.to_2d(center);
        let radius_ceil = radius.ceil() as i32;
        let dim = self.dimension as i32;

        let mut neighbors = Vec::new();

        for dr in -radius_ceil..=radius_ceil {
            for dc in -radius_ceil..=radius_ceil {
                let row = ((center_pos.row as i32 + dr) % dim + dim) % dim;
                let col = ((center_pos.col as i32 + dc) % dim + dim) % dim;

                let neighbor = GridPosition::new(row as u32, col as u32);
                let dist = self.wrapped_distance_2d(center_pos, neighbor);

                if dist < radius {
                    neighbors.push((self.to_linear(neighbor), dist));
                }
            }
        }

        neighbors
    }

    /// Apply Gaussian kernel weight based on distance.
    ///
    /// Uses the same exponential decay as Cortical.io's CreateKernel:
    /// weight = max_val / 2^distance
    #[inline]
    pub fn gaussian_weight(&self, distance: f64, sigma: f64) -> f64 {
        (-distance * distance / (2.0 * sigma * sigma)).exp()
    }

    /// Apply exponential decay weight (Cortical.io style).
    ///
    /// weight = 1 / 2^distance
    #[inline]
    pub fn exponential_weight(&self, distance: f64) -> f64 {
        2.0_f64.powf(-distance)
    }

    /// Compute the geodesic (shortest path on torus surface) between two points.
    ///
    /// On a flat torus embedded in 3D, this follows the surface.
    pub fn geodesic_distance(&self, p1: u32, p2: u32) -> f64 {
        // For a flat torus, geodesic = wrapped Euclidean
        // For a curved torus, we'd need more complex computation
        self.wrapped_distance(p1, p2)
    }

    /// Get the 8-connected neighbors (Moore neighborhood) with wrapping.
    pub fn moore_neighborhood(&self, center: u32) -> [u32; 8] {
        let pos = self.to_2d(center);
        let dim = self.dimension;

        let row_prev = if pos.row == 0 { dim - 1 } else { pos.row - 1 };
        let row_next = if pos.row == dim - 1 { 0 } else { pos.row + 1 };
        let col_prev = if pos.col == 0 { dim - 1 } else { pos.col - 1 };
        let col_next = if pos.col == dim - 1 { 0 } else { pos.col + 1 };

        [
            row_prev * dim + col_prev,  // top-left
            row_prev * dim + pos.col,   // top
            row_prev * dim + col_next,  // top-right
            pos.row * dim + col_prev,   // left
            pos.row * dim + col_next,   // right
            row_next * dim + col_prev,  // bottom-left
            row_next * dim + pos.col,   // bottom
            row_next * dim + col_next,  // bottom-right
        ]
    }

    /// Get the 4-connected neighbors (Von Neumann neighborhood) with wrapping.
    pub fn von_neumann_neighborhood(&self, center: u32) -> [u32; 4] {
        let pos = self.to_2d(center);
        let dim = self.dimension;

        let row_prev = if pos.row == 0 { dim - 1 } else { pos.row - 1 };
        let row_next = if pos.row == dim - 1 { 0 } else { pos.row + 1 };
        let col_prev = if pos.col == 0 { dim - 1 } else { pos.col - 1 };
        let col_next = if pos.col == dim - 1 { 0 } else { pos.col + 1 };

        [
            row_prev * dim + pos.col,   // top
            pos.row * dim + col_prev,   // left
            pos.row * dim + col_next,   // right
            row_next * dim + pos.col,   // bottom
        ]
    }

    /// Compute angle from center to position (in radians, 0 = right, π/2 = down).
    pub fn angle_to(&self, from: u32, to: u32) -> f64 {
        let p1 = self.to_2d(from);
        let p2 = self.to_2d(to);

        let dim = self.dimension as f64;

        // Compute wrapped delta (choose shortest path)
        let mut dr = p2.row as f64 - p1.row as f64;
        if dr.abs() > dim / 2.0 {
            dr = if dr > 0.0 { dr - dim } else { dr + dim };
        }

        let mut dc = p2.col as f64 - p1.col as f64;
        if dc.abs() > dim / 2.0 {
            dc = if dc > 0.0 { dc - dim } else { dc + dim };
        }

        dr.atan2(dc)
    }

    /// Check if three positions are collinear (on a geodesic).
    pub fn are_collinear(&self, p1: u32, p2: u32, p3: u32, tolerance: f64) -> bool {
        let d12 = self.wrapped_distance(p1, p2);
        let d23 = self.wrapped_distance(p2, p3);
        let d13 = self.wrapped_distance(p1, p3);

        // On a geodesic: d13 ≈ d12 + d23 or permutations
        let sum_min = d12.min(d23).min(d13);
        let sum_mid = d12.max(d23).max(d13).min(d12.min(d23).max(d13).max(d12.max(d23).min(d13)));
        let sum_max = d12.max(d23).max(d13);

        (sum_max - sum_min - sum_mid).abs() < tolerance
    }
}

/// Pre-computed distance matrix for fast lookups.
///
/// For small grids (≤256x256), pre-computing all pairwise distances
/// is feasible and provides O(1) distance queries.
pub struct DistanceMatrix {
    distances: Vec<f64>,
    dimension: u32,
}

impl DistanceMatrix {
    /// Create a new distance matrix (O(n²) space and time).
    pub fn new(grid: &ToroidalGrid) -> Self {
        let n = grid.grid_size() as usize;
        let mut distances = vec![0.0; n * n];

        for i in 0..n {
            for j in i..n {
                let dist = grid.wrapped_distance(i as u32, j as u32);
                distances[i * n + j] = dist;
                distances[j * n + i] = dist;
            }
        }

        Self {
            distances,
            dimension: grid.dimension(),
        }
    }

    /// Get pre-computed distance.
    #[inline]
    pub fn get(&self, p1: u32, p2: u32) -> f64 {
        let n = (self.dimension * self.dimension) as usize;
        self.distances[p1 as usize * n + p2 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrapped_distance() {
        let grid = ToroidalGrid::new(128);

        // Same position = 0
        assert_eq!(grid.wrapped_distance(0, 0), 0.0);

        // Adjacent = 1
        assert_eq!(grid.wrapped_distance(0, 1), 1.0);
        assert_eq!(grid.wrapped_distance(0, 128), 1.0);

        // Diagonal = sqrt(2)
        assert!((grid.wrapped_distance(0, 129) - 2.0_f64.sqrt()).abs() < 1e-10);

        // Wrapping: position 0 and position 127 are distance 1 apart
        assert_eq!(grid.wrapped_distance(0, 127), 1.0);
    }

    #[test]
    fn test_neighborhood() {
        let grid = ToroidalGrid::new(128);

        // Radius 1.5 should give 5 positions (center + 4 adjacent)
        let n = grid.neighborhood(1000, 1.5);
        assert_eq!(n.len(), 5);

        // Radius 1.8 should give ~9-13 positions (Cortical.io default)
        let n = grid.neighborhood(1000, 1.8);
        assert!(n.len() >= 9 && n.len() <= 13);
    }

    #[test]
    fn test_wrapping_at_edges() {
        let grid = ToroidalGrid::new(128);

        // Corner position (0,0) neighbors should wrap
        let n = grid.moore_neighborhood(0);
        assert!(n.contains(&127)); // left wraps to right edge
        assert!(n.contains(&(127 * 128))); // top wraps to bottom edge
        assert!(n.contains(&(128 * 128 - 1))); // diagonal wraps
    }
}
