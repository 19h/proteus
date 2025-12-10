//! Overlap similarity for SDRs.

use crate::fingerprint::Sdr;
use crate::similarity::SimilarityMeasure;

/// Overlap similarity measure (Szymkiewicz-Simpson coefficient).
///
/// |A ∩ B| / min(|A|, |B|)
///
/// This measure is useful when comparing SDRs of different sizes,
/// as it measures how much the smaller set is contained in the larger.
#[derive(Debug, Clone, Copy, Default)]
pub struct OverlapSimilarity;

impl SimilarityMeasure for OverlapSimilarity {
    fn similarity(&self, a: &Sdr, b: &Sdr) -> f64 {
        a.overlap_similarity(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical() {
        let a = Sdr::from_positions(&[1, 2, 3, 4, 5], 100);
        let sim = OverlapSimilarity.similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_overlap() {
        let a = Sdr::from_positions(&[1, 2, 3], 100);
        let b = Sdr::from_positions(&[4, 5, 6], 100);
        let sim = OverlapSimilarity.similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_partial_overlap() {
        let a = Sdr::from_positions(&[1, 2, 3, 4], 100);
        let b = Sdr::from_positions(&[3, 4, 5, 6], 100);

        let sim = OverlapSimilarity.similarity(&a, &b);
        // Intersection = {3, 4}, min(4, 4) = 4
        // Overlap = 2/4 = 0.5
        assert!((sim - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_complete_subset() {
        let a = Sdr::from_positions(&[1, 2], 100);
        let b = Sdr::from_positions(&[1, 2, 3, 4], 100);

        let sim = OverlapSimilarity.similarity(&a, &b);
        // Intersection = {1, 2}, min(2, 4) = 2
        // Overlap = 2/2 = 1.0
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_subset() {
        let a = Sdr::from_positions(&[1, 2, 5], 100);
        let b = Sdr::from_positions(&[1, 2, 3, 4], 100);

        let sim = OverlapSimilarity.similarity(&a, &b);
        // Intersection = {1, 2}, min(3, 4) = 3
        // Overlap = 2/3
        assert!((sim - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty() {
        let a = Sdr::new(100);
        let b = Sdr::from_positions(&[1, 2, 3], 100);
        let sim = OverlapSimilarity.similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }
}
