//! Cosine similarity for SDRs.

use crate::fingerprint::Sdr;
use crate::similarity::SimilarityMeasure;

/// Cosine similarity measure.
///
/// For binary vectors: overlap / sqrt(|A| * |B|)
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineSimilarity;

impl SimilarityMeasure for CosineSimilarity {
    fn similarity(&self, a: &Sdr, b: &Sdr) -> f64 {
        a.cosine_similarity(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical() {
        let a = Sdr::from_positions(&[1, 2, 3, 4, 5], 100);
        let sim = CosineSimilarity.similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_overlap() {
        let a = Sdr::from_positions(&[1, 2, 3], 100);
        let b = Sdr::from_positions(&[4, 5, 6], 100);
        let sim = CosineSimilarity.similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_partial_overlap() {
        let a = Sdr::from_positions(&[1, 2, 3, 4], 100);
        let b = Sdr::from_positions(&[3, 4, 5, 6], 100);

        let sim = CosineSimilarity.similarity(&a, &b);
        // Overlap = 2, sqrt(4*4) = 4, so cosine = 0.5
        assert!((sim - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty() {
        let a = Sdr::new(100);
        let b = Sdr::from_positions(&[1, 2, 3], 100);
        let sim = CosineSimilarity.similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let a = Sdr::from_positions(&[1, 2, 3, 4], 100);
        let b = Sdr::from_positions(&[3, 4, 5, 6], 100);

        let sim = CosineSimilarity.similarity(&a, &b);
        let dist = CosineSimilarity.distance(&a, &b);

        assert!((sim + dist - 1.0).abs() < 1e-10);
    }
}
