//! Jaccard similarity for SDRs.

use crate::fingerprint::Sdr;
use crate::similarity::SimilarityMeasure;

/// Jaccard similarity measure.
///
/// |A ∩ B| / |A ∪ B|
#[derive(Debug, Clone, Copy, Default)]
pub struct JaccardSimilarity;

impl SimilarityMeasure for JaccardSimilarity {
    fn similarity(&self, a: &Sdr, b: &Sdr) -> f64 {
        a.jaccard_similarity(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical() {
        let a = Sdr::from_positions(&[1, 2, 3, 4, 5], 100);
        let sim = JaccardSimilarity.similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_overlap() {
        let a = Sdr::from_positions(&[1, 2, 3], 100);
        let b = Sdr::from_positions(&[4, 5, 6], 100);
        let sim = JaccardSimilarity.similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_partial_overlap() {
        let a = Sdr::from_positions(&[1, 2, 3, 4], 100);
        let b = Sdr::from_positions(&[3, 4, 5, 6], 100);

        let sim = JaccardSimilarity.similarity(&a, &b);
        // Intersection = {3, 4}, Union = {1, 2, 3, 4, 5, 6}
        // Jaccard = 2/6 = 1/3
        assert!((sim - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_subset() {
        let a = Sdr::from_positions(&[1, 2], 100);
        let b = Sdr::from_positions(&[1, 2, 3, 4], 100);

        let sim = JaccardSimilarity.similarity(&a, &b);
        // Intersection = {1, 2}, Union = {1, 2, 3, 4}
        // Jaccard = 2/4 = 0.5
        assert!((sim - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty() {
        let a = Sdr::new(100);
        let b = Sdr::new(100);
        let sim = JaccardSimilarity.similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }
}
