//! Similarity measures for comparing SDRs.

mod cosine;
mod jaccard;
mod overlap;

pub use cosine::CosineSimilarity;
pub use jaccard::JaccardSimilarity;
pub use overlap::OverlapSimilarity;

use crate::fingerprint::Sdr;

/// Trait for similarity measures between SDRs.
pub trait SimilarityMeasure {
    /// Computes the similarity between two SDRs.
    ///
    /// Returns a value between 0.0 (completely different) and 1.0 (identical).
    fn similarity(&self, a: &Sdr, b: &Sdr) -> f64;

    /// Computes the distance between two SDRs.
    ///
    /// Default implementation: 1.0 - similarity.
    fn distance(&self, a: &Sdr, b: &Sdr) -> f64 {
        1.0 - self.similarity(a, b)
    }
}

/// Enum for different similarity measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityType {
    /// Cosine similarity.
    Cosine,
    /// Jaccard similarity.
    Jaccard,
    /// Overlap similarity (min-denominator).
    Overlap,
}

impl SimilarityType {
    /// Computes similarity using this measure.
    pub fn compute(&self, a: &Sdr, b: &Sdr) -> f64 {
        match self {
            SimilarityType::Cosine => a.cosine_similarity(b),
            SimilarityType::Jaccard => a.jaccard_similarity(b),
            SimilarityType::Overlap => a.overlap_similarity(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_types() {
        let a = Sdr::from_positions(&[1, 2, 3, 4], 100);
        let b = Sdr::from_positions(&[3, 4, 5, 6], 100);

        let cosine = SimilarityType::Cosine.compute(&a, &b);
        let jaccard = SimilarityType::Jaccard.compute(&a, &b);
        let overlap = SimilarityType::Overlap.compute(&a, &b);

        // All should be positive
        assert!(cosine > 0.0);
        assert!(jaccard > 0.0);
        assert!(overlap > 0.0);

        // Overlap should be highest for equal-sized sets with 50% overlap
        assert!(overlap >= jaccard);
    }
}
