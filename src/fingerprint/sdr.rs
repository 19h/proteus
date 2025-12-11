//! Sparse Distributed Representation (SDR) implementation.

use crate::roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::ops::{BitAnd, BitOr, BitXor};

/// A Sparse Distributed Representation (SDR).
///
/// SDRs are large binary vectors with only a small fraction of bits active.
/// They enable efficient semantic operations through bit-level manipulation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sdr {
    /// The active bit positions, stored in a compressed bitmap.
    bits: RoaringBitmap,
    /// Total size of the representation (number of possible positions).
    size: u32,
}

impl Sdr {
    /// Creates an empty SDR with the given size.
    pub fn new(size: u32) -> Self {
        Self {
            bits: RoaringBitmap::new(),
            size,
        }
    }

    /// Creates an SDR from a list of active positions.
    pub fn from_positions(positions: &[u32], size: u32) -> Self {
        let bits: RoaringBitmap = positions.iter().copied().collect();
        Self { bits, size }
    }

    /// Creates an SDR from a RoaringBitmap.
    pub fn from_bitmap(bits: RoaringBitmap, size: u32) -> Self {
        Self { bits, size }
    }

    /// Returns the total size of the SDR.
    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Returns the number of active bits.
    #[inline]
    pub fn cardinality(&self) -> u64 {
        self.bits.len()
    }

    /// Returns the sparsity (fraction of active bits).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        self.bits.len() as f64 / self.size as f64
    }

    /// Checks if a bit is active at the given position.
    #[inline]
    pub fn contains(&self, position: u32) -> bool {
        self.bits.contains(position)
    }

    /// Sets a bit at the given position.
    #[inline]
    pub fn insert(&mut self, position: u32) {
        if position < self.size {
            self.bits.insert(position);
        }
    }

    /// Clears a bit at the given position.
    #[inline]
    pub fn remove(&mut self, position: u32) {
        self.bits.remove(position);
    }

    /// Returns an iterator over active bit positions.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits.iter()
    }

    /// Converts to a vector of active positions.
    pub fn to_positions(&self) -> Vec<u32> {
        self.bits.iter().collect()
    }

    /// Returns the underlying RoaringBitmap.
    pub fn bitmap(&self) -> &RoaringBitmap {
        &self.bits
    }

    /// Returns a mutable reference to the underlying RoaringBitmap.
    pub fn bitmap_mut(&mut self) -> &mut RoaringBitmap {
        &mut self.bits
    }

    /// Computes the overlap (AND) with another SDR.
    pub fn overlap(&self, other: &Sdr) -> Sdr {
        let bits = &self.bits & &other.bits;
        Sdr::from_bitmap(bits, self.size.max(other.size))
    }

    /// Computes the union (OR) with another SDR.
    pub fn union(&self, other: &Sdr) -> Sdr {
        let bits = &self.bits | &other.bits;
        Sdr::from_bitmap(bits, self.size.max(other.size))
    }

    /// Computes the symmetric difference (XOR) with another SDR.
    pub fn xor(&self, other: &Sdr) -> Sdr {
        let bits = &self.bits ^ &other.bits;
        Sdr::from_bitmap(bits, self.size.max(other.size))
    }

    /// Computes the number of overlapping bits with another SDR.
    #[inline]
    pub fn overlap_count(&self, other: &Sdr) -> u64 {
        (&self.bits & &other.bits).len()
    }

    /// Computes the Jaccard similarity with another SDR.
    pub fn jaccard_similarity(&self, other: &Sdr) -> f64 {
        let intersection = (&self.bits & &other.bits).len();
        let union = (&self.bits | &other.bits).len();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Computes the overlap similarity (overlap / min cardinality).
    pub fn overlap_similarity(&self, other: &Sdr) -> f64 {
        let intersection = (&self.bits & &other.bits).len();
        let min_card = self.cardinality().min(other.cardinality());

        if min_card == 0 {
            0.0
        } else {
            intersection as f64 / min_card as f64
        }
    }

    /// Computes the cosine similarity with another SDR.
    ///
    /// For binary vectors, this is overlap / sqrt(card_a * card_b).
    pub fn cosine_similarity(&self, other: &Sdr) -> f64 {
        let intersection = (&self.bits & &other.bits).len();
        let product = self.cardinality() * other.cardinality();

        if product == 0 {
            0.0
        } else {
            intersection as f64 / (product as f64).sqrt()
        }
    }

    /// Sparsifies the SDR to have at most `max_bits` active.
    ///
    /// Keeps bits based on their position (arbitrary but deterministic).
    pub fn sparsify(&mut self, max_bits: usize) {
        if self.cardinality() as usize <= max_bits {
            return;
        }

        let positions: Vec<u32> = self.bits.iter().take(max_bits).collect();
        self.bits.clear();
        for pos in positions {
            self.bits.insert(pos);
        }
    }

    /// Creates a binary dense vector representation.
    pub fn to_dense(&self) -> Vec<u8> {
        let mut dense = vec![0u8; self.size as usize];
        for pos in self.bits.iter() {
            dense[pos as usize] = 1;
        }
        dense
    }

    /// Creates an SDR from a dense vector.
    pub fn from_dense(dense: &[u8]) -> Self {
        let size = dense.len() as u32;
        let bits: RoaringBitmap = dense
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0)
            .map(|(i, _)| i as u32)
            .collect();

        Self { bits, size }
    }

    /// Checks if this SDR is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Clears all bits.
    pub fn clear(&mut self) {
        self.bits.clear();
    }
}

impl BitAnd for &Sdr {
    type Output = Sdr;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.overlap(rhs)
    }
}

impl BitOr for &Sdr {
    type Output = Sdr;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl BitXor for &Sdr {
    type Output = Sdr;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}

impl Default for Sdr {
    fn default() -> Self {
        Self::new(16384) // Default 128x128 grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdr_creation() {
        let sdr = Sdr::new(1000);
        assert_eq!(sdr.size(), 1000);
        assert_eq!(sdr.cardinality(), 0);
        assert!(sdr.is_empty());
    }

    #[test]
    fn test_from_positions() {
        let sdr = Sdr::from_positions(&[1, 5, 10, 100], 1000);
        assert_eq!(sdr.cardinality(), 4);
        assert!(sdr.contains(1));
        assert!(sdr.contains(5));
        assert!(!sdr.contains(2));
    }

    #[test]
    fn test_insert_remove() {
        let mut sdr = Sdr::new(100);
        sdr.insert(50);
        assert!(sdr.contains(50));
        sdr.remove(50);
        assert!(!sdr.contains(50));
    }

    #[test]
    fn test_sparsity() {
        let sdr = Sdr::from_positions(&[0, 1, 2, 3, 4], 100);
        assert!((sdr.sparsity() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_overlap() {
        let a = Sdr::from_positions(&[1, 2, 3], 100);
        let b = Sdr::from_positions(&[2, 3, 4], 100);
        let overlap = a.overlap(&b);

        assert_eq!(overlap.cardinality(), 2);
        assert!(overlap.contains(2));
        assert!(overlap.contains(3));
    }

    #[test]
    fn test_union() {
        let a = Sdr::from_positions(&[1, 2], 100);
        let b = Sdr::from_positions(&[3, 4], 100);
        let union = a.union(&b);

        assert_eq!(union.cardinality(), 4);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = Sdr::from_positions(&[1, 2, 3, 4], 100);
        let b = Sdr::from_positions(&[3, 4, 5, 6], 100);

        // Intersection: {3, 4}, Union: {1, 2, 3, 4, 5, 6}
        // Jaccard = 2/6 = 0.333...
        let sim = a.jaccard_similarity(&b);
        assert!((sim - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Sdr::from_positions(&[1, 2, 3, 4], 100);
        let b = Sdr::from_positions(&[3, 4, 5, 6], 100);

        // Intersection: 2, sqrt(4*4) = 4
        // Cosine = 2/4 = 0.5
        let sim = a.cosine_similarity(&b);
        assert!((sim - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sparsify() {
        let mut sdr = Sdr::from_positions(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100);
        sdr.sparsify(5);
        assert_eq!(sdr.cardinality(), 5);
    }

    #[test]
    fn test_dense_conversion() {
        let sdr = Sdr::from_positions(&[0, 5, 9], 10);
        let dense = sdr.to_dense();

        assert_eq!(dense.len(), 10);
        assert_eq!(dense[0], 1);
        assert_eq!(dense[1], 0);
        assert_eq!(dense[5], 1);
        assert_eq!(dense[9], 1);

        let recovered = Sdr::from_dense(&dense);
        assert_eq!(recovered, sdr);
    }

    #[test]
    fn test_bitwise_operators() {
        let a = Sdr::from_positions(&[1, 2, 3], 100);
        let b = Sdr::from_positions(&[2, 3, 4], 100);

        let and = &a & &b;
        assert_eq!(and.cardinality(), 2);

        let or = &a | &b;
        assert_eq!(or.cardinality(), 4);

        let xor = &a ^ &b;
        assert_eq!(xor.cardinality(), 2);
    }
}
