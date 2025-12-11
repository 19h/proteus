//! SIMD-optimized operations for BitmapStore.
//!
//! This module provides vectorized implementations of common bitmap operations
//! using Rust's portable SIMD feature. These operations work on 1024-element
//! u64 arrays (8KB) which fit well in cache.

use std::simd::{u64x4, u64x8, Simd};
use std::simd::prelude::SimdUint;

use super::bitmap_store::BITMAP_LENGTH;

/// Number of u64 elements per SIMD vector (4 for 256-bit, 8 for 512-bit)
const LANES_4: usize = 4;
const LANES_8: usize = 8;

/// Compute population count (number of set bits) across an array of u64s using SIMD.
///
/// This uses vectorized counting where available and processes 4 u64s at a time.
/// The portable SIMD `count_ones()` method maps to efficient hardware instructions
/// on x86 (POPCNT/VPOPCNTQ), ARM (CNT/ADDV), etc.
#[inline]
pub fn popcount_simd(bits: &[u64; BITMAP_LENGTH]) -> u64 {
    // Process in chunks of 4 u64s (256-bit vectors)
    // 1024 / 4 = 256 iterations
    let mut total: u64x4 = Simd::splat(0);

    // Unroll by processing 4 vectors (16 u64s) per iteration for better ILP
    let mut i = 0;
    while i + 16 <= BITMAP_LENGTH {
        let v0 = u64x4::from_slice(&bits[i..i+4]);
        let v1 = u64x4::from_slice(&bits[i+4..i+8]);
        let v2 = u64x4::from_slice(&bits[i+8..i+12]);
        let v3 = u64x4::from_slice(&bits[i+12..i+16]);

        // count_ones returns a vector of the same type with counts per lane
        // Using reduce_sum would cause overflow issues, so we accumulate u64 counts
        let c0: u64x4 = v0.count_ones().cast();
        let c1: u64x4 = v1.count_ones().cast();
        let c2: u64x4 = v2.count_ones().cast();
        let c3: u64x4 = v3.count_ones().cast();

        total += c0 + c1 + c2 + c3;
        i += 16;
    }

    // Handle remaining (should be 0 for 1024, but be safe)
    let mut sum = total.reduce_sum();
    while i < BITMAP_LENGTH {
        sum += bits[i].count_ones() as u64;
        i += 1;
    }

    sum
}

/// Check if two bitmap arrays are disjoint (no common bits set) using SIMD.
///
/// Returns true if `(a & b) == 0` for all elements.
/// Uses early exit when any intersection is found.
#[inline]
pub fn is_disjoint_simd(bits1: &[u64; BITMAP_LENGTH], bits2: &[u64; BITMAP_LENGTH]) -> bool {
    // Process in chunks of 8 u64s (512-bit vectors) for wider coverage
    let mut i = 0;

    // Unroll to check 32 u64s per iteration for better memory throughput
    while i + 32 <= BITMAP_LENGTH {
        let a0 = u64x8::from_slice(&bits1[i..i+8]);
        let b0 = u64x8::from_slice(&bits2[i..i+8]);
        let a1 = u64x8::from_slice(&bits1[i+8..i+16]);
        let b1 = u64x8::from_slice(&bits2[i+8..i+16]);
        let a2 = u64x8::from_slice(&bits1[i+16..i+24]);
        let b2 = u64x8::from_slice(&bits2[i+16..i+24]);
        let a3 = u64x8::from_slice(&bits1[i+24..i+32]);
        let b3 = u64x8::from_slice(&bits2[i+24..i+32]);

        // OR all intersections together - if any bit is set, we have overlap
        let inter = (a0 & b0) | (a1 & b1) | (a2 & b2) | (a3 & b3);

        // Check if any lane has a set bit
        if inter.reduce_or() != 0 {
            return false;
        }
        i += 32;
    }

    // Handle remainder
    while i < BITMAP_LENGTH {
        if bits1[i] & bits2[i] != 0 {
            return false;
        }
        i += 1;
    }

    true
}

/// Check if bits1 is a subset of bits2 using SIMD.
///
/// Returns true if `(bits1 & bits2) == bits1` for all elements.
/// i.e., every bit set in bits1 is also set in bits2.
#[inline]
pub fn is_subset_simd(bits1: &[u64; BITMAP_LENGTH], bits2: &[u64; BITMAP_LENGTH]) -> bool {
    let mut i = 0;

    // Process in chunks of 8 u64s
    while i + 32 <= BITMAP_LENGTH {
        let a0 = u64x8::from_slice(&bits1[i..i+8]);
        let b0 = u64x8::from_slice(&bits2[i..i+8]);
        let a1 = u64x8::from_slice(&bits1[i+8..i+16]);
        let b1 = u64x8::from_slice(&bits2[i+8..i+16]);
        let a2 = u64x8::from_slice(&bits1[i+16..i+24]);
        let b2 = u64x8::from_slice(&bits2[i+16..i+24]);
        let a3 = u64x8::from_slice(&bits1[i+24..i+32]);
        let b3 = u64x8::from_slice(&bits2[i+24..i+32]);

        // For subset check: (a & b) must equal a
        // Equivalently, a & !b must be zero
        // If a has any bit that b doesn't have, a & !b will have that bit set
        let diff = (a0 & !b0) | (a1 & !b1) | (a2 & !b2) | (a3 & !b3);

        if diff.reduce_or() != 0 {
            return false;
        }
        i += 32;
    }

    // Handle remainder
    while i < BITMAP_LENGTH {
        if bits1[i] & !bits2[i] != 0 {
            return false;
        }
        i += 1;
    }

    true
}

/// Compute the intersection length (popcount of AND) between two bitmaps using SIMD.
#[inline]
pub fn intersection_len_simd(bits1: &[u64; BITMAP_LENGTH], bits2: &[u64; BITMAP_LENGTH]) -> u64 {
    let mut total: u64x4 = Simd::splat(0);
    let mut i = 0;

    while i + 16 <= BITMAP_LENGTH {
        let a0 = u64x4::from_slice(&bits1[i..i+4]);
        let b0 = u64x4::from_slice(&bits2[i..i+4]);
        let a1 = u64x4::from_slice(&bits1[i+4..i+8]);
        let b1 = u64x4::from_slice(&bits2[i+4..i+8]);
        let a2 = u64x4::from_slice(&bits1[i+8..i+12]);
        let b2 = u64x4::from_slice(&bits2[i+8..i+12]);
        let a3 = u64x4::from_slice(&bits1[i+12..i+16]);
        let b3 = u64x4::from_slice(&bits2[i+12..i+16]);

        let c0: u64x4 = (a0 & b0).count_ones().cast();
        let c1: u64x4 = (a1 & b1).count_ones().cast();
        let c2: u64x4 = (a2 & b2).count_ones().cast();
        let c3: u64x4 = (a3 & b3).count_ones().cast();

        total += c0 + c1 + c2 + c3;
        i += 16;
    }

    let mut sum = total.reduce_sum();
    while i < BITMAP_LENGTH {
        sum += (bits1[i] & bits2[i]).count_ones() as u64;
        i += 1;
    }

    sum
}

/// Perform bitwise OR between two bitmaps and count set bits, storing result in dest.
/// Returns the new cardinality.
#[inline]
pub fn or_assign_simd(dest: &mut [u64; BITMAP_LENGTH], src: &[u64; BITMAP_LENGTH]) -> u64 {
    let mut total: u64x4 = Simd::splat(0);
    let mut i = 0;

    while i + 16 <= BITMAP_LENGTH {
        // Load dest
        let d0 = u64x4::from_slice(&dest[i..i+4]);
        let d1 = u64x4::from_slice(&dest[i+4..i+8]);
        let d2 = u64x4::from_slice(&dest[i+8..i+12]);
        let d3 = u64x4::from_slice(&dest[i+12..i+16]);

        // Load src
        let s0 = u64x4::from_slice(&src[i..i+4]);
        let s1 = u64x4::from_slice(&src[i+4..i+8]);
        let s2 = u64x4::from_slice(&src[i+8..i+12]);
        let s3 = u64x4::from_slice(&src[i+12..i+16]);

        // OR
        let r0 = d0 | s0;
        let r1 = d1 | s1;
        let r2 = d2 | s2;
        let r3 = d3 | s3;

        // Store results
        dest[i..i+4].copy_from_slice(&r0.to_array());
        dest[i+4..i+8].copy_from_slice(&r1.to_array());
        dest[i+8..i+12].copy_from_slice(&r2.to_array());
        dest[i+12..i+16].copy_from_slice(&r3.to_array());

        // Count bits
        let c0: u64x4 = r0.count_ones().cast();
        let c1: u64x4 = r1.count_ones().cast();
        let c2: u64x4 = r2.count_ones().cast();
        let c3: u64x4 = r3.count_ones().cast();
        total += c0 + c1 + c2 + c3;

        i += 16;
    }

    let mut sum = total.reduce_sum();
    while i < BITMAP_LENGTH {
        dest[i] |= src[i];
        sum += dest[i].count_ones() as u64;
        i += 1;
    }

    sum
}

/// Perform bitwise AND between two bitmaps and count set bits, storing result in dest.
/// Returns the new cardinality.
#[inline]
pub fn and_assign_simd(dest: &mut [u64; BITMAP_LENGTH], src: &[u64; BITMAP_LENGTH]) -> u64 {
    let mut total: u64x4 = Simd::splat(0);
    let mut i = 0;

    while i + 16 <= BITMAP_LENGTH {
        let d0 = u64x4::from_slice(&dest[i..i+4]);
        let d1 = u64x4::from_slice(&dest[i+4..i+8]);
        let d2 = u64x4::from_slice(&dest[i+8..i+12]);
        let d3 = u64x4::from_slice(&dest[i+12..i+16]);

        let s0 = u64x4::from_slice(&src[i..i+4]);
        let s1 = u64x4::from_slice(&src[i+4..i+8]);
        let s2 = u64x4::from_slice(&src[i+8..i+12]);
        let s3 = u64x4::from_slice(&src[i+12..i+16]);

        let r0 = d0 & s0;
        let r1 = d1 & s1;
        let r2 = d2 & s2;
        let r3 = d3 & s3;

        dest[i..i+4].copy_from_slice(&r0.to_array());
        dest[i+4..i+8].copy_from_slice(&r1.to_array());
        dest[i+8..i+12].copy_from_slice(&r2.to_array());
        dest[i+12..i+16].copy_from_slice(&r3.to_array());

        let c0: u64x4 = r0.count_ones().cast();
        let c1: u64x4 = r1.count_ones().cast();
        let c2: u64x4 = r2.count_ones().cast();
        let c3: u64x4 = r3.count_ones().cast();
        total += c0 + c1 + c2 + c3;

        i += 16;
    }

    let mut sum = total.reduce_sum();
    while i < BITMAP_LENGTH {
        dest[i] &= src[i];
        sum += dest[i].count_ones() as u64;
        i += 1;
    }

    sum
}

/// Perform bitwise XOR between two bitmaps and count set bits, storing result in dest.
/// Returns the new cardinality.
#[inline]
pub fn xor_assign_simd(dest: &mut [u64; BITMAP_LENGTH], src: &[u64; BITMAP_LENGTH]) -> u64 {
    let mut total: u64x4 = Simd::splat(0);
    let mut i = 0;

    while i + 16 <= BITMAP_LENGTH {
        let d0 = u64x4::from_slice(&dest[i..i+4]);
        let d1 = u64x4::from_slice(&dest[i+4..i+8]);
        let d2 = u64x4::from_slice(&dest[i+8..i+12]);
        let d3 = u64x4::from_slice(&dest[i+12..i+16]);

        let s0 = u64x4::from_slice(&src[i..i+4]);
        let s1 = u64x4::from_slice(&src[i+4..i+8]);
        let s2 = u64x4::from_slice(&src[i+8..i+12]);
        let s3 = u64x4::from_slice(&src[i+12..i+16]);

        let r0 = d0 ^ s0;
        let r1 = d1 ^ s1;
        let r2 = d2 ^ s2;
        let r3 = d3 ^ s3;

        dest[i..i+4].copy_from_slice(&r0.to_array());
        dest[i+4..i+8].copy_from_slice(&r1.to_array());
        dest[i+8..i+12].copy_from_slice(&r2.to_array());
        dest[i+12..i+16].copy_from_slice(&r3.to_array());

        let c0: u64x4 = r0.count_ones().cast();
        let c1: u64x4 = r1.count_ones().cast();
        let c2: u64x4 = r2.count_ones().cast();
        let c3: u64x4 = r3.count_ones().cast();
        total += c0 + c1 + c2 + c3;

        i += 16;
    }

    let mut sum = total.reduce_sum();
    while i < BITMAP_LENGTH {
        dest[i] ^= src[i];
        sum += dest[i].count_ones() as u64;
        i += 1;
    }

    sum
}

/// Perform bitwise AND-NOT (dest & !src) between two bitmaps and count set bits.
/// Returns the new cardinality.
#[inline]
pub fn sub_assign_simd(dest: &mut [u64; BITMAP_LENGTH], src: &[u64; BITMAP_LENGTH]) -> u64 {
    let mut total: u64x4 = Simd::splat(0);
    let mut i = 0;

    while i + 16 <= BITMAP_LENGTH {
        let d0 = u64x4::from_slice(&dest[i..i+4]);
        let d1 = u64x4::from_slice(&dest[i+4..i+8]);
        let d2 = u64x4::from_slice(&dest[i+8..i+12]);
        let d3 = u64x4::from_slice(&dest[i+12..i+16]);

        let s0 = u64x4::from_slice(&src[i..i+4]);
        let s1 = u64x4::from_slice(&src[i+4..i+8]);
        let s2 = u64x4::from_slice(&src[i+8..i+12]);
        let s3 = u64x4::from_slice(&src[i+12..i+16]);

        // AND-NOT: dest & !src
        let r0 = d0 & !s0;
        let r1 = d1 & !s1;
        let r2 = d2 & !s2;
        let r3 = d3 & !s3;

        dest[i..i+4].copy_from_slice(&r0.to_array());
        dest[i+4..i+8].copy_from_slice(&r1.to_array());
        dest[i+8..i+12].copy_from_slice(&r2.to_array());
        dest[i+12..i+16].copy_from_slice(&r3.to_array());

        let c0: u64x4 = r0.count_ones().cast();
        let c1: u64x4 = r1.count_ones().cast();
        let c2: u64x4 = r2.count_ones().cast();
        let c3: u64x4 = r3.count_ones().cast();
        total += c0 + c1 + c2 + c3;

        i += 16;
    }

    let mut sum = total.reduce_sum();
    while i < BITMAP_LENGTH {
        dest[i] &= !src[i];
        sum += dest[i].count_ones() as u64;
        i += 1;
    }

    sum
}

/// Compute rank (count of set bits up to and including index) using SIMD.
///
/// Counts all bits in words [0, key) plus bits in word[key] up to bit position.
#[inline]
pub fn rank_simd(bits: &[u64; BITMAP_LENGTH], key: usize, bit: usize) -> u64 {
    let mut total: u64x4 = Simd::splat(0);
    let mut i = 0;

    // Count complete words before key using SIMD
    while i + 16 <= key {
        let v0 = u64x4::from_slice(&bits[i..i+4]);
        let v1 = u64x4::from_slice(&bits[i+4..i+8]);
        let v2 = u64x4::from_slice(&bits[i+8..i+12]);
        let v3 = u64x4::from_slice(&bits[i+12..i+16]);

        let c0: u64x4 = v0.count_ones().cast();
        let c1: u64x4 = v1.count_ones().cast();
        let c2: u64x4 = v2.count_ones().cast();
        let c3: u64x4 = v3.count_ones().cast();

        total += c0 + c1 + c2 + c3;
        i += 16;
    }

    let mut sum = total.reduce_sum();

    // Handle remaining complete words before key
    while i < key {
        sum += bits[i].count_ones() as u64;
        i += 1;
    }

    // Add partial count from the key word
    sum + (bits[key] << (63 - bit)).count_ones() as u64
}

/// Count set bits in a slice of bytes using SIMD.
/// Used for efficient cardinality counting when loading from bytes.
#[inline]
pub fn popcount_bytes_simd(bytes: &[u8]) -> u64 {
    use core::simd::u8x32;

    let mut total = 0u64;
    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= bytes.len() {
        let v = u8x32::from_slice(&bytes[i..i+32]);
        // count_ones on bytes, then sum
        let counts: core::simd::u8x32 = v.count_ones();
        total += counts.reduce_sum() as u64;
        i += 32;
    }

    // Handle remaining bytes
    for byte in &bytes[i..] {
        total += byte.count_ones() as u64;
    }

    total
}

/// Count bits in a word range for the count_runs operation.
/// Returns the number of runs in a bitmap.
#[inline]
pub fn count_runs_simd(bits: &[u64; BITMAP_LENGTH]) -> u64 {
    let mut num_runs = 0u64;
    let mut i = 0;

    // Process in chunks of 4 words
    while i + 4 < BITMAP_LENGTH {
        let v0 = u64x4::from_slice(&bits[i..i+4]);
        let next = u64x4::from_array([bits[i+1], bits[i+2], bits[i+3], bits[i+4]]);

        // Count transitions: ((word << 1) & !word) counts 0->1 transitions
        // Add (word >> 63) & !next_word for runs that span word boundaries
        let shifted_left = v0 << Simd::splat(1u64);
        let msb = v0 >> Simd::splat(63u64);

        // Count run starts: 0->1 transitions
        let run_starts: u64x4 = (shifted_left & !v0).count_ones().cast();

        // Count runs that continue to next word
        let continues = msb & !next;

        let counts = run_starts + continues;
        num_runs += counts.reduce_sum();
        i += 4;
    }

    // Handle remaining words
    while i < BITMAP_LENGTH - 1 {
        let word = bits[i];
        let next_word = bits[i + 1];
        num_runs += ((word << 1) & !word).count_ones() as u64 + ((word >> 63) & !next_word);
        i += 1;
    }

    // Handle last word
    let last = bits[BITMAP_LENGTH - 1];
    num_runs += ((last << 1) & !last).count_ones() as u64 + (last >> 63);

    num_runs
}

/// SIMD-optimized select operation using PDEP on x86_64 BMI2
///
/// Given a word and a rank n, returns the position of the n-th set bit.
/// Uses PDEP intrinsic when available for O(1) operation instead of O(popcount) loop.
#[inline]
pub fn select_in_word(value: u64, n: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // PDEP deposits bits from source to positions marked in mask
        // pdep(1 << n, value) puts a 1 at the n-th set bit position
        // Then trailing_zeros gives us the position
        use core::arch::x86_64::_pdep_u64;
        unsafe {
            let deposited = _pdep_u64(1u64 << n, value);
            deposited.trailing_zeros() as u64
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        // Fallback: reset n of the least significant bits and count trailing zeros
        let mut v = value;
        for _ in 0..n {
            v &= v - 1;
        }
        v.trailing_zeros() as u64
    }
}

/// SIMD-optimized population count for a partial range of the bitmap.
/// Counts bits in words [start_key..end_key) plus partial bits in boundary words.
#[inline]
pub fn popcount_range_simd(
    bits: &[u64; BITMAP_LENGTH],
    start_key: usize,
    start_bit: usize,
    end_key: usize,
    end_bit: usize,
) -> u64 {
    if start_key == end_key {
        // Single word case
        let mask = if end_bit == 63 { u64::MAX } else { (1u64 << (end_bit + 1)) - 1 };
        let mask = mask & !((1u64 << start_bit) - 1);
        return (bits[start_key] & mask).count_ones() as u64;
    }

    let mut total = 0u64;

    // Count partial first word
    let start_mask = !((1u64 << start_bit) - 1);
    total += (bits[start_key] & start_mask).count_ones() as u64;

    // Count complete words in the middle using SIMD
    let middle_start = start_key + 1;
    let middle_end = end_key;

    if middle_end > middle_start {
        let mut acc: u64x4 = Simd::splat(0);
        let mut i = middle_start;

        while i + 16 <= middle_end {
            let v0 = u64x4::from_slice(&bits[i..i+4]);
            let v1 = u64x4::from_slice(&bits[i+4..i+8]);
            let v2 = u64x4::from_slice(&bits[i+8..i+12]);
            let v3 = u64x4::from_slice(&bits[i+12..i+16]);

            let c0: u64x4 = v0.count_ones().cast();
            let c1: u64x4 = v1.count_ones().cast();
            let c2: u64x4 = v2.count_ones().cast();
            let c3: u64x4 = v3.count_ones().cast();

            acc += c0 + c1 + c2 + c3;
            i += 16;
        }

        total += acc.reduce_sum();

        while i < middle_end {
            total += bits[i].count_ones() as u64;
            i += 1;
        }
    }

    // Count partial last word
    let end_mask = if end_bit == 63 { u64::MAX } else { (1u64 << (end_bit + 1)) - 1 };
    total += (bits[end_key] & end_mask).count_ones() as u64;

    total
}

/// Find the minimum set bit position using SIMD.
/// Returns None if the bitmap is empty.
#[inline]
pub fn min_simd(bits: &[u64; BITMAP_LENGTH]) -> Option<u16> {
    // Process 8 words at a time looking for non-zero
    let mut i = 0;

    while i + 8 <= BITMAP_LENGTH {
        let v = u64x8::from_slice(&bits[i..i+8]);
        let combined = v.reduce_or();
        if combined != 0 {
            // Found non-zero, scan this group
            for j in i..i+8 {
                if bits[j] != 0 {
                    return Some((j * 64 + bits[j].trailing_zeros() as usize) as u16);
                }
            }
        }
        i += 8;
    }

    // Handle remainder
    while i < BITMAP_LENGTH {
        if bits[i] != 0 {
            return Some((i * 64 + bits[i].trailing_zeros() as usize) as u16);
        }
        i += 1;
    }

    None
}

/// Find the maximum set bit position using SIMD.
/// Returns None if the bitmap is empty.
#[inline]
pub fn max_simd(bits: &[u64; BITMAP_LENGTH]) -> Option<u16> {
    // Search backwards in chunks of 8 words
    let mut i = BITMAP_LENGTH;

    while i >= 8 {
        i -= 8;
        let v = u64x8::from_slice(&bits[i..i+8]);
        let combined = v.reduce_or();
        if combined != 0 {
            // Found non-zero, scan this group backwards
            for j in (i..i+8).rev() {
                if bits[j] != 0 {
                    return Some((j * 64 + (63 - bits[j].leading_zeros() as usize)) as u16);
                }
            }
        }
    }

    // Handle remainder at the start
    for j in (0..i).rev() {
        if bits[j] != 0 {
            return Some((j * 64 + (63 - bits[j].leading_zeros() as usize)) as u16);
        }
    }

    None
}

/// Count set bits in a range of words for size_hint optimization.
#[inline]
pub fn popcount_word_range_simd(bits: &[u64; BITMAP_LENGTH], start: usize, end: usize) -> u64 {
    if start >= end {
        return 0;
    }

    let mut total: u64x4 = Simd::splat(0);
    let mut i = start;

    while i + 16 <= end {
        let v0 = u64x4::from_slice(&bits[i..i+4]);
        let v1 = u64x4::from_slice(&bits[i+4..i+8]);
        let v2 = u64x4::from_slice(&bits[i+8..i+12]);
        let v3 = u64x4::from_slice(&bits[i+12..i+16]);

        let c0: u64x4 = v0.count_ones().cast();
        let c1: u64x4 = v1.count_ones().cast();
        let c2: u64x4 = v2.count_ones().cast();
        let c3: u64x4 = v3.count_ones().cast();

        total += c0 + c1 + c2 + c3;
        i += 16;
    }

    let mut sum = total.reduce_sum();
    while i < end {
        sum += bits[i].count_ones() as u64;
        i += 1;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcount_simd_empty() {
        let bits = [0u64; BITMAP_LENGTH];
        assert_eq!(popcount_simd(&bits), 0);
    }

    #[test]
    fn test_popcount_simd_full() {
        let bits = [u64::MAX; BITMAP_LENGTH];
        assert_eq!(popcount_simd(&bits), 64 * BITMAP_LENGTH as u64);
    }

    #[test]
    fn test_popcount_simd_pattern() {
        let mut bits = [0u64; BITMAP_LENGTH];
        bits[0] = 0b11111111; // 8 bits
        bits[100] = 0b1010101010101010; // 8 bits
        bits[500] = u64::MAX; // 64 bits
        assert_eq!(popcount_simd(&bits), 8 + 8 + 64);
    }

    #[test]
    fn test_is_disjoint_simd() {
        let a = [0u64; BITMAP_LENGTH];
        let b = [0u64; BITMAP_LENGTH];
        assert!(is_disjoint_simd(&a, &b));

        let mut a = [0u64; BITMAP_LENGTH];
        let mut b = [0u64; BITMAP_LENGTH];
        a[0] = 0b1010;
        b[0] = 0b0101;
        assert!(is_disjoint_simd(&a, &b));

        a[0] = 0b1111;
        assert!(!is_disjoint_simd(&a, &b));
    }

    #[test]
    fn test_is_subset_simd() {
        let a = [0u64; BITMAP_LENGTH];
        let b = [u64::MAX; BITMAP_LENGTH];
        assert!(is_subset_simd(&a, &b));

        let mut a = [0u64; BITMAP_LENGTH];
        let mut b = [0u64; BITMAP_LENGTH];
        a[0] = 0b0101;
        b[0] = 0b1111;
        assert!(is_subset_simd(&a, &b));

        a[100] = 1;
        assert!(!is_subset_simd(&a, &b));
    }

    #[test]
    fn test_intersection_len_simd() {
        let mut a = [0u64; BITMAP_LENGTH];
        let mut b = [0u64; BITMAP_LENGTH];
        a[0] = 0b11111111;
        b[0] = 0b00001111;
        assert_eq!(intersection_len_simd(&a, &b), 4);

        a[500] = u64::MAX;
        b[500] = u64::MAX;
        assert_eq!(intersection_len_simd(&a, &b), 4 + 64);
    }

    #[test]
    fn test_bitwise_ops_simd() {
        let mut dest = [0u64; BITMAP_LENGTH];
        let mut src = [0u64; BITMAP_LENGTH];

        dest[0] = 0b1100;
        src[0] = 0b1010;

        let mut test = dest;
        let count = or_assign_simd(&mut test, &src);
        assert_eq!(test[0], 0b1110);
        assert_eq!(count, 3);

        let mut test = dest;
        let count = and_assign_simd(&mut test, &src);
        assert_eq!(test[0], 0b1000);
        assert_eq!(count, 1);

        let mut test = dest;
        let count = xor_assign_simd(&mut test, &src);
        assert_eq!(test[0], 0b0110);
        assert_eq!(count, 2);

        let mut test = dest;
        let count = sub_assign_simd(&mut test, &src);
        assert_eq!(test[0], 0b0100);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_rank_simd() {
        let mut bits = [0u64; BITMAP_LENGTH];
        bits[0] = 0b11111111; // 8 bits in positions 0-7
        bits[1] = 0b11110000; // 4 bits in positions 4-7 of word 1

        // Rank at word 0, bit 3 should count bits 0-3 = 4 bits
        assert_eq!(rank_simd(&bits, 0, 3), 4);

        // Rank at word 1, bit 7 should count all of word 0 (8) + bits in word 1 up to 7 (4)
        assert_eq!(rank_simd(&bits, 1, 7), 8 + 4);
    }

    #[test]
    fn test_select_in_word() {
        let word = 0b10110100u64; // bits at positions 2, 4, 5, 7
        assert_eq!(select_in_word(word, 0), 2); // 0th set bit at position 2
        assert_eq!(select_in_word(word, 1), 4); // 1st set bit at position 4
        assert_eq!(select_in_word(word, 2), 5); // 2nd set bit at position 5
        assert_eq!(select_in_word(word, 3), 7); // 3rd set bit at position 7
    }

    #[test]
    fn test_min_max_simd() {
        let bits = [0u64; BITMAP_LENGTH];
        assert_eq!(min_simd(&bits), None);
        assert_eq!(max_simd(&bits), None);

        let mut bits = [0u64; BITMAP_LENGTH];
        bits[100] = 0b10000000; // bit 7 of word 100
        assert_eq!(min_simd(&bits), Some(100 * 64 + 7));
        assert_eq!(max_simd(&bits), Some(100 * 64 + 7));

        bits[500] = 0b11000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000u64;
        // bits 62, 63 set in word 500
        assert_eq!(min_simd(&bits), Some(100 * 64 + 7));
        assert_eq!(max_simd(&bits), Some(500 * 64 + 63));
    }

    #[test]
    fn test_popcount_word_range() {
        let mut bits = [0u64; BITMAP_LENGTH];
        bits[10] = u64::MAX; // 64 bits
        bits[20] = u64::MAX; // 64 bits
        bits[30] = u64::MAX; // 64 bits

        assert_eq!(popcount_word_range_simd(&bits, 0, 100), 192);
        assert_eq!(popcount_word_range_simd(&bits, 15, 25), 64);
        assert_eq!(popcount_word_range_simd(&bits, 0, 0), 0);
    }
}
