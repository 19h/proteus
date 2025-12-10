//! Utility functions for wtpsplit

use crate::wtpsplit::constants::PRIMES;

/// Sigmoid activation function
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid to a slice of values in place
pub fn sigmoid_inplace(values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = sigmoid(*v);
    }
}

/// Encode text as Unicode code points (ordinals)
pub fn encode_text(text: &str) -> Vec<i64> {
    text.chars().map(|c| c as i64).collect()
}

/// Hash encode character ordinals for WtP models
///
/// This implements the same hashing scheme as CANINE, using multiple hash
/// functions with prime multipliers.
///
/// # Arguments
/// * `ordinals` - Character ordinals (Unicode code points)
/// * `num_hashes` - Number of hash functions to use
/// * `num_buckets` - Number of hash buckets
///
/// # Returns
/// A 2D array of shape (len, num_hashes) containing hash IDs
pub fn hash_encode(ordinals: &[i64], num_hashes: usize, num_buckets: i64) -> Vec<Vec<i64>> {
    assert!(
        num_hashes <= PRIMES.len(),
        "num_hashes must be <= {}",
        PRIMES.len()
    );

    ordinals
        .iter()
        .map(|&ord| {
            (0..num_hashes)
                .map(|i| {
                    let shard_id = (ord + 1) * PRIMES[i];
                    shard_id.rem_euclid(num_buckets)
                })
                .collect()
        })
        .collect()
}

/// Convert split indices to sentences
///
/// Given text and indices where sentences end, reconstruct the sentences.
///
/// # Arguments
/// * `text` - The original text
/// * `indices` - Character indices where sentences end
/// * `strip_whitespace` - Whether to strip whitespace from sentences
///
/// # Returns
/// Vector of sentence strings
pub fn indices_to_sentences(text: &str, indices: &[usize], strip_whitespace: bool) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut offset = 0;

    for &idx in indices {
        let mut end_idx = idx + 1;

        // Skip trailing whitespace for next sentence start
        while end_idx < chars.len() && chars[end_idx].is_whitespace() {
            end_idx += 1;
        }

        let sentence: String = chars[offset..end_idx].iter().collect();
        let sentence = if strip_whitespace {
            sentence.trim().to_string()
        } else {
            sentence
        };

        if !sentence.is_empty() {
            sentences.push(sentence);
        }

        offset = end_idx;
    }

    // Handle the last sentence (after the last split point)
    if offset < chars.len() {
        let last_sentence: String = chars[offset..].iter().collect();
        let last_sentence = if strip_whitespace {
            last_sentence.trim().to_string()
        } else {
            last_sentence
        };

        if !last_sentence.is_empty() {
            sentences.push(last_sentence);
        }
    }

    // Handle case when indices is empty - return the whole text
    if indices.is_empty() && !text.is_empty() {
        let sentence = if strip_whitespace {
            text.trim().to_string()
        } else {
            text.to_string()
        };
        if !sentence.is_empty() {
            return vec![sentence];
        }
    }

    sentences
}

/// Map token-level probabilities to character-level probabilities
///
/// For subword models, we need to map predictions from tokens back to characters.
/// Each token's probability is assigned to the last character of that token.
///
/// # Arguments
/// * `text_len` - Length of the original text in characters
/// * `token_logits` - Logits for each token (shape: num_tokens x num_labels)
/// * `offsets` - Character offset mapping for each token (start, end)
/// * `num_labels` - Number of output labels
///
/// # Returns
/// Character-level logits (shape: text_len x num_labels)
pub fn token_to_char_probs(
    text_len: usize,
    token_logits: &[Vec<f32>],
    offsets: &[(usize, usize)],
    num_labels: usize,
) -> Vec<Vec<f32>> {
    // Initialize with -inf (very low probability)
    let mut char_probs = vec![vec![f32::NEG_INFINITY; num_labels]; text_len];

    for (token_idx, &(start, end)) in offsets.iter().enumerate() {
        if start < end && end > 0 && end <= text_len && token_idx < token_logits.len() {
            // Assign token's probability to the last character of the token
            let char_idx = end - 1;
            char_probs[char_idx] = token_logits[token_idx].clone();
        }
    }

    char_probs
}

/// Remove spaces from text and track their positions
///
/// This is used for the `remove_whitespace_before_inference` option.
///
/// # Returns
/// Tuple of (text without spaces, original positions of spaces)
pub fn remove_spaces(text: &str) -> (String, Vec<usize>) {
    let mut result = String::new();
    let mut space_positions = Vec::new();

    for c in text.chars() {
        if c == ' ' {
            space_positions.push(result.len() + space_positions.len());
        } else {
            result.push(c);
        }
    }

    (result, space_positions)
}

/// Reinsert space probabilities into the probability array
pub fn reinsert_space_probs(probs: &[f32], space_positions: &[usize]) -> Vec<f32> {
    let mut result = probs.to_vec();

    for &pos in space_positions {
        if pos <= result.len() {
            result.insert(pos, 0.0);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_encode_text() {
        let encoded = encode_text("abc");
        assert_eq!(encoded, vec![97, 98, 99]);
    }

    #[test]
    fn test_hash_encode() {
        let ordinals = vec![97, 98, 99]; // "abc"
        let hashed = hash_encode(&ordinals, 8, 8192);
        assert_eq!(hashed.len(), 3);
        assert_eq!(hashed[0].len(), 8);
        // All hash values should be in range [0, num_buckets)
        for row in &hashed {
            for &h in row {
                assert!(h >= 0 && h < 8192);
            }
        }
    }

    #[test]
    fn test_indices_to_sentences() {
        let text = "Hello world. This is a test.";
        let indices = vec![11]; // After "Hello world."
        let sentences = indices_to_sentences(text, &indices, false);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Hello world. ");
        assert_eq!(sentences[1], "This is a test.");
    }

    #[test]
    fn test_indices_to_sentences_strip() {
        let text = "Hello world.  This is a test.";
        let indices = vec![11]; // After "Hello world."
        let sentences = indices_to_sentences(text, &indices, true);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "This is a test.");
    }

    #[test]
    fn test_remove_spaces() {
        let (text, positions) = remove_spaces("hello world test");
        assert_eq!(text, "helloworldtest");
        assert_eq!(positions, vec![5, 11]);
    }
}
