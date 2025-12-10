//! Integration tests for the Proteus Semantic Folding engine.

use proteus::{
    FingerprintConfig, Retina, Sdr, Som, SomConfig, SomTrainer, Tokenizer, TrainingContext,
    WordFingerprint, WordFingerprinter,
};
use std::collections::HashMap;
use tempfile::tempdir;

/// Creates a simple test corpus for training.
fn create_test_corpus() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "A fast red fox leaps above the sleeping hound".to_string(),
        "The cat and the dog are good friends".to_string(),
        "Cats and dogs can be friendly animals".to_string(),
        "The king and queen ruled the kingdom".to_string(),
        "A queen is a female king in royal terms".to_string(),
        "The man and woman walked through the park".to_string(),
        "Men and women enjoy walking in parks".to_string(),
    ]
}

/// Creates training contexts from a corpus.
fn create_training_contexts(documents: &[String]) -> (Vec<TrainingContext>, HashMap<String, usize>) {
    let tokenizer = Tokenizer::default_config();

    // Build vocabulary
    let mut all_tokens: Vec<String> = Vec::new();
    let mut contexts: Vec<(String, Vec<String>)> = Vec::new();

    for doc in documents {
        let tokens = tokenizer.tokenize_to_strings(doc);
        all_tokens.extend(tokens.clone());

        for (center, context) in tokenizer.context_windows_as_strings(doc, 2) {
            contexts.push((center, context));
        }
    }

    let vocab: HashMap<String, usize> = all_tokens
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i))
        .collect();

    let training_contexts: Vec<TrainingContext> = contexts
        .iter()
        .map(|(center, context_words)| {
            let mut embedding = vec![0.0; vocab.len()];
            for word in context_words {
                if let Some(&idx) = vocab.get(word) {
                    embedding[idx] += 1.0;
                }
            }
            let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
            TrainingContext::new(center.clone(), embedding)
        })
        .collect();

    (training_contexts, vocab)
}

#[test]
fn test_end_to_end_training() {
    let corpus = create_test_corpus();
    let (training_contexts, vocab) = create_training_contexts(&corpus);

    // Configure a small SOM for testing
    let som_config = SomConfig {
        dimension: 16,
        weight_dimension: vocab.len(),
        iterations: 100,
        seed: Some(42),
        initial_radius: 8.0,
        final_radius: 1.0,
        ..Default::default()
    };

    let mut som = Som::new(&som_config);
    let mut trainer = SomTrainer::new(som_config);

    // Train
    let word_to_bmus = trainer.train(&mut som, &training_contexts).unwrap();

    // Verify training produced results
    assert!(!word_to_bmus.is_empty());

    // Create fingerprints
    let grid_size = 16 * 16;
    let fp_config = FingerprintConfig {
        max_active_bits: 10,
        ..Default::default()
    };
    let mut fingerprinter = WordFingerprinter::new(fp_config.clone(), grid_size as u32);
    fingerprinter.create_fingerprints(&word_to_bmus, None);

    // Verify fingerprints were created
    assert!(!fingerprinter.is_empty());

    // Create retina
    let retina = Retina::with_index(fingerprinter.into_fingerprints(), 16, fp_config);

    // Verify retina works
    assert!(retina.vocabulary_size() > 0);
}

#[test]
fn test_retina_save_load_roundtrip() {
    let mut fps = HashMap::new();

    fps.insert(
        "hello".to_string(),
        WordFingerprint::new("hello".to_string(), &[1, 2, 3, 4, 5], 256),
    );
    fps.insert(
        "world".to_string(),
        WordFingerprint::new("world".to_string(), &[3, 4, 5, 6, 7], 256),
    );
    fps.insert(
        "test".to_string(),
        WordFingerprint::new("test".to_string(), &[10, 20, 30, 40, 50], 256),
    );

    let fp_config = FingerprintConfig::default();
    let retina = Retina::with_index(fps, 16, fp_config);

    // Save
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.retina");
    retina.save(&path).unwrap();

    // Load
    let loaded = Retina::load(&path).unwrap();

    // Verify
    assert_eq!(loaded.vocabulary_size(), 3);
    assert!(loaded.contains("hello"));
    assert!(loaded.contains("world"));
    assert!(loaded.contains("test"));

    // Verify fingerprints match
    let hello_fp = loaded.get_word_fingerprint("hello").unwrap();
    assert!(hello_fp.contains(1));
    assert!(hello_fp.contains(5));
}

#[test]
fn test_text_fingerprinting() {
    let mut fps = HashMap::new();

    fps.insert(
        "the".to_string(),
        WordFingerprint::new("the".to_string(), &[1, 2, 3], 256),
    );
    fps.insert(
        "quick".to_string(),
        WordFingerprint::new("quick".to_string(), &[10, 11, 12], 256),
    );
    fps.insert(
        "brown".to_string(),
        WordFingerprint::new("brown".to_string(), &[20, 21, 22], 256),
    );
    fps.insert(
        "fox".to_string(),
        WordFingerprint::new("fox".to_string(), &[30, 31, 32], 256),
    );

    let fp_config = FingerprintConfig {
        max_active_bits: 20,
        ..Default::default()
    };
    let retina = Retina::with_index(fps, 16, fp_config);

    // Fingerprint text
    let fp = retina.fingerprint_text("the quick brown fox");

    // Should contain positions from all matching words
    assert!(fp.contains(1)); // from "the"
    assert!(fp.contains(10)); // from "quick"
    assert!(fp.contains(20)); // from "brown"
    assert!(fp.contains(30)); // from "fox"
}

#[test]
fn test_text_similarity() {
    let mut fps = HashMap::new();

    // Similar words share positions
    fps.insert(
        "cat".to_string(),
        WordFingerprint::new("cat".to_string(), &[1, 2, 3, 10, 11], 256),
    );
    fps.insert(
        "dog".to_string(),
        WordFingerprint::new("dog".to_string(), &[1, 2, 4, 10, 12], 256),
    );
    fps.insert(
        "fish".to_string(),
        WordFingerprint::new("fish".to_string(), &[1, 5, 6, 50, 51], 256),
    );
    fps.insert(
        "bird".to_string(),
        WordFingerprint::new("bird".to_string(), &[1, 5, 7, 50, 52], 256),
    );

    let fp_config = FingerprintConfig::default();
    let retina = Retina::with_index(fps, 16, fp_config);

    // Similar texts should have higher similarity
    let sim_cat_dog = retina.text_similarity("cat", "dog");
    let sim_cat_fish = retina.text_similarity("cat", "fish");

    // Cat and dog share more positions than cat and fish
    assert!(sim_cat_dog > sim_cat_fish);
}

#[test]
fn test_find_similar_words() {
    let mut fps = HashMap::new();

    // Create a cluster of related words
    fps.insert(
        "king".to_string(),
        WordFingerprint::new("king".to_string(), &[1, 2, 3, 100, 101], 256),
    );
    fps.insert(
        "queen".to_string(),
        WordFingerprint::new("queen".to_string(), &[1, 2, 4, 100, 102], 256),
    );
    fps.insert(
        "prince".to_string(),
        WordFingerprint::new("prince".to_string(), &[1, 2, 5, 100, 103], 256),
    );
    fps.insert(
        "car".to_string(),
        WordFingerprint::new("car".to_string(), &[50, 51, 52, 200, 201], 256),
    );

    let fp_config = FingerprintConfig::default();
    let retina = Retina::with_index(fps, 16, fp_config);

    let similar = retina.find_similar_words("king", 3).unwrap();

    // Queen and prince should be more similar to king than car
    assert!(similar.iter().any(|(w, _)| w == "queen"));
    assert!(similar.iter().any(|(w, _)| w == "prince"));

    // Car should have low similarity
    let car_sim = retina.word_similarity("king", "car").unwrap();
    let queen_sim = retina.word_similarity("king", "queen").unwrap();
    assert!(queen_sim > car_sim);
}

#[test]
fn test_sdr_operations() {
    let a = Sdr::from_positions(&[1, 2, 3, 4, 5], 100);
    let b = Sdr::from_positions(&[4, 5, 6, 7, 8], 100);

    // Test overlap
    let overlap = a.overlap(&b);
    assert_eq!(overlap.cardinality(), 2);
    assert!(overlap.contains(4));
    assert!(overlap.contains(5));

    // Test union
    let union = a.union(&b);
    assert_eq!(union.cardinality(), 8);

    // Test XOR
    let xor = a.xor(&b);
    assert_eq!(xor.cardinality(), 6); // Elements in A or B but not both

    // Test similarities
    let jaccard = a.jaccard_similarity(&b);
    assert!((jaccard - 2.0 / 8.0).abs() < 1e-10);

    let cosine = a.cosine_similarity(&b);
    assert!((cosine - 2.0 / 5.0).abs() < 1e-10);
}

#[test]
fn test_inverted_index_search() {
    let mut fps = HashMap::new();

    fps.insert(
        "apple".to_string(),
        WordFingerprint::new("apple".to_string(), &[1, 2, 3, 4, 5], 256),
    );
    fps.insert(
        "banana".to_string(),
        WordFingerprint::new("banana".to_string(), &[2, 3, 4, 5, 6], 256),
    );
    fps.insert(
        "cherry".to_string(),
        WordFingerprint::new("cherry".to_string(), &[3, 4, 5, 6, 7], 256),
    );
    fps.insert(
        "zebra".to_string(),
        WordFingerprint::new("zebra".to_string(), &[100, 101, 102, 103, 104], 256),
    );

    let fp_config = FingerprintConfig::default();
    let retina = Retina::with_index(fps, 16, fp_config);

    // Search with a query similar to apple/banana
    let query = Sdr::from_positions(&[2, 3, 4, 5], 256);
    let results = retina.find_similar_indexed(&query, 3).unwrap();

    // Should find apple, banana, cherry (not zebra)
    let words: Vec<&String> = results.iter().map(|(w, _)| w).collect();
    assert!(words.contains(&&"apple".to_string()));
    assert!(words.contains(&&"banana".to_string()));
    assert!(!words.contains(&&"zebra".to_string()));
}

#[test]
fn test_semantic_operations() {
    let mut fps = HashMap::new();

    fps.insert(
        "king".to_string(),
        WordFingerprint::new("king".to_string(), &[1, 2, 3, 100, 101, 102], 256),
    );
    fps.insert(
        "queen".to_string(),
        WordFingerprint::new("queen".to_string(), &[1, 2, 4, 100, 101, 103], 256),
    );
    fps.insert(
        "man".to_string(),
        WordFingerprint::new("man".to_string(), &[1, 3, 5, 200, 201, 102], 256),
    );
    fps.insert(
        "woman".to_string(),
        WordFingerprint::new("woman".to_string(), &[1, 4, 5, 200, 201, 103], 256),
    );

    let fp_config = FingerprintConfig::default();
    let retina = Retina::with_index(fps, 16, fp_config);

    // Test semantic center
    let center = retina.semantic_center(&["king", "queen"]);
    // Should contain positions common to both
    assert!(center.contains(1));
    assert!(center.contains(100));
    assert!(center.contains(101));

    // Test semantic difference
    let diff = retina.semantic_difference("king", "queen").unwrap();
    // Should contain positions unique to king
    assert!(diff.contains(3));
    assert!(diff.contains(102));
    assert!(!diff.contains(4));
    assert!(!diff.contains(103));
}

#[test]
fn test_tokenizer_unicode() {
    let tokenizer = Tokenizer::default_config();

    // Test basic Unicode handling
    let text = "Hello world Привет мир";
    let tokens = tokenizer.tokenize_to_strings(text);

    assert!(tokens.contains(&"hello".to_string()));
    assert!(tokens.contains(&"world".to_string()));
    assert!(tokens.contains(&"привет".to_string()));
    assert!(tokens.contains(&"мир".to_string()));
}

#[test]
fn test_empty_and_edge_cases() {
    let fp_config = FingerprintConfig::default();
    let retina = Retina::new(HashMap::new(), 16, fp_config);

    // Empty retina should handle operations gracefully
    assert_eq!(retina.vocabulary_size(), 0);

    let fp = retina.fingerprint_text("hello world");
    assert!(fp.is_empty());

    // Empty SDR operations
    let empty = Sdr::new(100);
    let non_empty = Sdr::from_positions(&[1, 2, 3], 100);

    assert_eq!(empty.cosine_similarity(&non_empty), 0.0);
    assert_eq!(empty.jaccard_similarity(&non_empty), 0.0);
}

#[test]
fn test_som_convergence() {
    // Test that SOM training produces consistent results with a seed
    let config = SomConfig {
        dimension: 8,
        weight_dimension: 10,
        iterations: 50,
        seed: Some(12345),
        initial_radius: 4.0,
        final_radius: 1.0,
        ..Default::default()
    };

    let contexts = vec![
        TrainingContext::new("a".to_string(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        TrainingContext::new("b".to_string(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        TrainingContext::new("c".to_string(), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];

    // Train twice with same seed
    let mut som1 = Som::new(&config);
    let mut trainer1 = SomTrainer::new(config.clone());
    let result1 = trainer1.train(&mut som1, &contexts).unwrap();

    let mut som2 = Som::new(&config);
    let mut trainer2 = SomTrainer::new(config);
    let result2 = trainer2.train(&mut som2, &contexts).unwrap();

    // Results should be identical
    assert_eq!(result1.len(), result2.len());
    for (word, bmus) in &result1 {
        assert_eq!(bmus, result2.get(word).unwrap());
    }
}

#[test]
fn test_sparsification() {
    let mut sdr = Sdr::from_positions(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100);

    // Sparsify to 5 bits
    sdr.sparsify(5);
    assert_eq!(sdr.cardinality(), 5);

    // Sparsifying to more than current should be a no-op
    sdr.sparsify(10);
    assert_eq!(sdr.cardinality(), 5);
}

#[test]
fn test_dense_representation() {
    let sdr = Sdr::from_positions(&[0, 5, 9], 10);
    let dense = sdr.to_dense();

    assert_eq!(dense.len(), 10);
    assert_eq!(dense[0], 1);
    assert_eq!(dense[1], 0);
    assert_eq!(dense[5], 1);
    assert_eq!(dense[9], 1);

    // Round-trip
    let recovered = Sdr::from_dense(&dense);
    assert_eq!(sdr, recovered);
}
