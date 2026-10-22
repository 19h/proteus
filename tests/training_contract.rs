//! Numerical, retrieval and persistence oracles for the public training pipeline.
use proteus::training::{read_snippets, train_corpus, DocumentMode, TrainingConfig};
use proteus::{
    FingerprintConfig, Retina, RetinaFormat, Sdr, Som, SomConfig, SomTrainer, TrainingContext,
    WordFingerprint,
};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::io::Cursor;

fn config() -> SomConfig {
    SomConfig {
        dimension: 3,
        weight_dimension: 2,
        iterations: 7,
        initial_learning_rate: 0.5,
        final_learning_rate: 0.1,
        initial_radius: 4.0,
        final_radius: 0.5,
        seed: Some(17),
        ..SomConfig::default()
    }
}

#[test]
fn online_training_matches_independent_scalar_equations() {
    // Deliberately use a radius larger than the torus: every cell is updated once.
    // The oracle does not call Som::update, neighbourhood(), or SIMD helpers.
    for toroidal in [false, true] {
        let c = SomConfig {
            toroidal,
            ..config()
        };
        let mut actual = Som::new(&c);
        let mut expected: Vec<Vec<f64>> =
            actual.neurons.iter().map(|n| n.weights.clone()).collect();
        // The production representation is f32; align only the initial conditions.
        for row in &mut expected {
            for w in row {
                *w = (*w as f32) as f64;
            }
        }
        let inputs = vec![vec![0.0f32, 1.0], vec![1.0, 0.0], vec![0.7, 0.7]];
        let mut order = [0, 1, 2];
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        for t in 0..c.iterations {
            if t % order.len() == 0 {
                order.shuffle(&mut rng);
            }
            let input = &inputs[order[t % order.len()]];
            let bmu = (0..9)
                .min_by(|&a, &b| {
                    let distance = |i: usize| {
                        (0..2)
                            .map(|j| (expected[i][j] - input[j] as f64).powi(2))
                            .sum::<f64>()
                    };
                    distance(a).total_cmp(&distance(b)).then(a.cmp(&b))
                })
                .unwrap();
            let fraction = t as f64 / (c.iterations - 1) as f64;
            let lr = c.initial_learning_rate
                * (c.final_learning_rate / c.initial_learning_rate).powf(fraction);
            let sigma = c.initial_radius * (c.final_radius / c.initial_radius).powf(fraction);
            for (i, row) in expected.iter_mut().enumerate() {
                let mut dr = (i / 3).abs_diff(bmu / 3);
                let mut dc = (i % 3).abs_diff(bmu % 3);
                if toroidal {
                    dr = dr.min(3 - dr);
                    dc = dc.min(3 - dc);
                }
                let h = (-((dr * dr + dc * dc) as f64) / (2.0 * sigma * sigma)).exp();
                for j in 0..2 {
                    row[j] += lr * h * (input[j] as f64 - row[j]);
                }
            }
        }
        let assignments = SomTrainer::new(c)
            .train_vectors(&mut actual, &inputs, |_, _| {})
            .unwrap();
        for (row, neuron) in expected.iter().zip(&actual.neurons) {
            for (&a, &b) in row.iter().zip(&neuron.weights) {
                assert!((a - b).abs() < 3e-7, "{a} != {b}");
            }
        }
        // Assignments must be recomputed after the last update, including unvisited inputs.
        for (input, &bmu) in inputs.iter().zip(&assignments) {
            assert_eq!(
                bmu,
                actual
                    .find_bmu(&input.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .unwrap()
            );
        }
    }
}

#[test]
fn endpoint_schedules_single_update_and_invalid_vectors() {
    let mut c = config();
    let trainer = SomTrainer::new(c.clone());
    assert!((trainer.learning_rate(0) - c.initial_learning_rate).abs() < 1e-15);
    assert!((trainer.learning_rate(6) - c.final_learning_rate).abs() < 1e-15);
    assert!((trainer.radius(6) - c.final_radius).abs() < 1e-15);
    c.iterations = 1;
    let mut som = Som::new(&c);
    let mut trainer = SomTrainer::new(c);
    let samples = [
        TrainingContext::new("first".into(), vec![1.0, 0.0]),
        TrainingContext::new("second".into(), vec![0.0, 1.0]),
    ];
    let mapping = trainer.train(&mut som, &samples).unwrap();
    assert_eq!(mapping.values().map(Vec::len).sum::<usize>(), 2);
    let before = som.neurons[0].weights.clone();
    assert!(trainer
        .train_vectors(&mut som, &[vec![f32::NAN, 0.0]], |_, _| {})
        .is_err());
    assert!(trainer
        .train_vectors(&mut som, &[vec![1.0]], |_, _| {})
        .is_err());
    assert_eq!(before, som.neurons[0].weights);
}

#[test]
fn thread_count_does_not_change_training() {
    let inputs: Vec<_> = (0..5)
        .map(|i| (0..256).map(|j| ((i + j) % 7) as f32 / 7.0).collect())
        .collect();
    let c = SomConfig {
        dimension: 32,
        weight_dimension: 256,
        iterations: 3,
        num_threads: 1,
        ..config()
    };
    let mut a = Som::new(&c);
    let mut b = a.clone();
    let aa = SomTrainer::new(c.clone())
        .train_vectors(&mut a, &inputs, |_, _| {})
        .unwrap();
    let bb = SomTrainer::new(SomConfig {
        num_threads: 2,
        ..c
    })
    .train_vectors(&mut b, &inputs, |_, _| {})
    .unwrap();
    assert_eq!(aa, bb);
    for (x, y) in a.neurons.iter().zip(b.neurons) {
        assert_eq!(x.weights, y.weights);
    }
}

fn corpus_config() -> TrainingConfig {
    TrainingConfig {
        som: SomConfig {
            dimension: 6,
            iterations: 1200,
            initial_radius: 3.0,
            final_radius: 0.3,
            seed: Some(42),
            ..config()
        },
        fingerprint: FingerprintConfig {
            sparsity: 0.25,
            max_active_bits: 9,
            min_active_bits: 0,
            weighted_aggregation: true,
        },
        document_mode: DocumentMode::Lines,
        snippet_tokens: 64,
        min_count: 1,
        max_document_frequency: 0.8,
        ..TrainingConfig::default()
    }
}

fn topic_corpus() -> String {
    let mut lines = Vec::new();
    for term in ["king", "queen", "prince", "monarch"] {
        for context in [
            "royal crown palace throne kingdom reign",
            "royal ruler crown palace kingdom court",
            "throne reign royal court crown ruler",
        ] {
            lines.push(format!("{term} {context}"));
        }
    }
    for term in ["cat", "dog", "horse", "animal"] {
        for context in [
            "fur paws tail mammal pet animal",
            "pet mammal fur animal food paws",
            "animal tail paws fur mammal creature",
        ] {
            lines.push(format!("{term} {context}"));
        }
    }
    for term in ["ship", "boat", "vessel", "sailor"] {
        for context in [
            "ocean sail harbor sea anchor voyage",
            "sea ocean harbor anchor deck voyage",
            "sail deck voyage harbor sea ocean",
        ] {
            lines.push(format!("{term} {context}"));
        }
    }
    lines.join("\n")
}

#[test]
fn controlled_topics_retain_semantic_separation_across_seeds() {
    for seed in [1, 42, 123] {
        let mut config = corpus_config();
        config.som.seed = Some(seed);
        let (retina, report) =
            train_corpus(Cursor::new(topic_corpus()), config, |_, _| {}).unwrap();
        assert_eq!(report.mapped_snippets, 36);
        assert!(
            retina.word_similarity("king", "queen").unwrap()
                > retina.word_similarity("king", "boat").unwrap()
        );
        assert!(
            retina.word_similarity("cat", "dog").unwrap()
                > retina.word_similarity("cat", "queen").unwrap()
        );
        let top = retina.find_similar_words("king", 5).unwrap();
        assert!(
            top.iter()
                .all(|(w, _)| !["cat", "dog", "boat", "ocean", "animal"].contains(&w.as_str())),
            "seed={seed}, {top:?}"
        );
        assert!(report.quantization_error.is_finite());
    }
}

#[test]
fn seeded_models_and_reloads_are_byte_identical_and_queries_stable() {
    for projection in [0, 16] {
        let mut c = corpus_config();
        c.projection_dim = projection;
        let (a, _) = train_corpus(Cursor::new(topic_corpus()), c.clone(), |_, _| {}).unwrap();
        let (b, _) = train_corpus(Cursor::new(topic_corpus()), c, |_, _| {}).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let ap = dir.path().join("a.retina");
        let bp = dir.path().join("b.retina");
        a.save(&ap).unwrap();
        b.save(&bp).unwrap();
        assert_eq!(std::fs::read(&ap).unwrap(), std::fs::read(&bp).unwrap());
        let expected = a.find_similar_words("king", 12).unwrap();
        for _ in 0..10 {
            let loaded = Retina::load(&ap).unwrap();
            assert_eq!(expected, loaded.find_similar_words("KING", 12).unwrap());
            assert_eq!(
                a.fingerprint_text("King queen"),
                loaded.fingerprint_text("King queen")
            );
            assert_eq!(
                a.word_fingerprinter().get("king").unwrap().frequency,
                loaded.word_fingerprinter().get("king").unwrap().frequency
            );
            assert_eq!(
                loaded.training_report().unwrap().config.projection_dim,
                projection
            );
        }
    }
}

#[test]
fn wrapping_boundaries_and_read_errors_are_explicit() {
    let c = TrainingConfig {
        snippet_tokens: 4,
        min_count: 1,
        ..TrainingConfig::default()
    };
    let snippets = read_snippets(
        Cursor::new("alpha beta\ngamma delta epsilon\n\nzeta eta"),
        &c,
    )
    .unwrap();
    assert_eq!(
        snippets,
        vec![
            vec!["alpha", "beta", "gamma", "delta"],
            vec!["epsilon"],
            vec!["zeta", "eta"]
        ]
    );
    let c = TrainingConfig {
        document_mode: DocumentMode::Lines,
        ..c
    };
    assert_eq!(
        read_snippets(Cursor::new("alpha beta\ngamma delta"), &c)
            .unwrap()
            .len(),
        2
    );
    assert!(read_snippets(Cursor::new([0xff]), &c).is_err());
    assert!(train_corpus(Cursor::new(""), c, |_, _| {}).is_err());
}

#[test]
fn folding_uses_snippet_incidence_and_preserves_token_frequencies() {
    let c = TrainingConfig {
        som: SomConfig {
            dimension: 2,
            iterations: 3,
            initial_radius: 1.0,
            ..config()
        },
        min_count: 1,
        max_document_frequency: 1.0,
        fingerprint: FingerprintConfig {
            sparsity: 1.0,
            max_active_bits: 4,
            min_active_bits: 0,
            weighted_aggregation: true,
        },
        ..TrainingConfig::default()
    };
    let (retina, _) = train_corpus(Cursor::new("alpha alpha beta"), c, |_, _| {}).unwrap();
    let alpha = retina.word_fingerprinter().get("alpha").unwrap();
    let beta = retina.word_fingerprinter().get("beta").unwrap();
    assert_eq!(alpha.fingerprint, beta.fingerprint);
    assert_eq!(alpha.cardinality(), 1); // one context, no synthetic padding
    assert_eq!(alpha.frequency, 2.0 / 3.0);
    assert_eq!(beta.frequency, 1.0 / 3.0);
}

#[test]
fn indexed_cosine_matches_exhaustive_with_variable_cardinalities_and_ties() {
    let mut fps = HashMap::new();
    fps.insert(
        "perfect".into(),
        WordFingerprint::new("perfect".into(), &[0, 1], 100),
    );
    fps.insert(
        "zero".into(),
        WordFingerprint::new("zero".into(), &[99], 100),
    );
    fps.insert(
        "empty".into(),
        WordFingerprint::new("empty".into(), &[], 100),
    );
    for i in 0..10 {
        let word = format!("distractor{i}");
        let positions: Vec<u32> = (0..50).collect();
        fps.insert(word.clone(), WordFingerprint::new(word, &positions, 100));
    }
    let retina = Retina::with_index(fps, 10, FingerprintConfig::default());
    for q in [Sdr::from_positions(&[0, 1, 2], 100), Sdr::new(100)] {
        for k in [0, 1, 3, 20, usize::MAX] {
            assert_eq!(
                retina.find_similar_to_fingerprint(&q, k),
                retina.find_similar_indexed(&q, k).unwrap()
            );
        }
    }
    assert_eq!(
        retina
            .find_similar_indexed(&Sdr::from_positions(&[0, 1, 2], 100), 1)
            .unwrap()[0]
            .0,
        "perfect"
    );
}

#[test]
fn malformed_files_return_errors_in_both_readers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.retina");
    for bytes in [vec![], vec![0; 63], vec![0; 64]] {
        std::fs::write(&path, bytes).unwrap();
        assert!(Retina::load(&path).is_err());
        assert!(RetinaFormat::mmap(&path).is_err());
    }
    let (retina, _) =
        train_corpus(Cursor::new(topic_corpus()), corpus_config(), |_, _| {}).unwrap();
    retina.save(&path).unwrap();
    let valid = std::fs::read(&path).unwrap();
    for (range, bytes) in [
        (4..6, vec![255; 2]),
        (8..12, vec![255; 4]),
        (16..24, vec![255; 8]),
        (24..32, vec![0; 8]),
        (32..40, vec![255; 8]),
    ] {
        let mut bad = valid.clone();
        bad[range].copy_from_slice(&bytes);
        std::fs::write(&path, bad).unwrap();
        assert!(Retina::load(&path).is_err());
        assert!(RetinaFormat::mmap(&path).is_err());
    }
    for cut in [64, valid.len() / 2, valid.len() - 1] {
        std::fs::write(&path, &valid[..cut]).unwrap();
        assert!(Retina::load(&path).is_err());
        assert!(RetinaFormat::mmap(&path).is_err());
    }
}

#[test]
fn legacy_v1_remains_readable_without_inventing_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.retina");
    let mut bitmap = Vec::new();
    Sdr::from_positions(&[1, 3], 100)
        .bitmap()
        .serialize_into(&mut bitmap)
        .unwrap();
    let mut header = proteus::RetinaHeader::new(10, 1);
    header.version = 1;
    header.index_offset = 64;
    header.fingerprint_offset = 76;
    header.string_pool_offset = 76 + bitmap.len() as u64;
    let mut bytes = header.to_bytes();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&(bitmap.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&bitmap);
    bytes.extend_from_slice(b"king");
    std::fs::write(&path, bytes).unwrap();
    let retina = Retina::load(&path).unwrap();
    assert!(retina.training_report().is_none());
    assert_eq!(
        retina.get_word_fingerprint("king").unwrap().to_positions(),
        vec![1, 3]
    );
    assert_eq!(
        RetinaFormat::mmap(&path)
            .unwrap()
            .get("king")
            .unwrap()
            .to_positions(),
        vec![1, 3]
    );
}

#[test]
fn signed_analogy_removes_source_only_features() {
    let mut fps = HashMap::new();
    for (word, positions) in [
        ("aa", vec![1]),
        ("bb", vec![2]),
        ("cc", vec![3]),
        ("right", vec![2, 3]),
        ("wrong", vec![1, 2, 3]),
    ] {
        fps.insert(
            word.into(),
            WordFingerprint::new(word.into(), &positions, 100),
        );
    }
    let retina = Retina::with_index(fps, 10, FingerprintConfig::default());
    let results = retina.analogy("AA", "BB", "CC", usize::MAX).unwrap();
    assert_eq!(results[0].0, "right");
    assert!((results[0].1 - 2.0 / 6.0f64.sqrt()).abs() < 1e-15);
}

#[test]
fn custom_normalization_survives_reload_and_invalid_metadata_cannot_replace_file() {
    let mut config = corpus_config();
    config.text.lowercase = false;
    let (mut retina, report) = train_corpus(
        Cursor::new(topic_corpus().replace("king", "KING")),
        config,
        |_, _| {},
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("case.retina");
    retina.save(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    let loaded = Retina::load(&path).unwrap();
    assert!(loaded.contains("KING"));
    assert!(!loaded.contains("king"));
    assert_eq!(
        retina.fingerprint_text("KING queen"),
        loaded.fingerprint_text("KING queen")
    );
    let mut invalid = report;
    invalid.vocabulary += 1;
    retina.set_training_report(invalid);
    assert!(retina.save(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
}
