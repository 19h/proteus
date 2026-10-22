use proteus::evaluation::{evaluate_pairs, WordPair};
use proteus::training::{
    train_corpus, DocumentMode, FeatureModel, FoldingConfig, ProjectionMethod, TrainingConfig,
};
use proteus::{FingerprintConfig, Retina, SomConfig, WordFingerprint};
use std::collections::HashMap;
use std::io::Cursor;

fn training_config(feature_model: FeatureModel) -> TrainingConfig {
    TrainingConfig {
        som: SomConfig {
            dimension: 8,
            iterations: 1200,
            initial_radius: 4.0,
            final_radius: 0.5,
            seed: Some(7),
            ..SomConfig::default()
        },
        fingerprint: FingerprintConfig {
            sparsity: 0.125,
            max_active_bits: 8,
            min_active_bits: 0,
            weighted_aggregation: true,
        },
        min_count: 1,
        max_document_frequency: 1.0,
        projection_dim: 16,
        feature_model,
        document_mode: DocumentMode::Lines,
        ..TrainingConfig::default()
    }
}

const CORPUS: &str = "king wears royal crown\nqueen wears royal crown\nking rules palace court\nqueen rules palace court\ncat has fur paws\ndog has fur paws\ncat runs with tail\ndog runs with tail\n";

#[test]
fn quality_modes_remain_deterministic_and_survive_both_readers() {
    let feature_settings = [
        (FeatureModel::Tfidf, ProjectionMethod::Random),
        (FeatureModel::ContextPpmi, ProjectionMethod::Random),
        (FeatureModel::ContextPpmi, ProjectionMethod::Spectral),
    ];
    for (model, projection_method) in feature_settings {
        for projected in [0, 16] {
            if projected == 0 && projection_method == ProjectionMethod::Spectral {
                continue;
            }
            let mut config = training_config(model);
            config.projection_dim = projected;
            config.projection_method = projection_method;
            let (first_retina, report) =
                train_corpus(Cursor::new(CORPUS), config.clone(), |_, _| {}).unwrap();
            let (second_retina, _) = train_corpus(Cursor::new(CORPUS), config, |_, _| {}).unwrap();
            assert!(report.quality.as_ref().unwrap().mean_active_bits > 0.0);
            let directory = tempfile::tempdir().unwrap();
            let first_path = directory.path().join("a.retina");
            let second_path = directory.path().join("b.retina");
            first_retina.save(&first_path).unwrap();
            second_retina.save(&second_path).unwrap();
            assert_eq!(
                std::fs::read(&first_path).unwrap(),
                std::fs::read(second_path).unwrap()
            );
            let loaded = Retina::load(&first_path).unwrap();
            let mapped = proteus::MmappedRetina::open(&first_path).unwrap();
            assert_eq!(
                first_retina.find_similar_words("KING", 12).unwrap(),
                loaded.find_similar_words("KING", 12).unwrap()
            );
            assert_eq!(
                first_retina.find_similar_words("KING", 12).unwrap(),
                mapped.find_similar("KING", 12).unwrap()
            );
            let report = loaded.training_report().unwrap();
            assert_eq!(report.config.feature_model, model);
            assert_eq!(report.config.projection_method, projection_method);
            assert_eq!(report.config.folding, FoldingConfig::default());
        }
    }
}

#[test]
fn missing_legacy_fields_do_not_opt_old_models_into_new_training() {
    let mut value = serde_json::to_value(TrainingConfig::default()).unwrap();
    for key in [
        "feature_model",
        "folding",
        "context_window",
        "context_alpha",
        "projection_method",
        "projection_iterations",
    ] {
        value.as_object_mut().unwrap().remove(key);
    }
    let old: TrainingConfig = serde_json::from_value(value).unwrap();
    assert_eq!(old.folding, FoldingConfig::incidence());
    assert_eq!(old.feature_model, FeatureModel::Tfidf);
    old.validate().unwrap();
}

#[test]
fn invalid_quality_parameters_are_rejected_before_training() {
    let mut cases = Vec::new();
    for sigma in [-1.0, f64::INFINITY, f64::NAN] {
        let mut config = training_config(FeatureModel::Tfidf);
        config.folding.sigma = sigma;
        cases.push(config);
    }
    for exponent in [-0.1, 1.1, f64::NAN] {
        let mut config = training_config(FeatureModel::Tfidf);
        config.folding.background_exponent = exponent;
        cases.push(config);
    }
    for alpha in [0.0, -1.0, 1.1, f64::NAN] {
        let mut config = training_config(FeatureModel::ContextPpmi);
        config.context_alpha = alpha;
        cases.push(config);
    }
    let mut invalid_spectral = training_config(FeatureModel::Tfidf);
    invalid_spectral.projection_method = ProjectionMethod::Spectral;
    cases.push(invalid_spectral);
    let mut config = training_config(FeatureModel::ContextPpmi);
    config.context_window = 0;
    cases.push(config);
    for config in cases {
        assert!(train_corpus(Cursor::new(CORPUS), config, |_, _| {}).is_err());
    }
    let mut config = training_config(FeatureModel::ContextPpmi);
    config.context_window = usize::MAX;
    assert!(train_corpus(Cursor::new(CORPUS), config, |_, _| {}).is_ok());
}

#[test]
fn evaluation_keeps_unknown_pairs_explicit_and_normalizes_queries() {
    let fingerprints = HashMap::from([
        (
            "king".into(),
            WordFingerprint::new("king".into(), &[0, 1], 16),
        ),
        (
            "queen".into(),
            WordFingerprint::new("queen".into(), &[0, 1], 16),
        ),
        (
            "cat".into(),
            WordFingerprint::new("cat".into(), &[2, 3], 16),
        ),
    ]);
    let retina = Retina::with_index(fingerprints, 4, FingerprintConfig::default());
    let pair = |word1: &str, word2: &str, score| WordPair {
        word1: word1.into(),
        word2: word2.into(),
        score,
    };
    let pairs = vec![
        pair("KING", "Queen", 9.0),
        pair("king", "cat", 0.0),
        pair("unknown", "cat", 5.0),
    ];
    let report = evaluate_pairs(&retina, &pairs).unwrap();
    assert_eq!(report.evaluated_pairs, 2);
    assert_eq!(report.coverage, 2.0 / 3.0);
    assert_eq!(report.spearman, Some(1.0));
    assert_eq!(report.predictions[2].predicted, None);
    assert_eq!(evaluate_pairs(&retina, &[]).unwrap().spearman, None);
    assert!(evaluate_pairs(&retina, &[pair("king", "queen", f64::NAN)]).is_err());
}

#[test]
fn angular_retinas_preserve_substitution_and_provenance_across_readers() {
    use proteus::training::angular::{train_distributional, AngularConfig};

    for seed in [1, 42, 123] {
        let config = AngularConfig {
            dimension: 16,
            min_count: 1,
            max_document_frequency: 1.0,
            document_mode: DocumentMode::Lines,
            embedding_dimension: 8,
            seed,
            ..AngularConfig::default()
        };
        let (first, report) =
            train_distributional(Cursor::new(CORPUS), config.clone(), |_, _| {}).unwrap();
        let (second, _) = train_distributional(Cursor::new(CORPUS), config, |_, _| {}).unwrap();
        assert_eq!(first.word_similarity("king", "queen").unwrap(), 1.0);
        assert!(first.word_similarity("king", "cat").unwrap() < 1.0);
        assert!(!first.has_spatial_topology());
        assert!(first.training_report().is_none());
        assert_eq!(report.active_bits, 6);

        let directory = tempfile::tempdir().unwrap();
        let first_path = directory.path().join("first.retina");
        let second_path = directory.path().join("second.retina");
        first.save(&first_path).unwrap();
        second.save(&second_path).unwrap();
        assert_eq!(
            std::fs::read(&first_path).unwrap(),
            std::fs::read(&second_path).unwrap()
        );
        let loaded = Retina::load(&first_path).unwrap();
        let mapped = proteus::MmappedRetina::open(&first_path).unwrap();
        assert_eq!(
            first.find_similar_words("KING", 10).unwrap(),
            loaded.find_similar_words("KING", 10).unwrap()
        );
        assert_eq!(
            first.find_similar_words("KING", 10).unwrap(),
            mapped.find_similar("KING", 10).unwrap()
        );
        assert!(!loaded.has_spatial_topology());
        assert_eq!(
            serde_json::to_value(first.angular_report()).unwrap(),
            serde_json::to_value(loaded.angular_report()).unwrap()
        );
    }
}

#[test]
fn angular_thread_count_does_not_change_fingerprints() {
    use proteus::training::angular::{train_distributional, AngularConfig};

    let mut config = AngularConfig {
        dimension: 16,
        min_count: 1,
        max_document_frequency: 1.0,
        document_mode: DocumentMode::Lines,
        embedding_dimension: 8,
        threads: 1,
        ..AngularConfig::default()
    };
    let (first, _) = train_distributional(Cursor::new(CORPUS), config.clone(), |_, _| {}).unwrap();
    config.threads = 2;
    let (second, _) = train_distributional(Cursor::new(CORPUS), config, |_, _| {}).unwrap();
    for (word, fingerprint) in first.word_fingerprinter().iter() {
        assert_eq!(
            &fingerprint.fingerprint,
            second.get_word_fingerprint(word).unwrap()
        );
    }
}

#[test]
fn angular_legacy_metadata_preserves_unwhitened_semantics() {
    use proteus::training::angular::AngularConfig;

    let mut serialized = serde_json::to_value(AngularConfig::default()).unwrap();
    serialized.as_object_mut().unwrap().remove("whiten");
    let legacy: AngularConfig = serde_json::from_value(serialized).unwrap();

    assert!(!legacy.whiten);
    assert!(AngularConfig::default().whiten);
    legacy.validate().unwrap();
}

#[test]
fn inconsistent_angular_metadata_cannot_replace_an_existing_model() {
    use proteus::training::angular::{train_distributional, AngularConfig};

    let config = AngularConfig {
        dimension: 16,
        min_count: 1,
        max_document_frequency: 1.0,
        embedding_dimension: 8,
        document_mode: DocumentMode::Lines,
        ..AngularConfig::default()
    };
    let (mut retina, mut report) =
        train_distributional(Cursor::new(CORPUS), config, |_, _| {}).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("protected.retina");
    retina.save(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    report.vocabulary += 1;
    retina.set_angular_report(report);

    assert!(retina.save(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[test]
fn angular_case_policy_survives_regular_and_mapped_loading() {
    use proteus::training::angular::{train_distributional, AngularConfig};

    let config = AngularConfig {
        dimension: 16,
        min_count: 1,
        max_document_frequency: 1.0,
        embedding_dimension: 8,
        document_mode: DocumentMode::Lines,
        text: proteus::TextConfig {
            lowercase: false,
            ..Default::default()
        },
        ..AngularConfig::default()
    };
    let corpus = "King royal Crown\nQueen royal Crown\nDog furry paws\nCat furry paws\n";
    let (retina, _) = train_distributional(Cursor::new(corpus), config, |_, _| {}).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("case-sensitive.retina");
    retina.save(&path).unwrap();
    let loaded = Retina::load(&path).unwrap();
    let mapped = proteus::MmappedRetina::open(&path).unwrap();

    assert!(loaded.contains("King"));
    assert!(!loaded.contains("king"));
    assert!(mapped.get("King").is_some());
    assert!(mapped.get("king").is_none());
    assert_eq!(
        retina.find_similar_words("King", 5).unwrap(),
        loaded.find_similar_words("King", 5).unwrap()
    );
}
