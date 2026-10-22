use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proteus"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn cli_training_reload_and_binary_shape_are_reproducible() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("corpus.txt");
    std::fs::write(
        &input,
        "king queen crown palace\ncat dog fur paws\nking queen royal throne\ncat dog pets tail\n",
    )
    .unwrap();
    let paths = [dir.path().join("a.retina"), dir.path().join("b.retina")];
    for path in &paths {
        let out = run(&[
            "train",
            "-i",
            input.to_str().unwrap(),
            "-o",
            path.to_str().unwrap(),
            "-d",
            "4",
            "-n",
            "30",
            "--documents",
            "lines",
            "--min-count",
            "1",
            "--seed",
            "42",
        ]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert_eq!(
        std::fs::read(&paths[0]).unwrap(),
        std::fs::read(&paths[1]).unwrap()
    );
    let path = paths[0].to_str().unwrap();
    let first = run(&["similar", "-r", path, "king"]);
    assert!(first.status.success());
    for _ in 0..5 {
        assert_eq!(first.stdout, run(&["similar", "-r", path, "king"]).stdout);
    }
    let binary = run(&["fingerprint", "-r", path, "king", "--format", "binary"]);
    let binary = String::from_utf8(binary.stdout).unwrap();
    assert_eq!(binary.lines().count(), 4);
    assert!(binary.lines().all(|l| l.len() == 4));
    let info = run(&["info", path]);
    let info = String::from_utf8(info.stdout).unwrap();
    assert!(info.contains("semantic-folding-tfidf-online-som-v1"));
    assert!(info.contains("corpus_sha256"));
    let migrated = dir.path().join("migrated.retina");
    let out = run(&["migrate", "-i", path, "-o", migrated.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let old = proteus::Retina::load(path).unwrap();
    let new = proteus::Retina::load(&migrated).unwrap();
    assert_eq!(
        serde_json::to_value(old.training_report()).unwrap(),
        serde_json::to_value(new.training_report()).unwrap()
    );
    assert_eq!(
        old.find_similar_words("king", 10).unwrap(),
        new.find_similar_words("king", 10).unwrap()
    );
    for args in [
        vec!["lookup", "-r", path, "0", "--combined"],
        vec!["semantic-lookup", "-r", path, "0", "-k", "2"],
        vec![
            "segment",
            "-r",
            path,
            "-i",
            input.to_str().unwrap(),
            "--fast",
        ],
    ] {
        let out = run(&args);
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let images = dir.path().join("images");
    let out = run(&[
        "fingerprint-image",
        "-r",
        path,
        "--text",
        "king queen",
        "--words",
        "-o",
        images.to_str().unwrap(),
        "--scale",
        "1",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let pngs: Vec<_> = std::fs::read_dir(images)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    assert_eq!(pngs.len(), 2);
    for png in pngs {
        assert_eq!(
            image::ImageReader::open(png)
                .unwrap()
                .into_dimensions()
                .unwrap(),
            (4, 4)
        );
    }
    assert!(!run(&["similar", "-r", path, "missingword"])
        .status
        .success());
    assert!(
        !run(&["fingerprint", "-r", path, "king", "--format", "unknown"])
            .status
            .success()
    );
}

#[test]
fn cli_rejects_invalid_parameters_and_corpus_without_publishing() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.txt");
    let output = dir.path().join("out.retina");
    std::fs::write(&input, "king queen\ncat dog").unwrap();
    for extra in [
        ["-d", "0"],
        ["-n", "0"],
        ["--sparsity", "NaN"],
        ["--snippet-tokens", "0"],
        ["--initial-radius", "0"],
    ] {
        let mut args = vec![
            "train",
            "-i",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ];
        args.extend(extra);
        assert!(!run(&args).status.success());
        assert!(!output.exists());
    }
    // Invalid UTF-8 must not turn into a successful, silently truncated model.
    std::fs::write(&input, b"king queen\n\xff\ncat dog").unwrap();
    assert!(!run(&[
        "train",
        "-i",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap()
    ])
    .status
    .success());
    assert!(!output.exists());
}

#[test]
fn cli_evaluation_reports_coverage_and_rejects_nonfinite_labels() {
    let directory = tempfile::tempdir().unwrap();
    let model_path = directory.path().join("model.retina");
    let pairs_path = directory.path().join("pairs.tsv");
    let fingerprints = std::collections::HashMap::from([
        (
            "king".into(),
            proteus::WordFingerprint::new("king".into(), &[0, 1], 16),
        ),
        (
            "queen".into(),
            proteus::WordFingerprint::new("queen".into(), &[0, 1], 16),
        ),
        (
            "cat".into(),
            proteus::WordFingerprint::new("cat".into(), &[2, 3], 16),
        ),
    ]);
    let retina = proteus::Retina::with_index(fingerprints, 4, Default::default());
    retina.save(&model_path).unwrap();
    std::fs::write(
        &pairs_path,
        "word1\tword2\tpos\tscore\nKING\tQueen\tN\t9\nking\tcat\tN\t0\nmissing\tcat\tN\t5\n",
    )
    .unwrap();
    let arguments = [
        "evaluate",
        "-r",
        model_path.to_str().unwrap(),
        "--pairs",
        pairs_path.to_str().unwrap(),
        "--header",
        "--score-column",
        "4",
    ];

    let output = run(&arguments);
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["model_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(report["reference_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(report["total_pairs"], 3);
    assert_eq!(report["evaluated_pairs"], 2);
    assert_eq!(report["spearman"], 1.0);
    assert!(report["predictions"][2]["predicted"].is_null());

    std::fs::write(
        &pairs_path,
        "word1\tword2\tpos\tscore\nking\tqueen\tN\tNaN\n",
    )
    .unwrap();
    let invalid_output = run(&arguments);
    assert!(!invalid_output.status.success());
    assert!(invalid_output.stdout.is_empty());
}

#[test]
fn distributional_cli_preserves_metadata_and_rejects_spatial_lookup() {
    let directory = tempfile::tempdir().unwrap();
    let corpus_path = directory.path().join("corpus.txt");
    let model_path = directory.path().join("angular.retina");
    let migrated_path = directory.path().join("migrated.retina");
    std::fs::write(
        &corpus_path,
        "king wears royal crown\nqueen wears royal crown\ncat has fur paws\ndog has fur paws\n",
    )
    .unwrap();
    let output = run(&[
        "train-distributional",
        "-i",
        corpus_path.to_str().unwrap(),
        "-o",
        model_path.to_str().unwrap(),
        "-d",
        "16",
        "--embedding-dimension",
        "8",
        "--min-count",
        "1",
        "--max-document-frequency",
        "1",
        "--documents",
        "lines",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let info = run(&["info", model_path.to_str().unwrap()]);
    let info = String::from_utf8(info.stdout).unwrap();
    assert!(info.contains("distributional-ppmi-spectral-angular-v2"));
    assert!(info.contains("non-spatial"));
    assert!(
        run(&["similar", "-r", model_path.to_str().unwrap(), "KING"])
            .status
            .success()
    );
    assert!(
        !run(&["semantic-lookup", "-r", model_path.to_str().unwrap(), "0"])
            .status
            .success()
    );

    let migration = run(&[
        "migrate",
        "-i",
        model_path.to_str().unwrap(),
        "-o",
        migrated_path.to_str().unwrap(),
    ]);
    assert!(migration.status.success());
    let original = proteus::Retina::load(&model_path).unwrap();
    let migrated = proteus::Retina::load(&migrated_path).unwrap();
    assert_eq!(
        serde_json::to_value(original.angular_report()).unwrap(),
        serde_json::to_value(migrated.angular_report()).unwrap()
    );
    assert!(!migrated.has_spatial_topology());

    let invalid_path = directory.path().join("invalid.retina");
    for extra in [
        ["--power-iterations", "0"],
        ["--embedding-dimension", "0"],
        ["--iterations", "10"],
    ] {
        let mut arguments = vec![
            "train-distributional",
            "-i",
            corpus_path.to_str().unwrap(),
            "-o",
            invalid_path.to_str().unwrap(),
        ];
        arguments.extend(extra);
        assert!(!run(&arguments).status.success());
        assert!(!invalid_path.exists());
    }
}
