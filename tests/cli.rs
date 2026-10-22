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
