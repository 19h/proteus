# Library guide

[Documentation index](../README.md#documentation) · [CLI guide](cli.md) · [Advanced APIs](advanced.md)

## Build and embed

Proteus exposes the same retina, fingerprint and training types used by the CLI.
A sibling checkout can be used as a path dependency:

```toml
[dependencies]
proteus = { path = "../proteus" }
```

Build the consuming crate with nightly Rust; Proteus currently enables
`portable_simd`. Generate the full API reference locally with
`cargo +nightly doc --no-deps --open`. The public surface is collected in
[`src/lib.rs`](../src/lib.rs).

## Load, fingerprint and query

```rust,no_run
use proteus::{Result, Retina};

fn main() -> Result<()> {
    let retina = Retina::load("english.retina")?;
    let text = retina.fingerprint_text("the king and queen");
    println!("{} active cells", text.cardinality());
    println!("cosine: {}", retina.text_similarity("king", "queen"));
    for (word, score) in retina.find_similar_words("king", 10)? {
        println!("{word}: {score:.4}");
    }
    if let Some(report) = retina.training_report() {
        println!("{}: {}", report.algorithm, report.corpus_sha256);
    }
    Ok(())
}
```

Word lookup uses the model's saved normalization. Unknown words produce
`WordNotFound` in word-query methods; unknown tokens contribute no bits to text
fingerprints. An entirely unknown text has an empty fingerprint and zero cosine
similarity. Search ties use lexical word order.

## Train a retina

```rust,no_run
use proteus::training::{train_corpus, TrainingConfig};
use std::{fs::File, io::BufReader};

fn main() -> proteus::Result<()> {
    let mut config = TrainingConfig::default();
    config.som.dimension = 16;
    config.som.initial_radius = 8.0;
    config.som.iterations = 1000;
    let (retina, report) = train_corpus(
        BufReader::new(File::open("corpus.txt")?), config, |_, _| {}
    )?;
    retina.save("english.retina")?;
    println!("{} words; error {}", report.vocabulary, report.quantization_error);
    Ok(())
}
```

`TrainingConfig` owns input boundaries and feature representation as well as
SOM, text and fingerprint settings. The callback reports online updates;
the final assignment pass is additional work. `TrainingReport` records the
resolved configuration, normalized-corpus digest, occupied cells and mean
quantization error. See the [algorithm contract](algorithm.md) for equations and
the [CLI configuration tables](cli.md#training-configuration) for exposed knobs.

The public `FeatureModel`, `ProjectionMethod` and `FoldingConfig` types in
`proteus::training` select explicit context-map alternatives.
`TrainingConfig::default()` retains TF-IDF, random/exact projection and raw
incidence. `FoldingConfig::gaussian()` constructs the optional one-cell Gaussian
preset. See the [quality contract](quality.md) for semantics and limits.

The separate `proteus::training::angular::{train_distributional, AngularConfig}`
API produces non-spatial angular retinas and an `AngularReport`. The report is
available through `Retina::angular_report()` and survives migration.
`Retina::has_spatial_topology()` distinguishes these models from the legacy
spatial interpretation. Its progress callback counts encoded words, not updates.

For labelled pairs, `proteus::evaluation::{read_word_pairs, evaluate_pairs}`
returns all predictions, missing-vocabulary coverage and tie-aware Spearman
correlation. Undefined correlations are `None`; missing words are not scored as
zero. The CLI exposes this as `evaluate`.

For numerical vectors, `SomTrainer::train_vectors` takes a mutable `Som`,
`&[Vec<f32>]` and a progress callback, returning final-map BMU indices.
`train`/`train_fast` are compatibility adapters. `WordEmbeddings` is the legacy
random-indexing adapter, not a skip-gram model or the corpus trainer.

## Configuration ownership

| Type | Controls | Boundary |
| --- | --- | --- |
| `TrainingConfig` | Corpus records, snippets, features and component configurations | Main entry point for `train_corpus`; sets the SOM vector dimension from features |
| `SomConfig` | Grid, topology, update count, schedules, seed and threads | Changing `dimension` in Rust does not automatically change `initial_radius` |
| `TextConfig` | Case, NFD normalization, ASCII punctuation, numeric filtering, token byte lengths | Defaults retain numbers; byte lengths are not Unicode character counts |
| `FingerprintConfig` | Density cap, absolute bit cap and weighted text aggregation | Density and absolute caps both apply; `min_active_bits` does not manufacture evidence |
| `StorageConfig` | Legacy storage configuration fields | The CLI does not expose a generic config-file loader; explicit `Retina`/`MmappedRetina` calls select loading behavior |

Definitions and defaults: [`training.rs`](../src/training.rs) and
[`config.rs`](../src/config.rs). The aggregate `Config` remains available for
component-level consumers; it is not interchangeable with `TrainingConfig`.

## SDRs and semantic operations

```rust
use proteus::Sdr;

let a = Sdr::from_positions(&[1, 2, 3], 256);
let b = Sdr::from_positions(&[2, 3, 4], 256);
assert_eq!(a.overlap_count(&b), 2);
assert_eq!(a.union(&b).cardinality(), 4);
assert_eq!(a.xor(&b).cardinality(), 2);
```

| API | Meaning |
| --- | --- |
| `Sdr::{overlap, union, xor}` | Bitmap intersection, union and symmetric difference |
| `Sdr::{cosine_similarity, jaccard_similarity, overlap_similarity}` | Cosine, intersection/union, and intersection/minimum-cardinality scores |
| `Retina::semantic_center` | Retain cells supported by a strict majority of supplied words; unknown words still count in the majority threshold |
| `Retina::semantic_difference` | Cells present in the first word but absent from the second |
| `Retina::analogy(a, b, c, k)` | Rank words against the signed vector `B + C - A` |
| `Retina::{words_at_position, words_at_positions}` | Inspect words attached to one or several grid positions |

Use compatible grid sizes for bitmap operations. Cell numbers are comparable
within one model; matching dimensions alone do not align independently trained
models. Arithmetic behavior does not establish linguistic analogy quality.

## Sentence and topic segmentation

`SentenceSegmenter` wraps the vendored wtpsplit SaT implementation. Constructing
it can download a model and requires ONNX Runtime. `segment`, `segment_batch`
and `predict_proba` expose sentences and boundary probabilities.

`SemanticSegmenter::with_config(retina, config)` consumes a `Retina` and detects
topic boundaries from adjacent block fingerprints. Set
`SegmentationConfig { fast_mode: true, ..Default::default() }` for regex sentence
splitting without loading a neural model. Its `segment` result contains text,
sentence ranges, boundaries, similarity scores and depth scores.

This topic splitter adapts [Hearst's TextTiling](https://aclanthology.org/J97-1003/)
to SDRs. Corpus-training snippets are a separate token-bounded representation.
See [segmentation and images](cli.md#segmentation-and-images) for CLI examples.

## Persistence

`Retina::load` materializes a model and creates an index if necessary.
`MmappedRetina::open` provides mapped word lookup and similarity search;
`get` returns an owned decoded `Sdr`. Opening validates the file and index and
is not constant-time. Do not modify a mapped backing file in place.

`Retina::save` atomically writes the current format with available metadata.
Loading an old file cannot recover missing training provenance or change its
learned geometry. See [format and migration](architecture.md#format-and-migration).
