# Proteus

Proteus learns binary word fingerprints from a corpus using the published
Semantic Folding architecture: text contexts are mapped to a two-dimensional
self-organizing map (SOM), and each word records the cells containing its contexts.
It is an independent implementation with explicit numerical choices, not the
proprietary Cortical.io implementation or its pretrained models.

```
UTF-8 documents -> bounded snippets -> TF-IDF -> online SOM
                                                    |
                              final snippet/cell assignments
                                                    |
                            word/context incidence -> bitmaps
                                                    |
                                   exact cosine similarity
```

[Algorithm, assumptions, equations, complexity and references](docs/algorithm.md)
describe the implementation contract. [Validation](docs/validation.md) records the
exercised boundaries and corpus experiment.

## Build

The vendored bitmap implementation and explicit SIMD currently require Rust
nightly (validated with rustc 1.100.0-nightly, 2026-09-05).

```sh
cargo +nightly build --release
cargo +nightly test
```

The binary is `target/release/proteus`. The optional neural sentence segmenter
uses ONNX Runtime and downloads its model when invoked. Ordinary training and
similarity queries do not download or invoke a language model. `--features gpu`
exposes the standalone GPU helpers; the verified online training path uses CPU
arithmetic and exhaustive BMU search.

## Train and query

```sh
# Reference feature representation: exact vocabulary TF-IDF.
# Use a modest map for a small, domain-specific corpus.
proteus train -i corpus.txt -o english.retina -d 16 -n 10000 --seed 42
proteus similar -r english.retina king -k 10
proteus fingerprint -r english.retina 'the king and queen'
proteus similarity -r english.retina 'the king' 'the queen'
proteus info english.retina
```

Blank lines delimit documents by default; physical wrapped lines are joined.
For a corpus with one independent document per line, add `--documents lines`.
Documents are split into nonoverlapping snippets of at most 64 normalized tokens.
This is a deterministic token rule, not sentence detection. Smaller or larger
contexts change the association being learned.

Exact vocabulary TF-IDF can require substantial memory and compute. A named,
opt-in approximation reduces vector dimension:

```sh
proteus train -i corpus.txt -o compact.retina -d 32 -n 10000 \
  --projection-dim 256 --seed 42
```

Projection can change neighbours. Evaluate it against exact features for the
application; it does not provide a fixed accuracy guarantee.

`-n` now means exactly the requested number of online SOM updates. It does not
mean tokens, epochs or examples retained. A final frozen-map assignment pass
always maps every usable context, even when `-n` is smaller than the corpus.
A value such as `100000000` performs one hundred million updates and can be
very expensive. It cannot compensate for missing or unrepresentative contexts.

## Why the previous results varied

The previous query implementation sampled hash-map entries to classify
stopwords, so results could change when loading the same file. Its fast training
path also ignored the requested iteration count, used stale precomputed BMUs,
and could index beyond small training sets. Seeded projections depended on
unordered vocabulary iteration; ties had no deterministic ordering. Saved files
discarded frequencies and configuration.

Queries now apply exact cosine ranking with lexical tie-breaking and no hidden
word-length or stopword filters. Training uses current BMUs, configured schedules,
unique toroidal updates and final-map assignments. Seeded runs are reproducible
on the same numerical platform. New files retain configuration, frequency,
training diagnostics and a digest of the normalized context data.

Old `.retina` files remain readable, with unknown provenance displayed by `info`.
Loading one does not repair its learned geometry: retrain to use the new pipeline.
A book-specific model describes that book's usages; general-English synonym
quality is a separate empirical question. The implementation does not promise
that `king` retrieves `queen` from every corpus.

## Exposed training parameters

Run `proteus train --help` for the full interface.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--dimension`, `-d` | 128 | Grid side; total cells are its square |
| `--iterations`, `-n` | 100000 | Online updates |
| `--seed`, `-s` | 42 | Recorded deterministic random seed |
| `--documents` | paragraphs | `paragraphs` or `lines` |
| `--snippet-tokens` | 64 | Maximum normalized tokens per context |
| `--min-count` | 2 | Minimum corpus token count for retained words |
| `--max-document-frequency` | 0.5 | Maximum snippet fraction for TF-IDF features |
| `--projection-dim` | 0 | Zero means exact; positive enables projection |
| `--initial-learning-rate` | 0.1 | Initial online update coefficient |
| `--final-learning-rate` | 0.01 | Final online update coefficient |
| `--initial-radius` | dimension/2 | Initial Gaussian width in cells |
| `--final-radius` | 1 | Final Gaussian width in cells |
| `--planar` | false | Disable toroidal wrapping |
| `--threads` | 0 | Rayon worker count; zero uses its default |
| `--sparsity` | 0.02 | Maximum fingerprint fraction, rounded up to a cell |
| `--max-active-bits` | no extra cap | Additional absolute cardinality limit |

No random padding fills a low-support word to the density limit. Rare words may
have very few bits. Higher density, a different snippet rule, or a different map
size changes the representation and requires retraining.

## Library

```rust
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

`SomTrainer::train_vectors` exposes the same optimizer for finite numerical
vectors. `train` and `train_fast` remain compatibility adapters. The legacy
`WordEmbeddings` type implements random indexing, not skip-gram, and is not used
by `train_corpus`.

## Additional upstream interfaces

The rebase retains the HNSW, LSH, product-quantization and HDC APIs, the separate
batch SOM trainer, semantic position lookup, and PNG fingerprint rendering.
These are separate algorithms; approximate indexes and the batch trainer do not
replace the reference online training and exact cosine-ranking path.

```sh
# Upgrade old layouts while retaining available metadata and adding an index.
proteus migrate -i old.retina -o indexed.retina
proteus lookup -r indexed.retina '100,101' --combined
proteus semantic-lookup -r indexed.retina '100,101' --explain
proteus segment -r indexed.retina -i document.txt --fast
proteus fingerprint-image -r indexed.retina --text 'king queen' --words -o images
```

`segment --fast` uses the upstream regex sentence splitter and does not load the
neural model. New files use format v4 to combine metadata with a persisted index.
Readers accept upstream v1/v2/v3 files and the earlier local metadata-only v2
layout. Memory-mapped searches validate their file and index before access, so
opening a file is not an O(1) operation.

## Tests and limits

```sh
cargo test --test training_contract
cargo test --test integration_tests
cargo test --doc
cargo check --all-features
```

The contract tests compare the online update against independent scalar
calculations, exercise large-radius toroidal maps and final assignments, compare
seeded model bytes, reload queries repeatedly, compare indexed and exhaustive
search with unequal bit counts, validate malformed files and preserve legacy
file loading. Controlled topic tests cover several seeds. These are correctness
checks, not a substitute for a held-out semantic benchmark.

The implementation keeps normalized corpus tokens and dense context vectors in
memory. It is not a terabyte-scale streaming trainer. File sections retain
32-bit offsets and 16-bit individual record lengths; oversized records are
rejected. The CPU path is validated independently of the optional GPU helpers
and neural segmentation model.
