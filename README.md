# Proteus

Proteus is a Rust Semantic Folding engine. Its default trainer learns sparse binary word
fingerprints from a text corpus by mapping contexts onto a two-dimensional
self-organizing map (SOM), then recording the cells associated with each word.
Text fingerprints compose known words; shared bits support similarity search.

It is an independent implementation of the published context-map architecture,
with explicit numerical choices. It does not reproduce Cortical.io's proprietary
training implementation or distribute its pretrained retinas.

## Start with a corpus

Build with nightly Rust, then train a small model from your own UTF-8 text:

```sh
cargo +nightly build --release

./target/release/proteus train -i corpus.txt -o english.retina \
  -d 16 -n 10000 --seed 42
./target/release/proteus info english.retina
./target/release/proteus similar -r english.retina king -k 10
./target/release/proteus fingerprint -r english.retina 'the king and queen'
```

Blank lines separate documents; wrapped lines are joined. Add `--documents lines`
for one independent document per line. The query word must occur in the retained
vocabulary. A model describes the usages present in its corpus; a book-specific
model is not a general-English synonym dictionary.

The build currently requires nightly `portable_simd`. Ordinary training and
queries do not load a neural language model. The optional neural sentence
splitter uses ONNX Runtime and may download its model when invoked. See the
[build and CLI guide](docs/cli.md#build) for options and prerequisites.

## Current project shape

| Surface | Available behavior | Detail |
| --- | --- | --- |
| Corpus training | Bounded snippets, TF-IDF, optional seeded projection, online SOM and final word/cell folding | [Algorithm](docs/algorithm.md#contexts-and-features) |
| Distributional training | Local PPMI, spectral projection and non-spatial sparse angular codes | [Quality](docs/quality.md#non-spatial-distributional-retinas) |
| Fingerprints | Roaring-backed SDRs, word/text composition, cosine, Jaccard and overlap measures | [Library](docs/library.md#sdrs-and-semantic-operations) |
| Search | Exact cosine ranking, inverted index, cell and spatial neighbourhood lookup | [Queries](docs/cli.md#query-and-inspection-commands) |
| Persistence | Versioned retina files, metadata, atomic saves, validated mapped reads and legacy readers | [Format](docs/architecture.md#format-and-migration) |
| Text and visualization | Regex/neural sentence splitting, SDR-based topic segmentation and PNG fingerprints | [Segmentation and images](docs/cli.md#segmentation-and-images) |
| Additional library algorithms | HNSW, LSH, product quantization, batch/growing SOM and HDC | [Advanced APIs](docs/advanced.md) |

These are distinct interfaces. Approximate indexes, the batch trainer and GPU
helpers do not replace the reference online trainer or exact cosine query path.

## Documentation

This README is the documentation entry point. Focused pages link back here;
there is no separate `docs/README.md`.

### Build and use

- [Build and CLI](docs/cli.md) - installation, all commands, training options and troubleshooting.
- [Training configuration](docs/cli.md#training-configuration) - defaults, input boundaries, update counts and projection.
- [Segmentation and images](docs/cli.md#segmentation-and-images) - sentence/topic splitting and word or segment PNGs.
- [Library guide](docs/library.md) - embedding, training, querying, configuration and persistence.
- [Advanced APIs](docs/advanced.md) - HNSW, LSH, PQ, batch SOM, HDC and their separate contracts.

### Understand and develop

- [Architecture](docs/architecture.md) - data flow, source ownership and format compatibility.
- [Algorithm contract](docs/algorithm.md) - equations, precision, complexity, assumptions and primary references.
- [Development](docs/development.md) - test targets, measurement, contribution and documentation conventions.
- [Agent engineering guide](AGENTS.md) - repository contracts, validation and required documentation upkeep.
- [Quality and evaluation](docs/quality.md) - encoder choices, support diagnostics and labelled comparisons.
- [Validation record](docs/validation.md) - exercised boundaries, corpus results and historical binary provenance.
- [Scaling and cross-lingual research](docs/roadmap.md) - condensed roadmap, dependencies and falsification probes.

## Execution and quality boundaries

```text
UTF-8 documents -> bounded snippets -> TF-IDF -> online SOM
                                                    |
                                         final cell assignments
                                                    |
                                        word incidence -> SDRs
                                                    |
                                           cosine similarity
```

`-n` counts online updates exactly. Every usable context is assigned to the final
map afterward. Training keeps normalized tokens and dense vectors in memory;
this is not a terabyte-scale streaming or distributed trainer. The
[cost model](docs/algorithm.md#cost-and-bounded-extensions) describes how grid,
feature dimension and update count affect work and memory.

Seeded training is reproducible on the same numerical platform. Search uses
lexical tie-breaking and no hidden stopword/word-length filter. These properties
do not establish held-out semantic accuracy. Projection, low word support and
corpus coverage affect neighbours; [recorded results](docs/validation.md#corpus-execution)
show both useful and weak associations.

Old retinas remain readable. Loading or migrating them does not repair their
learned geometry or recover missing provenance. See [migration](docs/architecture.md#format-and-migration)
and the [explanation of the training changes](docs/algorithm.md#online-som).

## Verify a change

```sh
cargo +nightly test
cargo +nightly test --test training_contract
cargo +nightly test --test rebase_compat
cargo +nightly test --doc
cargo +nightly check --all-features
```

The [test map](docs/development.md#local-checks) separates numerical, compatibility
and CLI coverage. Neural-model and large bitmap examples include explicit skips;
compiling optional GPU code does not establish GPU runtime behavior.

## Name, attribution and license

Proteus is named after the shape-changing figure in Greek mythology. The project
builds on Semantic Folding research, SOMs, TextTiling, Roaring bitmaps, wtpsplit
and Rayon. Primary references appear with the relevant
[algorithm](docs/algorithm.md#primary-references) and
[advanced APIs](docs/advanced.md#hyperdimensional-computing).

Package metadata declares MIT licensing; see
[attribution and license scope](docs/development.md#attribution-and-license).
