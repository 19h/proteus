# Build and command-line guide

[Documentation index](../README.md#documentation) · [Library guide](library.md)

## Build

From the repository root:

```sh
cargo +nightly build --release
./target/release/proteus --help
```

The crate uses Rust edition 2021 and nightly `portable_simd`, including the
vendored bitmap code. The recorded compiler baseline is rustc
1.100.0-nightly (2026-09-05); the old Rust 1.70 claim does not apply.
`Cargo.toml` defines no default features. `--features gpu` adds standalone wgpu
helpers; it does not select GPU training in the CLI.

ONNX Runtime and the Hugging Face client are dependencies of the neural sentence
splitter. Building may need dependency downloads. Ordinary training and queries
do not load a neural model; neural segmentation can download one when invoked.

## Train and query

The commands below assume `target/release` is on `PATH` and `corpus.txt` contains
UTF-8 text. Blank lines separate documents; wrapped lines are joined.

```sh
proteus train -i corpus.txt -o english.retina -d 16 -n 10000 --seed 42
proteus info english.retina
proteus similar -r english.retina king -k 10
proteus fingerprint -r english.retina 'the king and queen'
proteus similarity -r english.retina 'the king' 'the queen'
```

For one independent document per line, use `--documents lines`. Each document
is split into nonoverlapping snippets of at most 64 normalized tokens by default.
Training snippets do not use sentence detection. Query words must occur in the
retained vocabulary; these examples do not promise a particular neighbour list.

## Training configuration

`proteus train --help` lists the interface; `-v`/`--verbose` is global.

| Option | Default | Meaning |
| --- | --- | --- |
| `-i`, `--input` | required | UTF-8 corpus file |
| `-o`, `--output` | required | Destination retina, written atomically |
| `-d`, `--dimension` | 128 | Square grid side; total cells are its square |
| `-n`, `--iterations` | 100000 | Exact number of online SOM updates |
| `-s`, `--seed` | 42 | Persisted random seed |
| `--documents` | paragraphs | `paragraphs` or `lines` |
| `--snippet-tokens` | 64 | Maximum normalized tokens per context |
| `--min-count` | 2 | Minimum corpus token count for retained words |
| `--max-document-frequency` | 0.5 | Maximum snippet fraction for TF-IDF features; does not itself remove output words |
| `--projection-dim` | 0 | Zero uses vocabulary coordinates; positive enables the selected projection |
| `--feature-model` | tfidf | `tfidf` or `context-ppmi` |
| `--projection-method` | random | Seeded random coordinates or `spectral` word-context subspace |
| `--projection-iterations` | 2 | Orthogonal power iterations for spectral projection |
| `--context-window` | 5 | Maximum token distance in PPMI co-occurrence counting |
| `--context-alpha` | 0.75 | PPMI context-marginal smoothing exponent |
| `--folding-sigma` | 0 | Gaussian interpolation width in cells; zero retains raw incidence |
| `--background-exponent` | 0 | Occupancy correction power for positive folding sigma |
| `--initial-learning-rate` | 0.1 | Initial update coefficient |
| `--final-learning-rate` | 0.01 | Final update coefficient |
| `--initial-radius` | dimension/2 | Initial Gaussian width in cells |
| `--final-radius` | 1 | Final Gaussian width in cells |
| `--planar` | false | Disable toroidal wrapping |
| `--threads` | 0 | Rayon worker count; zero uses its default |
| `--sparsity` | 0.02 | Maximum fingerprint fraction, rounded up to a cell |
| `--max-active-bits` | no extra cap | Additional absolute cardinality limit |

The iteration count is not a token count or epoch count. Every usable context is
assigned to the frozen map after training, even if the update count is smaller
than the corpus. One hundred million requested updates means one hundred million
updates; the trainer does not impose a hidden cap.

Projection reduces dense feature storage and distance work but changes geometry:

```sh
proteus train -i corpus.txt -o compact.retina -d 32 -n 10000 \
  --projection-dim 256 --seed 42
```

It has no fixed neighbour-quality guarantee. Density is a cap: low-support words
can have fewer bits. Increasing iterations or density cannot supply absent
contexts. See [training cost](algorithm.md#cost-and-bounded-extensions).

## Distributional training

`train-distributional` learns local PPMI word vectors, projects them into a
spectral subspace, then encodes sparse Gaussian response ranks. Its bit positions
are non-spatial; it does not use a SOM. Its dedicated command accepts no SOM
learning-rate, radius, topology or update-count flags.

```sh
proteus train-distributional -i corpus.txt -o distributional.retina \
    --embedding-dimension 128 --power-iterations 2 --seed 42
```

It shares `--input`, `--output`, `--dimension`, `--seed`, `--documents`,
`--snippet-tokens`, `--min-count`, `--max-document-frequency`, `--context-window`,
`--context-alpha`, `--sparsity`, `--max-active-bits` and `--threads` with the
corresponding meanings above. `--embedding-dimension` defaults to 128 and
`--power-iterations` to 2. Spectral whitening is enabled by default;
`--no-whitening` preserves spectral-direction scale. Progress counts encoded words. Model information
explicitly identifies non-spatial coordinates. `semantic-lookup` rejects these
models; exact bitmap lookup and word similarity remain defined.

The [quality guide](quality.md) gives the equations, limitations, and evaluation
protocol. This command is a separate distributional representation, not a claim
to reproduce the context-map architecture or a proprietary retina.

## Knowledge-grounded training and queries

[The semantic guide](semantic.md#prepare-train-and-inspect) gives the complete
preparation and training workflow, resource installation and configuration table.
These are native `proteus semantic` subcommands:

| Subcommand | Required arguments | Optional query controls |
| --- | --- | --- |
| `train` | `-i ANNOTATIONS --lexicon JSON -o RETINA` | All [grounded training settings](semantic.md#configuration-fingerprints-and-persistence); `-i -` reads stdin |
| `disambiguate` | `-r RETINA WORD --context TEXT` | `--pos noun`, paired `--target-start BYTE --target-end BYTE`, optional `--target-lemma WORD` |
| `contextual-similar` | `-r RETINA WORD --context TEXT` | Same target controls; `--relation similarity`, `-k 10` |
| `compare-contexts` | `-r RETINA --pairs JSONL` | Complete [context-pair schema](semantic.md#complete-contextual-evaluation), optional labels |
| `senses` | `-r RETINA WORD` | `--context TEXT`, `--sense ID` |
| `similar` | `-r RETINA WORD` | `--relation similarity`, `--context TEXT`, `--sense ID`, `-k 10` |
| `score` | `-r RETINA LEFT RIGHT` | `--relation similarity`, `--context TEXT`, `--sense ID`, explicit bilateral `--pos noun` |
| `search` | `-r RETINA --query ANNOTATIONS` | `-k 10`, `--conflict-penalty 1` |
| `evaluate` | `-r RETINA --pairs TSV` | `--score-column 3`, `--header`, `--relation similarity`, explicit `--pos-column N` |
| `events` | `-r RETINA` | `--document ID`, `-k 20` |

Outputs are JSON, including nullable pair scores and explicit evidence states.
Search queries require one complete annotated document. Relation names and exact
score meanings are specified in the [word-score contract](semantic.md#word-scores-and-sense-evidence).
These commands reject legacy models lacking semantic metadata. Ordinary `similar`
and `evaluate` dispatch to the stored graph score for grounded models; `info`
prints the semantic report. Generic bitmap lookup/composition remain bitmap
operations. `semantic-lookup` rejects both angular and grounded non-spatial grids.

`semantic train --sense-inventory observed` restricts each training occurrence
to its parser lemma and compatible POS, including a unique dependency head for
recognized phrases. The default `lexical` inventory preserves v1 behavior.
The [observed-inventory contract](semantic.md#observed-lemma-and-pos-inventory)
defines unsupported occurrences, counts, retraining and compatibility. This is a
training option, not a query-time reinterpretation of an existing model.

## Query and inspection commands

| Command | Inputs / options | Result |
| --- | --- | --- |
| `fingerprint` | `-r FILE TEXT`; `-f positions\|binary\|hex` | Text fingerprint; default format is `positions` |
| `similar` | `-r FILE WORD`; `-k COUNT` (10) | Cosine-ranked words, excluding the query; lexical tie-breaking |
| `similarity` | `-r FILE TEXT1 TEXT2` | Cosine similarity of text fingerprints |
| `info` | `FILE` | Grid, vocabulary, support diagnostics and available training provenance |
| `evaluate` | `-r FILE --pairs TSV`; `--score-column N` (3), `--header` | JSON pair predictions, coverage and tie-aware Spearman correlation |
| `migrate` | `-i OLD -o NEW` | Current file format with index and available metadata |
| `lookup` | `-r FILE 'POS,POS'`; `-k COUNT` (10), `--combined` | Words at cells; combined mode ranks raw position overlap |
| `semantic-lookup` | `-r FILE 'POS,POS'`; `-k COUNT` (20) | Spatially weighted word lookup |

`semantic-lookup` exposes `--radius` (1.8 cells), `--sigma` (1.5 cells),
`--adaptive`, `--explain`, `--clusters`, `--stats` and `--kernel`. These control
spatial scoring, not the standard word-query cosine metric. Positions use
`row * dimension + column` with zero-based coordinates.

```sh
proteus migrate -i old.retina -o indexed.retina
proteus lookup -r indexed.retina '100,101' --combined
proteus semantic-lookup -r indexed.retina '100,101' --explain --stats
```

Migration preserves available metadata; it does not retrain an old model or
recover missing provenance. See [format compatibility](architecture.md#format-and-migration).

## Segmentation and images

```sh
proteus segment -r english.retina -i document.txt --fast --debug
proteus fingerprint-image -r english.retina --text 'king queen' --words -o images
proteus fingerprint-image -r english.retina -i document.txt --segment --fast -o sections
```

`segment` compares blocks of sentences and finds depth-score boundaries. Options
are `-b`/`--block-size` (3 sentences), `-t`/`--threshold` (0.5 standard deviations),
`-m`/`--min-segment` (2 sentences), `--debug` and `-f`/`--fast`. Fast mode uses
the regex splitter; the default loads the `sat-3l-sm` neural sentence model.

`fingerprint-image` accepts `-i`/`--input` (a file or `-` for stdin) or
`-t`/`--text`; input takes precedence if both are supplied. At least one of
`--words` or `--segment` selects what to render. `-o`/`--output` defaults to
`fingerprints`; `--scale` defaults to 4 pixels per cell and must be positive.
Segment mode also accepts `--block-size`, `--threshold`, `--min-segment` and
`--fast`. Images use white active cells on a black square grid.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Stable Rust rejects SIMD features | Run the build with `+nightly` |
| No vocabulary or usable snippets | Inspect input boundaries, normalization, `--min-count` and document-frequency filtering |
| Unexpected neighbours | Check corpus senses, word support, occupied cells and projection; exact ranking is not a semantic-quality guarantee |
| Training uses excessive memory or time | Reduce grid/features or use explicit projection; update count multiplies dense SOM work |
| Neural segmentation cannot initialize | Check ONNX/model availability, or use `--fast` for regex splitting |
| Old file has unknown provenance | Missing metadata is not recoverable by migration; retraining creates new provenance |

Argument definitions: [`src/main.rs`](../src/main.rs) and
[`src/cli/quality.rs`](../src/cli/quality.rs). Numerical assumptions and
edge cases: [algorithm contract](algorithm.md#assumption-register).
