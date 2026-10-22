# Architecture and repository map

[Documentation index](../README.md#documentation) · [Algorithm contract](algorithm.md)

## Data flow

```text
UTF-8 documents -> normalized bounded snippets -> TF-IDF [-> projection]
                                                        |
                                                  online SOM
                                                        |
                                               frozen-map assignment
                                                        |
                                             word/cell incidence
                                                        |
                                               capped SDR bitmaps
                                                        |
                      .retina <-> Retina + inverted index -> cosine queries
                              <-> MmappedRetina
```

The retina contains word fingerprints on a shared two-dimensional grid. A cell
represents a learned region of context space. Text composition combines known
word fingerprints; bitmap overlap supplies cosine, Jaccard and overlap scores.
No neural language model is used by the corpus-training path.

The online trainer computes current exhaustive best-matching units and updates
each cell once per sample with a Gaussian neighbourhood. A final assignment
pass builds word/cell incidence. Seeded reproducibility is scoped to the same
numerical platform, binary, inputs and configuration. The [algorithm](algorithm.md)
specifies schedules, ties, floating-point behavior and sparsification.

The default diagram shows context-incidence folding. Optional PPMI coordinates,
spatial interpolation, and non-spatial angular codes are separate named choices described
in the [quality guide](quality.md). Angular codes select Gaussian response features directly;
they do not claim snippet-incidence or spatial-neighborhood semantics.

## Source ownership

| Area | Source | Responsibility |
| --- | --- | --- |
| CLI | [`main.rs`](../src/main.rs), [`cli/quality.rs`](../src/cli/quality.rs) | Argument mapping, quality commands and PNG rendering |
| Corpus training | [`training.rs`](../src/training.rs), [`training/corpus.rs`](../src/training/corpus.rs) | Input records, configuration, provenance and shared corpus preparation |
| Feature encoding | [`training/features.rs`](../src/training/features.rs), [`training/spectral.rs`](../src/training/spectral.rs) | Lexical/PPMI vectors and orthogonal subspace projection |
| Fingerprint construction | [`training/folding.rs`](../src/training/folding.rs), [`training/angular.rs`](../src/training/angular.rs) | Optional spatial interpolation and non-spatial distributional encoding |
| Quality evaluation | [`evaluation.rs`](../src/evaluation.rs) | Labelled pair parsing, coverage, predictions and rank correlation |
| Configuration | [`config.rs`](../src/config.rs) | SOM, normalization, fingerprint and legacy storage options |
| Text | [`text/`](../src/text/) | Tokenization, normalization and sentence-model wrapper |
| Neural sentence model | [`wtpsplit/`](../src/wtpsplit/) | Vendored SaT model loading and inference |
| SOM | [`som/`](../src/som/) | Map representation, online and batch trainers, SIMD helpers |
| Fingerprints | [`fingerprint/`](../src/fingerprint/) | SDR algebra, word/text composition and HDC |
| Search | [`index/`](../src/index/) | Postings, mapped index, HNSW, LSH, PQ and spatial lookup |
| Similarity | [`similarity/`](../src/similarity/) | Cosine, Jaccard and overlap measures |
| Persistence | [`storage/`](../src/storage/) | Retina object, serialization, validation and mapped reads |
| Optional acceleration | [`accel/`](../src/accel/) | GPU helpers; separate from CLI online training |
| Bitmap storage | [`roaring/`](../src/roaring/) | Vendored compressed bitmap implementation |

Low-level and experimental modules remain available independently of the main
training path. See [advanced interfaces](advanced.md) for their metrics and use.

## Format and migration

New files use format v4: a 64-byte header, JSON metadata, word index, bitmaps,
string pool, and optional persisted posting index. Metadata includes inference
configuration, frequencies, POS tags and an optional map or angular training report.
Canonical ordering supports same-environment byte reproducibility.

Readers accept upstream v1, v2 (bincode index), v3 (mapped index), and the earlier
local metadata-only v2 layout. Version, flags and offsets distinguish layouts.
Legacy missing metadata remains unknown or uses documented defaults.

Both loaded and mapped readers validate bounds, UTF-8, bitmap structure and
index agreement. Saving uses a temporary file in the destination directory and
atomic replacement. Mapping avoids eagerly retaining decoded word bitmaps, but
opening still validates the file; it is not an O(1) operation.

File sections use 32-bit offsets and individual record lengths use 16 bits.
Oversized records are rejected. These limits, in-memory dense training and
single-model coordinates constrain scaling; see the [research roadmap](roadmap.md).
