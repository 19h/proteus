# Validation record

[Documentation index](../README.md#documentation) · [Development guide](development.md)

Environment: macOS arm64; rustc 1.100.0-nightly (f248f4038, 2026-09-05).
This record distinguishes algorithm correctness, execution reproducibility and
semantic quality. It does not identify an unavailable user's training corpus.

## Automated checks

The pre-rebase default-feature run passed 138 executable tests and 94 documentation
tests (232 total); six tests/examples were explicitly ignored. The all-feature
documentation run also passed 94 examples with five ignored. `cargo check
--all-features` and formatting/whitespace checks for the changed core sources
completed successfully.

`cargo test` covers library, CLI, integration, numerical-contract and documentation
tests. The numerical oracle independently implements the online SOM equations
in binary64, compares weights within `3e-7`, checks both planar and toroidal
topology, and includes a neighborhood wider than the grid.

The contract suite checks:

- Endpoint schedules, one-update training and rejected malformed vectors.
- Frozen-map assignment for all samples, independent of training draw count.
- Bit-identical training with different Rayon thread counts.
- Controlled separation of royalty, animals and shipping contexts across three seeds.
- Byte-identical retraining and repeated-load queries, with and without projection.
- Token frequencies versus per-snippet word incidence; no synthetic padding.
- Explicit line/paragraph boundaries, invalid UTF-8 and empty corpora.
- Exact indexed/exhaustive equivalence under variable cardinality, ties, empty
  queries and zero/very large result counts.
- Truncated, invalid-version and invalid-offset files in ordinary and mapped readers.
- Legacy version 1 loading without invented training provenance.
- Saved custom case normalization and failed-save preservation of an existing file.
- Signed analogy arithmetic, separately from linguistic analogy performance.

The CLI tests train two files in separate processes, compare bytes, reload queries
repeatedly, check non-128 binary grids, inspect saved provenance and reject invalid
arguments/input without publishing an output file.

The neural sentence-model test and large bitmap examples remain explicitly
ignored. They are outside the corpus-training path. GPU feature checks compile
the optional helpers and their documentation example; they do not establish
GPU execution, numerical parity or performance. Clippy reports existing warnings
in optional GPU, segmentation and vendored code; it is not warning-free.

## Corpus execution

The supplementary source is Project Gutenberg's edition 2701 of *Moby Dick*:
<https://www.gutenberg.org/cache/epub/2701/pg2701.txt>.
The downloaded file's Gutenberg start/end wrapper is removed; the book's contents,
front matter and paragraph boundaries are otherwise retained. This differs from
the unavailable `../../data/mobydick.txt` in the original report.

`scripts/evaluate_corpus.py` records corpus, executable and model SHA-256 hashes,
exact command, elapsed time, saved model diagnostics and ten independent reloads
for each queried word. It refuses to report if the executable changes during
evaluation. The output is JSON, not an accuracy score.

```sh
python3 scripts/evaluate_corpus.py target/validation/mobydick.txt \
  target/validation/final --dimension 32 --iterations 20000 --projection-dim 0
python3 scripts/evaluate_corpus.py target/validation/mobydick.txt \
  target/validation/final --dimension 128 --iterations 100000 --projection-dim 256
```

The exact-feature and projected-feature runs are **not** a controlled comparison:
their grid sizes and iteration counts also differ. They exercise both paths and
the original 128x128 grid size, with an explicitly chosen projection for bounded
runtime. The original 100,000,000-update command has not been run.

Final release-binary results (the two training processes shared the same host):

| Setting/result | Exact TF-IDF | Projected TF-IDF |
| --- | ---: | ---: |
| Grid | 32x32 | 128x128 |
| Online updates | 20000 | 100000 |
| Vector dimension | 9842 | 256 |
| Usable/extracted snippets | 4968/4969 | 4968/4969 |
| Output words | 9847 | 9847 |
| Occupied cells | 956/1024 | 4270/16384 |
| Mean quantization error | 0.959657 | 0.869176 |
| Elapsed seconds | 37.039 | 88.838 |

For `king`, the exact-feature list starts with `queen<U+2019>s` (0.3086),
then `method` and `soul<U+2019>s` (both 0.2520). The projected list starts with
`post` (0.2434), `christendom` and `tranquo` (both 0.2041); `queen` is seventh
(0.1455). These lists directly show the remaining quality limitation. All four
queried words (`king`, `queen`, `whale`, `ship`) returned identical output across
ten independent process loads each, for both models.

The cleaned corpus SHA-256 is
`c3f600c0dd5e400b57fe410e37e9723455f67867f28c2d26aa412a7782c8effc`.
The evaluated release executable SHA-256 is
`7adbd1bd24127c3c73dbe081f91b828b908de45049fffd8b7eae057855c3395b`.
Complete commands, outputs and model hashes are retained in
`target/validation/final/d32-p0-s42.json` and
`target/validation/final/d128-p256-s42.json` (generated artifacts, not source files).

No human-rated held-out benchmark is provided, so general-English accuracy is
unknown. The observed neighbour lists contain both topical associations and weak
matches. More deterministic execution and correct SOM updates do not by themselves
establish high-quality synonym retrieval from a single novel.

## Repository scope

The starting checkout contained edits and untracked corpora/models, collector
code and semantic segmentation code. They are retained. Evaluation models were
written only under `target/validation`. The core
training implementation, CLI, search, persistence and associated documentation
were rewritten; vendored documentation import paths and SIMD API compatibility
were corrected to allow the existing checks to run.

## Rebase integration

The rewrite is integrated with upstream `9bc7e29`. Upstream ANN/HDC modules,
lookup/image commands, batch SOM API and regex segmentation are retained. The
reference trainer and exact cosine ranking retain their numerical contract.
New files use version 4 to combine metadata and the persisted posting index.
Compatibility tests distinguish upstream v2/v3 indexes from the earlier local
metadata-only v2 format and exercise all of these readers plus mmap ranking.
Migration preserves the inference/training metadata.

Validation covers 151 library tests, 29 integration/CLI/contract tests, and 95
Rust documentation examples; the existing neural-model and large bitmap example
skips remain. The long HNSW recall test was verified separately from the other
library tests. The only failing upstream assertion assumed that radius 1.5
excluded diagonal cells; it now checks the actual strict Euclidean-ball contract
at radii 1.0, 1.1 and 1.5. No neighborhood algorithm change was required.

Both pre-rebase Moby-Dick model files were loaded by the integrated CLI. Their
stored query output for `king`, `queen`, `whale` and `ship` was reproduced exactly.
The earlier corpus timings and binary hash above remain measurements of the
pre-rebase release binary, not new performance measurements of the merged code.
