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

## Quality experiments, 2026-10-22

These experiments start from `e08a0677013ca7c74f347515769d1f8c93d253e1` with the
quality implementation in the working tree. The host is an Apple M4 Max,
137438953472 bytes (128 GiB) of RAM, macOS 27.2 build 26B5112a, and the nightly
compiler identified above. Release builds use the committed profile; no GPU is
involved. Trial outputs and immutable copies of intermediate executables are
under `target/validation/quality/`.

The implementation adds explicitly named feature and folding alternatives and
labelled evaluation. The reference default remains TF-IDF with raw incidence.
A failed empirical comparison is not converted into a new default or hidden
query filter. The [quality contract](quality.md) states the equations, assumptions,
precision, boundaries, and reproducibility scope.

### Benchmark protocol and initial failures

The SimLex-999 source is the [authors' distribution](https://fh295.github.io/simlex.html),
retrieved on 2026-10-22. Its UTF-8 data file SHA-256 is
`ca00ff8515a36cde32d13c2b83def23f5a167b67db93ebfcfb7dae039ba22c0b`.
Development membership is the word-pair SHA-256 rule in the quality contract:
338 reference pairs, of which 170 are covered by these Moby-Dick models. The
other 661 pairs contain 353 covered pairs. All correlations use average ranks
for ties and exclude OOV pairs only through an explicitly reported common
coverage set. The original Moby-Dick corpus and its identity are unchanged.

| Moby-Dick trial | Grid / updates / coordinates | Development Spearman | Other evidence |
| --- | --- | ---: | --- |
| Reference TF-IDF, raw incidence | 128 / 100000 / 256 random | -0.008450 | Reserved split: -0.029724 |
| TF-IDF, Gaussian sigma 1 and occupancy exponent 0.5 | 128 / 100000 / 256 random | 0.077461 | Reserved split: -0.045698 |
| Context PPMI with the same Gaussian folding | 128 / 100000 / 256 random | 0.027863 | No held-out gain established |
| Word PPMI, Euclidean prototypes, nearest-prototype bits | 128 / 100000 / 128 spectral | 0.078262 | Superseded: training and encoding metrics differed |
| Word PPMI, unit-sphere prototypes | 32 / 20000 / 128 spectral | 0.056825 | Grid and update count also differ; not an isolated comparison |
| Word PPMI, unit-sphere prototypes | 128 / 100000 / 128 spectral | 0.057929 | External WordSim-353 trial below |

The initially reserved split rejected the proposed smoothing improvement:
change in Spearman `-0.015974`, descriptive paired bootstrap 95% percentile
interval `[-0.102525, 0.074679]`, 2000 resamples, seed 42. Smoothing remains
optional. Later trials are exploratory, not fresh confirmations on that split.
Different vector dimensions and sample definitions in the word-map trials mean
that any pipeline difference cannot be attributed to one isolated parameter.

[WordSim-353](https://www.gabrilovich.com/resources/data/wordsim353/wordsim353.html)
was then evaluated as a separate relatedness benchmark. The original 353 rows,
including repeated rated pairs, are retained. On 129 commonly covered rows,
the reference model gives Spearman `0.172150`, while the first unit-sphere
word-map trial gives `0.086956`; change `-0.085195`, descriptive interval
`[-0.289970, 0.130218]`. This does not establish an improvement. General-English
accuracy from this one-book training corpus remains unsupported.

The first trials used executable SHA-256
`a6f4e8b1eaff3031f35345534791b5c44ae578361c0c04e521e779d457f920a4`.
The superseded word-map executable is
`6bcd4730dd37cfbb0c7079bbadef9543ab067f653626757a624ad5b8e2bdd3a5`.
The unit-sphere executable is
`7c67923b7b04d925f7bb1d5fff9c7ec9875b7931a0a5101dc845d19d6fb25001`.
These identities distinguish models generated before and after the geometry
correction and the CLI provenance additions.

### Correctness and interpretation

The new scalar fixtures cover mass-preserving planar and toroidal convolution,
unique wrapped offsets, bounded interpolation support, raw-incidence mode,
local PPMI without word co-occurrence, forbidden cross-snippet associations,
full-rank and rank-deficient subspace projection, normalized SOM updates,
antipodal midpoint behavior, tied rank correlation and undefined correlations.
Round trips exercise lexical, PPMI and word-map modes, exact/projected features,
legacy defaults and both normal and mapped readers. The reference scalar SOM
oracle remains unchanged and passes after adding the separate unit-sphere path.

The first word-map trial exposed a numerical-contract mismatch: it trained
Euclidean centroids but selected fingerprint cells by cosine. The replacement
keeps prototypes on the unit sphere, making the objectives agree in real
arithmetic. A binary64 scalar oracle bounds coordinate error by `5e-7` on its
fixture, and the parallel path produces identical weights with one and two
workers. Correcting this contract still does not by itself prove semantic gain.

`king` in this book also occurs in the compound name `King-Post` (see the
Stubb/Flask introduction), as well as royal and family contexts. A high-ranking
`post` association is therefore partly supported by the corpus. This observation
does not excuse other irrelevant neighbours or turn the book into a general
English synonym training set.

The unsuccessful word-prototype experiments were removed from the final public
interface. They remain in this historical record and their archived binaries,
not as an advertised feature. The final distributional backend uses direct
Gaussian angular encoding with explicit non-spatial metadata. The reference
context SOM was restored without the experimental spherical optimizer.

### Final distributional backend and broader-corpus comparison

The retained non-spatial backend uses local PPMI, orthogonal subspace iteration,
column whitening, and fixed-cardinality Gaussian response ranks. It is exposed
as `train-distributional`; it neither invokes nor claims to reproduce the SOM.
The reference `train` command retains its previous default. Whitening equalizes
spectral directions before word normalization; `--no-whitening` is an explicit
ablation. Version-1 angular metadata without this field defaults to false.

The broader corpus is the **training split only** of Salesforce's WikiText-2 raw
release, retrieved from the [publisher's dataset](https://huggingface.co/datasets/Salesforce/wikitext)
at revision `6231e49f19a707241d6b84d9cff60a3a86b85a85` on 2026-10-22.
The source parquet SHA-256 is
`49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701`.
Export preserves the 36,718 text records in order: records already ending in a
newline are copied, and every other record receives one newline, including empty
records. This preserves paragraph separators. The prepared UTF-8 corpus is
10,927,796 bytes, SHA-256
`57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92`.
An initial export that dropped empty-record separators was stopped before
quality evaluation and is excluded from the following results.

Both pipelines use this same normalized corpus, paragraph mode, 64-token
snippets, minimum count 2, maximum feature document frequency 0.5, a 128x128 bit
universe, and two workers. The reference uses 256 random feature coordinates and
100000 online SOM updates. Distributional training uses 128 spectral coordinates,
two subspace iterations, whitening, and 328 active angular features. This is an
end-to-end pipeline comparison, not attribution of all improvement to one knob.

| Seed | SimLex test reference | SimLex test distributional | WordSim-353 reference | WordSim-353 distributional |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.043959 | 0.078010 | 0.299496 | 0.407838 |
| 42 | 0.063827 | 0.181671 | 0.281257 | 0.366705 |
| 123 | 0.047033 | 0.158883 | 0.271364 | 0.364061 |
| Median | 0.047033 | 0.158883 | 0.281257 | 0.366705 |

The SimLex test subset contains 661 reference pairs, with **628 covered by both
models** at every seed. WordSim-353 contains 353 rated observations, with **329
commonly covered** at every seed. Coverage is unchanged by the pipeline change.
The final distributional model has 39,141 words; four retained words with zero
context vectors are explicitly omitted. No query-time stopword or word-length
filter changes ranking.

At seed 42, the paired SimLex test difference is `+0.117844`, with descriptive
95% percentile interval `[0.037803, 0.199691]`. The WordSim difference is
`+0.085448`, interval `[-0.021323, 0.196412]`. These intervals use 2000 pair
resamples and seed 42. Pairs can share words, datasets were examined during
iterative development, and three training seeds are not three independent
linguistic datasets: these are bounded empirical results, not a confirmatory
universal-accuracy claim. Absolute correlations remain modest.

Moby-Dick remains a counterexample to a blanket improvement claim. With final
whitening, its SimLex test result is `-0.002618` versus reference `-0.029724`, and
WordSim-353 is `0.168456` versus `0.172150`. The broader-corpus result does not
retroactively establish a high-quality general-English retina from the novel.

The measured whitening executable SHA-256 is
`baf3da7c8e48afa32193d0b603915a89f94e4e04fc7780ceb7a64a69f3eac8da`.
An intermediate release executable was
`cd5181b725894b058170d9258f9e607511837974dee9842831c01f0e21134917`.
The final source build is
`83751d090746d682ed83b64622bc3957017ade010761cc7c5fcf264fd317a601`.
Fresh full WikiText-2 runs with both executables reproduced the entire seed-42
file byte-for-byte: model SHA-256
`8be5095570819b7f220ac6ef4a1628128d781351b1657c7a42c51f1ae4dc0166`.
Four independent queries also produced identical output through the current
reader and two earlier v4 readers. Current migration retains angular metadata;
older writers that do not know it can discard it and should not be used to
preserve the non-spatial contract.

Reproduction commands, using the prepared corpus and downloaded benchmark files:

```sh
# Reference, repeated with --seed 1, 42, and 123.
proteus train -i wikitext-2-documents.txt -o reference.retina \
    -d 128 -n 100000 --projection-dim 256 --threads 2 --seed 42

# Distributional, repeated with the same three seeds.
proteus train-distributional -i wikitext-2-documents.txt -o distributional.retina \
    --embedding-dimension 128 --power-iterations 2 --threads 2 --seed 42

proteus evaluate -r reference.retina --pairs SimLex-999.txt \
    --score-column 4 --header > reference-simlex.json
proteus evaluate -r distributional.retina --pairs SimLex-999.txt \
    --score-column 4 --header > distributional-simlex.json
python3 scripts/compare_quality.py reference-simlex.json distributional-simlex.json \
    --split test --resamples 2000 --seed 42
```

The local artifact directory contains model files, configuration/provenance,
evaluation predictions, common-coverage comparisons, binary hashes and command
output. These generated artifacts are not added to source control. Timings in
those logs are rounded CLI durations from shared-host runs; they are not an
isolated performance benchmark.

### Final acceptance checks

`cargo test --locked` completed successfully: 167 library tests, 40 CLI and
integration/contract tests, and 95 Rust documentation examples (302 checks).
The existing six ignored cases remain explicit. This includes the long HNSW
recall case. An earlier concurrent-build run lost an intermediate rlib while
running doctests; the final sequential acceptance run passed all 95 examples.

`cargo check --locked --all-features` also passed. It establishes compilation,
not GPU runtime behavior. Clippy completed with existing warnings outside the
new training/evaluation modules. The changed handwritten Rust files pass their
scoped rustfmt checks with four-space indentation; unrelated formatting was
preserved. Relative documentation targets resolve, and both invalid-input
publication checks and angular metadata migration checks pass.

`target/validation/quality/manifest.json` records model/input/executable hashes,
per-seed comparisons, and a source snapshot for this working-tree implementation.
Detailed command output and acceptance logs are retained alongside it. The
pre-existing user corpora, models and other untracked files were not modified.
