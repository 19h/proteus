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

## Theory audit and independent verb evaluation

On 2026-10-22, a theory review at revision `afebd3a` evaluated the six existing
WikiText-2 models without retraining or selecting parameters. The protocol and
model hashes were recorded before downloading the official
[SimVerb-3500 attachment](https://aclanthology.org/attachments/D16-1235.Attachment.zip).
Its documented test split contains 3000 verb pairs, with scores in column 4 and
lexical-relation annotations in column 5. There is no header.

| Artifact | SHA-256 |
| --- | --- |
| Official attachment archive | `65ffdb000166c8c3e422416c17afaf5e528c2396baa3e9ec7535713cc00a86e7` |
| `SimVerb-3000-test.txt` | `4e689048a940833239733e91a313715a49e9ecb958cd6c98069bbd103f1a066f` |
| Evaluating executable | `83751d090746d682ed83b64622bc3957017ade010761cc7c5fcf264fd317a601` |
| Frozen representation probe executable | `48ea988fc38f9dc782d7199bd740b9a4e5a9499e0fcf26cddf4895cd3c06c8d6` |

Corpus and model identities are the same as in the preceding WikiText-2 record.
The normalized corpus digest is
`9892d794469ba8ff55707523820bedb6c8a6e462ec4cd333d9aece91cdd17c69`.
All six models cover the same 2381 pairs, or 79.3667% of the test set. The other
619 pairs remain explicitly missing; they are not assigned zero scores.

| Training seed | Reference SOM Spearman | Distributional Spearman | Difference | Descriptive 95% pair-bootstrap interval for difference |
| --- | ---: | ---: | ---: | --- |
| 1 | -0.009760 | 0.003263 | 0.013023 | [-0.036994, 0.060376] |
| 42 | 0.006532 | -0.013573 | -0.020104 | [-0.067939, 0.026649] |
| 123 | 0.001478 | 0.009643 | 0.008165 | [-0.035852, 0.056531] |
| Median across seeds | 0.001478 | 0.003263 | | |

Intervals use 2000 pair resamples and bootstrap seed 42, as in
`scripts/compare_quality.py`. Shared words violate pair independence; these
intervals do not establish population-level uncertainty or domain transfer.
The results do not establish a useful verb-similarity improvement.

### Locating the loss before binary encoding

An isolated diagnostic harness copied the current corpus, feature and spectral
modules. Its sole change to feature construction was an observation callback
before projection; it did not modify weights. The harness used dependency
versions from the root lockfile. A first, unused harness build resolved a fresh
dependency graph; it was replaced with the pinned build before any representation
measurements. The project lockfile and production executable were unchanged.

The probe recovered the word vectors and independently computed pairwise cosine
using binary64 accumulation. It verified the normalized corpus hash and
recreated the Gaussian feature ranking for 65 deterministically spaced words
per seed, exactly matching their stored bit positions. This validates the
diagnostic alignment on those words. It is not an independent numerical oracle
for feature construction, and it is not an exhaustive re-encoding check.

| Seed | Raw PPMI | Dense spectral, whitened | Final bitmap | Dense/bitmap rank agreement |
| --- | ---: | ---: | ---: | ---: |
| 1 | -0.002096 | 0.008803 | 0.003263 | 0.882311 |
| 42 | -0.002096 | 0.002636 | -0.013573 | 0.891827 |
| 123 | -0.002096 | 0.021957 | 0.009643 | 0.886534 |

All values are Spearman correlations on the same 2381 pairs. Raw PPMI is
deterministic and is repeated to show the stage comparison at each seed.
The final bitmap retains much of the dense pair ranking, while both are weak
against reference labels. Failure is already present before compression;
this does not identify whether corpus coverage, context definitions or sense
pooling is the dominant cause.

For comparison, the same diagnostic on all 961 commonly covered SimLex pairs
gives raw PPMI `0.097161`, dense values `0.071186` to `0.126018`, and bitmap values
`0.062398` to `0.131939`. These are **all-pair** results, not the earlier
661-pair test subset, and must not be substituted into the preceding table of
historical SimLex test results.

### Relation, overlap and frequency diagnostics

At each seed, average bitmap similarity is higher for the 86 covered antonym
pairs than for the 213 covered synonym pairs. At seed 42 these means are
`0.051758` and `0.047750`, respectively; average human reference scores are
`0.894186` and `6.720282`. This is a descriptive comparison with different word
frequencies and pair distributions, not proof that antonymy alone causes the
ranking failure.

There are 142 unordered SimVerb test pairs that also occur in the previously
inspected SimLex dataset. Excluding them as a secondary diagnostic leaves 2244
commonly covered pairs, with bitmap correlations `0.009402`, `-0.006827`, and
`0.019833` at seeds 1, 42 and 123. The main result retains every test observation.
Future experiments must treat SimVerb as inspected data after this review.

Post-hoc frequency diagnostics at seed 42 also remain weak:

| Minimum training count for each word | Covered pairs | Bitmap Spearman |
| --- | ---: | ---: |
| 2 | 2381 | -0.013573 |
| 10 | 1467 | -0.019515 |
| 100 | 228 | -0.062502 |

These subsets change the evaluation population and are not model improvements.
They show that this particular failure cannot be dismissed solely by excluding
the lowest-frequency words.

### Structural counterexamples and reproduction

The new [semantic invariant audit](../scripts/audit_semantic_invariants.py)
trained each backend on eight original and eight reversed three-token documents.
All 23 word fingerprints matched between original and reversed corpora for each
backend. `dogs chase cats` and `cats chase dogs` also produced identical nonempty
text fingerprints. Different full model hashes correctly reflected different
corpus provenance. The script independently checks permutation-invariant cosine,
lossy union, unequal-cardinality cosine distance, a full-rank whitening
counterexample and the uniform-set overlap calculation.
It also checks the proposed signed-evidence algebra on manually canonicalized
assertions: agreement scores `1`, explicit conflict `-1`, and missing or reversed
argument evidence `0` with unknown support. This validates only that algebra,
not an automatic assertion extractor or a language-quality improvement.

```sh
# Repeat for the six frozen reference/distributional model files.
proteus evaluate -r model.retina --pairs SimVerb-3000-test.txt \
    --score-column 4 > model-simverb.json

python3 scripts/compare_quality.py reference-simverb.json \
    distributional-simverb.json --split all --resamples 2000 --seed 42

python3 scripts/audit_semantic_invariants.py --binary target/release/proteus \
    --output target/validation/theory-reproduction
```

`target/validation/theory/` retains the frozen-model protocol, original benchmark,
all predictions, relation/frequency summaries, stage comparisons, isolated
representation-probe source and executable, dependency lock, hashes and command
logs. The probe's `PROTEUS_AUDIT_PAIRS` input is a four-column TSV of dataset
name, left word, right word and reference score. `PROTEUS_AUDIT_RAW` selects its
raw-PPMI output, and `PROTEUS_AUDIT_COUNTS` optionally selects word counts.
Positional arguments are retina, corpus and dense-output paths. For example:

```sh
env RAYON_NUM_THREADS=4 \
    PROTEUS_AUDIT_PAIRS=target/validation/theory/pairs.tsv \
    PROTEUS_AUDIT_RAW=target/validation/theory/ppmi-repeat.tsv \
    target/validation/theory/probe-frozen-bin \
    target/validation/quality/whitened-wiki.retina \
    target/validation/quality/wikitext-2-documents.txt \
    target/validation/theory/dense-repeat.tsv
```

These are local research artifacts, not additional production APIs. The host was
macOS on Apple M4 Max with 128 GiB RAM; the representation probe used four Rayon
workers. No runtime or memory-speed comparison is claimed. The maintained
script was executed successfully, and documentation links and Python syntax
were checked. No production Rust changed, so the prior full Rust test run is
not relabeled as a new test run for this review.

The resulting [theory and redesign](theory.md) records assumptions, formal
limitations, primary sources, complexity and falsification criteria. Its proposed
sense/relational models are unimplemented and have no measured improvement yet.

## Knowledge-grounded implementation

Recorded 2026-10-22, after the theory review above. This implements a separate
`knowledge-grounded-senses-events-v1` backend. The historical review's statements
about an unimplemented redesign refer to its reviewed revision; they do not
relabel these later results. [The semantic contract](semantic.md) specifies
representation, exact equations, preparation, scope and assumptions S1-S6.

### Resources and experimental boundary

Native training used default `SemanticConfig` (128 squared non-spatial slots,
minimum count 2, context window 5, marginal exponent 0.75, 32 association
coordinates, 3 examples per sense, 5-token phrase spans, prior mass 5,
support cutoff 0.5, temperature 0.15 and sparsity cap 0.02). There is no random
seed or SOM update count. All settings are retained in each report and model.

The external lexical source was Princeton WordNet 3.0, exported through NLTK
3.10.4: 117,659 senses, with source archive SHA-256 and the original license
retained. Its canonical JSON SHA-256 in the native reports is
`4cc1af3597bd051fdcbea24378106e1f0581487290bdb0b69afb1d5b862f06fc`.
This digest hashes the native canonical serialization, not the preparation file's
trailing newline. Source relations are knowledge inputs, not labels learned
from either corpus. No lexical-resource/benchmark disjointness is claimed.

Preparation used CPython 3.11, spaCy 3.8.16, `en_core_web_trf:3.8.0`,
spacy-curated-transformers 0.3.1 and the MPS backend on Apple M4 Max, macOS,
128 GiB RAM. Parser model directory SHA-256 was
`a1c9807372c70f535d5779576cbabf6f481fb2cd71567a921d0735ec01629f4f`.
Original paragraph text, source byte offsets, annotations and resource identities
are stored in each retina. The preparer initially exercised `en_core_web_sm`,
which misparsed `Dogs chase cats.` as a noun phrase; the extractor rejected that
query. The reported final experiments use the explicitly stronger parser.
Neither parser correctness nor contextual sense accuracy is established by model
size. Preparation runtime was not separately instrumented; no end-to-end timing
or parser throughput comparison is claimed.

| Corpus | Original UTF-8 SHA-256 | Complete paragraphs | Retained nonempty parser tokens | Word/phrase entries | Entries with senses | Extracted events |
| --- | --- | --- | --- | --- | --- | --- |
| Moby-Dick | `c3f600c0dd5e400b57fe410e37e9723455f67867f28c2d26aa412a7782c8effc` | 2,814 | 209,879 | 10,375 | 9,791 | 20,074 |
| WikiText-2 documents | `57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92` | 11,669 | 1,677,390 | 43,755 | 31,077 | 159,246 |

Phrase occurrences are additional targets; vocabulary count and frequency mass
are not raw token counts. Corpus report digests hash length-prefixed canonical
annotation records, preserving more information than a normalized word stream.
The annotation policy differs from the old snippet tokenizer. Comparisons
therefore assess the complete named backends, not an isolated optimizer change.

These benchmarks were already inspected during the prior review or implementation
prototyping. The following results are development diagnostics, not a newly
preregistered untouched holdout or a claim of cross-domain generalization. No
configuration was selected by sweeping the reported native benchmark scores.
All rated observations, including duplicate WordSim pairs, were retained. Native
WikiText best-pair and expected-score correlations agree with an independent
SciPy 1.17.1 calculation to absolute error below `1e-12` (dimensionless).

### Matched-coverage lexical results

Spearman correlations use average ranks for ties. WikiText reference/angular
columns are medians of archived seeds 1, 42 and 123, each compared on the same
common observations with the grounded model. Moby-Dick has one archived seed.
All rows use the complete supplied benchmark, rather than mixing the previous
hash test subset with all-pair results.

| Corpus / task | Common observations / total | Reference SOM | Angular | Grounded best-sense |
| --- | --- | --- | --- | --- |
| WikiText / SimLex-999 | 961 / 999 | 0.0216 | 0.1167 | 0.4968 |
| WikiText / SimVerb-3000 | 2,380 / 3,000 | 0.0019 | 0.0033 | 0.4290 |
| WikiText / WordSim-353 | 329 / 353 | 0.2813 | 0.3667 | 0.3667 |
| Moby-Dick / SimLex-999 | 522 / 999 | -0.0213 | 0.0482 | 0.4796 |
| Moby-Dick / WordSim-353 | 129 / 353 | 0.1722 | 0.1685 | 0.4741 |

Grounded-only coverage is 525/999 for Moby-Dick SimLex and 130/353 for its
WordSim; their full-covered correlations are 0.4815 and 0.4707. The table uses
522 and 129 common observations instead. Moby-Dick SimVerb covers 1,270/3,000
with correlation 0.4430; no matched archived Moby-Dick SimVerb baseline is
claimed. WikiText has one fewer covered SimVerb pair than the prior encoder;
the table recomputes every baseline on the 2,380 common pairs.

The maintained comparison script used 2,000 paired resamples, seed 42, `--split
all`. For the seed-42 angular comparison, descriptive 95% percentile intervals
for the correlation difference were [0.2878, 0.4382] on WikiText SimLex,
[0.3901, 0.4910] on SimVerb, and [-0.1341, 0.1396] on WordSim. Shared words and
source-resource dependence limit inferential interpretation. WordSim shows no
clear gain over angular codes; it is not omitted because the result is weaker.

### Score ablations and attribution

The following WikiText rows have identical covered observations within each
column. Each channel is explicitly selected on the same saved model:

| Score | SimLex, 961 pairs | SimVerb, 2,380 pairs | WordSim, 329 pairs |
| --- | --- | --- | --- |
| Best lexical sense pair | 0.4968 | 0.4290 | 0.3667 |
| Prior-weighted expected similarity | 0.2570 | 0.4135 | 0.3541 |
| Corpus-only local PPMI association | 0.0421 | 0.0463 | 0.3012 |

These results attribute most context-free lexical improvement to retained
external sense/taxonomy knowledge and the best-pair contract. They do not show
that gloss adaptation learned equivalent quality from the corpus. Expected
similarity is substantially weaker on SimLex; mixing all senses is not an
automatic cure for rare-sense false matches. Contextual weighting has controlled
fixtures and real-parser examples, but lacks a larger contextual benchmark here.

For Moby-Dick, default `king` neighbors begin with `baron`, `kings`, `mogul`,
`power`, `powers`, `queen`, `queens`, then named monarchs. The explanations expose
the lexical senses behind these ties. Supplying `the king is a male sovereign
ruler of a kingdom` shifts the list toward `kings`, `jeroboam`, `frederick`,
`ahab`, `alfred`, `edmund`, `solomon`, `tarquin`, `xerxes`, `monarchs`. This is an
inspectable example, not an independent accuracy metric. A synonym relation
means explicit shared lexical sense; it does not assert interchangeability in
every sentence. Explicit sense selection remains available.

### Bound evidence and compatibility

A nine-paragraph hand-authored corpus was parsed with the actual transformer,
trained natively, saved, reloaded and queried through the CLI with five separately
prepared queries. Relative to `Dogs chase cats.`, active and passive assertions
scored 1, reversed roles and modal assertions scored 0/unknown, and explicit
opposite negation scored -1/conflicted. The negative query reversed support and
conflict correctly. `Doctors treat patients.` matched its passive paraphrase.
The contract fixtures independently test tense, unsupported negation scope,
one-letter identifiers, non-reusable participant assignments, Unicode ranges,
phrases and absent lexical evidence. These probes verify the supported extraction
patterns, not general entailment or source truth.

Fresh Moby-Dick retraining produced byte-identical v5 files. Migration preserved
those exact bytes. Loaded/mapped rankings and small-fixture migration agree.
Current reference-model SimVerb predictions equal the archived predictions
exactly. The archived v4 binary rejects v5 with `Unsupported retina version or
flags`; it does not silently query the new coordinate semantics.

Malformed graph cycles/references, dependency cycles, invalid UTF-8 boundaries,
missing completion/footer checksums, excessive decoded metadata sizes, corrupted
zlib data and mismatched metadata are rejected. Failed preparation/training/save
attempts preserve existing destinations. Rich metadata is bounded to 1 GiB
decoded and still allocated by mapped readers.

### Cost and reproducibility record

These native train/save measurements include validation and compression, but
exclude external preparation. The two corpus jobs overlapped on the same host;
no cross-encoder throughput comparison or extrapolated capacity is asserted.

| Corpus | Wall time / s | Maximum resident set / bytes | Retina / bytes |
| --- | --- | --- | --- |
| Moby-Dick | 10.79 | 883,703,808 | 18,152,065 |
| WikiText | 71.17 | 3,105,243,136 | 72,270,519 |

Measured executable SHA-256:
`81b16d1e30838d016cd3680592936b3b2d8f1c47323ad50a50a58a09cbab061e`.
It remains archived as `target/validation/semantic/proteus-grounded-build`.
Later CLI-only changes report the actual migration format and remove stale v4
help text. The final executable SHA-256 is
`5de725fcdc586e6183c0f5206b680339b9d35daeca9ce26dd932e93ceaadc0ea`;
all three WikiText lexical evaluation reports are identical under that executable.
Its identity and score agreement are recorded separately in the manifest. The semantic algorithms are unchanged by those messages.

| Artifact | SHA-256 |
| --- | --- |
| Moby-Dick retina, repeated training and migrated copy | `777c1b12d164a161a06ccee01cb1ae5128b9161fbded79e19593016be70475f0` |
| WikiText retina | `f3e370656b51a1a60d658f63ff201d1250a55170ef4948025a042ba5ea8c9d2b` |

`target/validation/semantic/` retains completed annotation streams, lexical source,
parser/source identities, configuration reports, all predictions for three score
channels, all-seed comparisons, resample intervals, source/binary/model hashes,
commands, timing logs, real-parser probe outputs and environment identity in
`manifest.json`. Benchmark hashes are recorded in each native evaluation JSON.
These large local artifacts are ignored by Git; preparation and evaluation
commands are maintained in the source and guides.

The full locked Rust suite passed: 219 unit/integration tests and 95 doctests,
with 6 existing explicit skips. All 11 new semantic contract tests passed again
after query error hardening. Three Python preparation tests, scoped Rustfmt,
Black, all-feature compilation, local documentation links and whitespace checks
passed. Clippy completed with existing findings in older/vendored code; new
semantic modules and their tests have no remaining findings. No Rust GPU runtime
was exercised; actual MPS execution was confined to explicit Python preparation.

Quality gates distinguish exact numerical fixtures, empirical word metrics,
resource attribution, role/negation probes, deterministic persistence, legacy
compatibility and formatting. The remaining scientific limits are larger-domain
sense/event accuracy, rare-sense false matches, parser dependence, lexical/domain
coverage and in-memory costs. No universal semantic accuracy, calibrated
confidence or novel learning breakthrough is claimed.


## Contextual and information-content extension

Recorded 2026-10-22 against committed baseline `580020f`, without changing the
reference SOM, angular encoder, default grounded score or existing model bytes.
[The extended semantic contract](semantic.md#target-aware-contextual-comparison)
specifies the public APIs, exact equations, assumptions C1-C4, error handling and
complexity. This record separates an information-content similarity option from
a target-aware contextual classifier; neither is a calibrated confidence model.

### Native lexical measurements

`information-similarity` applies a Lin-style common-information ratio using
stored occurrence counts/sense priors plus one pseudo-occurrence per lexical
sense. It counts distinct ancestors once under multiple inheritance and derives
an immutable nonserialized cache. No coefficient was selected by a sweep.
The comparison uses the same saved v5 retinas and observations as the previous
best-sense scorer. All benchmark labels below were already inspected in prior
work; these are development diagnostics, not a newly untouched lexical holdout.

| Corpus / task | Common observations / total | Previous best-sense | Information-content |
| --- | --- | --- | --- |
| wiki / simlex | 961 / 999 | 0.4968 | 0.5376 |
| wiki / simverb | 2,380 / 3,000 | 0.4290 | 0.4797 |
| wiki / wordsim | 329 / 353 | 0.3667 | 0.3594 |
| moby / simlex | 525 / 999 | 0.4815 | 0.5093 |
| moby / simverb | 1,270 / 3,000 | 0.4430 | 0.4962 |
| moby / wordsim | 130 / 353 | 0.4707 | 0.4411 |

The maintained comparison script used all observations, 2,000 paired resamples
and seed 42. WikiText correlation-difference intervals were [0.01836, 0.06624] for
SimLex, [0.03690, 0.06480] for SimVerb and [-0.04931, 0.03252] for WordSim.
These are descriptive pair-resampling intervals; repeated words and shared
resources limit independence. WordSim declines, particularly on Moby-Dick.
The scorer therefore remains explicit rather than replacing default scores.
No universal relevance improvement or new similarity theory is claimed.

### Contextual design selection and freeze

The author WiC v1.0 package was retrieved from
[the dataset page](https://pilehvar.github.io/wic/), including its schema and
licensing README. Archive SHA-256:
`f1a2fb67d903c5b9b1180f1035d4228f7c0254e8f5f868d556235457046bd4b2`.
The local protocol was recorded before development scoring. Development data and
labels were used for selection; the author test data/gold remained sealed in
the archive until the final source/configuration and binary were frozen.

Three Python development prototypes used approximately matching tokenization:
legacy gloss MAP equality, POS/target-aware gloss MAP equality, and personalized
PageRank over the limited existing parent/similar/antonym graph. Among 527
covered development pairs, correct counts were 380, 418 and 295 respectively.
A 50/50 gloss/PageRank posterior mixture gave 350 correct. The random-walk
prototype used damping 0.85 and 40 iterations; it was not a full UKB replication,
and does not falsify graph WSD generally. It was excluded from production.
The target-aware gloss model was retained. Native tokenization changes one
legacy development outcome relative to the approximate prototype.

An initial native run rejected 29 otherwise covered inflected targets whose
surface forms were not observed in the corpus. Those observations remained
unavailable in the saved report. The interface was corrected before the test
freeze to accept an explicitly supplied target lemma, requiring a byte range
and matching the requested lexical identity. WiC supplies that lemma/POS; the
model neither invents morphology nor runs a tagger during inference. The final
native development run restores all 527 baseline-covered pairs. These changes
jointly use known POS, target exclusion and supplied lemma annotations; results
are not attributed to any one component in isolation.

`freeze.json` records hashes of all Rust sources and the evaluation script,
configuration and final binary. `test-opened.json` records the subsequent
opening of the test split. Test evaluation was performed once with no model,
threshold or parameter selection afterward. Classification is equality of the
two independently conditioned MAP sense identities, not a fitted threshold.

### Contextual outcomes and coverage

Both baseline and candidate use the same v5 WikiText retina and lexical source.
The baseline is the original context-gloss scorer without supplied POS or target
exclusion; the candidate uses the author annotations described above.

| Split | Covered / all observations | Baseline correct / covered | Target-aware correct / covered | Paired gains / losses |
| --- | --- | --- | --- | --- |
| Development | 527 / 638 | 381 / 527 (72.30%) | 418 / 527 (79.32%) | 60 / 23 |
| Frozen test | 1,195 / 1,400 | 876 / 1,195 (73.31%) | 930 / 1,195 (77.82%) | 119 / 65 |

Development has 111 unavailable observations; test has 205. Counting unavailable
predictions as errors gives development accuracy 59.72% versus 65.52% and test
accuracy 62.57% versus 66.43%. No unavailable observation is dropped from the
report or assigned a successful prediction. Repeated observations remain in
order. The complete native reports retain every label, prediction, selected
sense, unknown reason and invalid-query error.

Paired target-lemma bootstrap used 2,000 resamples and seed 42. The development
difference interval was [0.03619, 0.10300] with 488 lemma clusters; test was
[0.02211, 0.06811] with 991 clusters. These are descriptive accuracy differences,
not percentage-point values; shared lexical resources can still induce
cross-cluster dependence. This is a pair-label holdout, not resource-disjoint
contextual generalization or a comparison against published WiC systems.

### Lexical-source overlap audit

The stored lexical glosses include WordNet definitions and examples, while WiC
can use those examples as contexts. The evaluator checks normalized whole-context
substrings with word boundaries against the exact same canonical lexical source,
rejecting a different source hash. The initial permissive exploratory audit
counted 573 development pairs with overlap; the maintained word-boundary audit
counts 572. This audit has a declared case/NFD/alphanumeric normalization and
identifies literal overlap, not all indirect resource contamination.

The frozen test split has 1,227/1,400 pairs with at least one such context match:

| Source-overlap stratum | All observations | Common covered | Baseline correct | Target-aware correct |
| --- | --- | --- | --- | --- |
| Neither complete context matches | 173 | 145 | 84 | 99 |
| At least one complete context matches | 1,227 | 1,050 | 792 | 831 |
| Both complete contexts match | 733 | 625 | 552 | 571 |

The both-context row is a subset of the at-least-one row. In the neither-context
stratum, covered accuracy rises from 84/145 (57.93%) to 99/145 (68.28%); the small
sample and remaining shared knowledge limit interpretation. Absence of a literal
match does not establish resource independence. This audit is a high-impact
scientific boundary: the overall 77.82% figure must not be presented as evidence
of unseen-domain language understanding or calibrated word-sense inference.

### Correctness, compatibility and reproducibility

Seven new Rust contract tests verify independent subtree-mass calculations,
redundant DAG paths, zero/full-root information, POS separation, target removal,
caller-supplied lemma handling, UTF-8/partial-word rejection, bilateral symmetry,
uncertainty diagnostics, unavailable/error denominators, duplicate IDs, exact
mixture ranking, zero k, loaded/mapped agreement, immutable-cache serialization
and CLI argument/output wiring. Four standard-library Python tests cover target
index conversion, label shape, lemma-cluster resampling, failed report publication
and input/output alias protection. The three existing preparation tests pass.

The full locked Rust suite passed: 226 unit/integration tests and 95 doctests,
with six existing explicit skips. All-feature compilation and Clippy completed;
no new contextual/information-module or test findings remain, while pre-existing
older/vendored findings are retained. Scoped Rustfmt, Black, whitespace and local
documentation links/anchors were checked. No additional neural or Rust GPU
inference is involved in the new queries.

Final native default WikiText predictions for SimLex, SimVerb and WordSim equal
the previous saved reports exactly. Information-content predictions equal the
pre-freeze development executable's reports. The same fixed retina/model hashes
are used throughout; no retraining is needed for these explicit new query APIs.
V5 layout and training algorithm identity are unchanged, and populated caches
are absent from serialization. The host and optional preparation environment
are those recorded in the preceding implementation record; current toolchain,
source and artifact identities are captured again in the local manifest.

Final executable SHA-256:
`9bb5f9fe6334454b5e9ec69282e35cf87339406404903cff44483ff3ab24e05a`.

`target/validation/contextual/` retains the protocol, failed development
prototypes, original WiC archive/README, original split files, transformed JSONL,
source-overlap audit, all native observations, per-lemma bootstrap intervals,
lexical predictions, compatibility checks, source freeze and test-open record,
command/timing logs and `manifest.json`. Outputs are published atomically, and
the evaluator rejects an output path aliasing a source input. These local
resources remain outside source control.

Bounded findings: **high** — lexical-resource example overlap limits contextual
claims, caller POS/lemma correctness remains an assumption, and best-sense
candidate ambiguity persists. **Medium** — information-content cache construction
adds first-query work/storage and helps similarity tasks while weakening the
recorded association task. Calibration, contrastive training, richer event scope
and domain-specific contextual evaluation remain research directions.
