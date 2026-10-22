# Proteus Agent Engineering Guide

## 1. Purpose and scope

This is the root engineering contract for agents working on Proteus. Read it
in full before changing the repository. It applies to the entire tree; a deeper
`AGENTS.md` may add more specific requirements for its subtree. Re-read the
relevant sections when a task changes scope.

Proteus is a Rust Semantic Folding engine. Its principal correctness boundaries
are corpus interpretation, numerical training, fingerprint construction, query
ranking, persisted model compatibility, and reproducible evaluation. A local
change can affect all of these without changing a public function signature.

This guide defines how to work. The [README](README.md) is the user-facing
documentation entry point. Do not turn this file into a second feature catalog
or copy volatile benchmark numbers and test counts into it.

When instructions conflict, system, developer and current user instructions
take precedence. Follow explicit task scope and accepted decisions from the
conversation. Repository guidance must not introduce an unnecessary approval
step for work the user has already authorized.

## 2. Evidence and authority

Distinguish a required contract from evidence that it is implemented.

| Question | Primary source |
| --- | --- |
| What was requested? | Current user instructions and explicit scope decisions |
| How must work be performed? | This guide and applicable subtree instructions |
| What is compiled and exposed? | `Cargo.toml`, `Cargo.lock`, `src/lib.rs`, feature gates and executable behavior |
| What does a CLI option do? | Argument definitions and handler wiring in `src/main.rs` |
| What does a library call do? | Its implementation, callers and relevant tests |
| What is the intended numerical contract? | `docs/algorithm.md`, identified algorithm references and approved changes |
| What can be read or written on disk? | `src/storage/format.rs`, readers/writers and compatibility fixtures |
| What has actually been validated? | Command output, exercised cases and identified artifacts |
| What is currently advertised? | README and the maintained guide for that surface |
| What was true during an earlier experiment? | The dated/revisioned record in `docs/validation.md` |

Source code can contain a defect; tests can encode the same defect. A paper can
define an algorithm family without defining this implementation's choices.
Resolve disagreement explicitly rather than treating one convenient source as
unconditionally correct.

Do not infer correctness from a type name, comment, module's presence, successful
compilation, or a plausible neighbour list. A function named `train_fast` need
not implement a different optimizer; a field named `seed` need not control every
random stream. Trace the actual data and control flow.

Use primary references for scientific claims: the original paper, author or
publisher publication, algorithm specification, or official dependency docs.
Verify titles, versions, DOI/PMID/arXiv identifiers and the relevant content.
Never invent a citation or claim a paper proves an unmeasured property here.

If a fact cannot be established, state `unknown`, identify any assumption needed
to proceed, and give a concrete falsification probe. Do not replace missing
evidence with stronger language.

## 3. Task execution and worktree ownership

### 3.1 Start with the current state

Before editing:

1. Resolve the repository with `git rev-parse --show-toplevel`.
2. Run `git status --short --branch` and record the baseline commit.
3. Inspect staged paths and the existing diffs for intended edit locations.
4. Locate applicable instructions and read the owning source and guides.
5. Translate the task into concrete acceptance criteria.
6. Record which paths this task owns and which changes already existed.

Use `rg` and `rg --files` for initial discovery. Narrow searches by subsystem.
Truncated output is not evidence that a symbol, instruction or file is absent;
repeat the query with a smaller scope. Read complete relevant files rather than
implementing from an isolated search hit.

### 3.2 Preserve concurrent work

Assume the checkout may be shared with a human or another agent. Pre-existing
tracked edits, untracked files and index entries are not yours to discard.

- Never obtain a clean tree with broad reset, checkout, clean or stash commands.
- Do not overwrite, format, rename or stage unrelated paths.
- Recheck overlapping files before writing them. Resolve overlapping edits
  deliberately; timestamps do not establish ownership or correctness.
- Keep experiments, downloaded data and trained models separate from source.
- A test of a dirty tree validates the combined tree. Account for other edits
  or use an isolated revision when claiming evidence for a particular commit.
- Record the revision used for documentation or compatibility work. If the user
  selects a committed baseline, unfinished working-tree features do not become
  documented stable behavior merely because their files exist.

Build outputs are shared mutable state too. Do not overwrite a binary another
task is measuring and then attribute its results to the earlier build. Use an
isolated build directory or identify and hash the binary actually exercised.

### 3.3 Staging and publication

Stage, commit or push only when the user has authorized that action. A request
to stage the files from a task means the explicit owned path set for that task,
not every dirty file in the checkout.

For staging:

1. Inspect the index before changing it.
2. Use `git add -- <explicit paths>`; never `git add .` or `git add -A`.
3. Preserve pre-existing staged entries.
4. Compare the index with the expected initial entries plus owned paths.
5. Run `git diff --cached --check` and inspect the staged diff.
6. Report what was staged and what remains untouched.

Do not infer permission to commit, rewrite history or publish from staging.
If a path is ignored, establish why before considering an exact-path force add.
Do not edit or suppress license text merely to silence a whitespace check.

### 3.4 Complete the authorized work

For a review, explanation or diagnosis, inspect and report without making an
unrequested implementation change. For an implementation request, complete the
code, relevant tests, documentation and reviewable result; a plan alone is not
completion.

Make reversible, evidence-backed decisions within established scope. Ask for
missing information only when it materially selects a different contract,
destructive action or incompatible result. Continue independent work while
waiting when possible. Do not treat elapsed time as required approval.

## 4. Assumptions, scope and scientific rigor

### 4.1 Assumption Register

Maintain an Assumption Register for nontrivial numerical, architectural,
compatibility or empirical work. Keep it with the design or evidence rather
than adding a new permanent report for every small change.

| Field | Required content |
| --- | --- |
| ID | Stable identifier, such as `A1` |
| Assumption | Precise statement being treated as true |
| Basis | Evidence or reason the assumption is necessary |
| Dependent results | Decisions, measurements or conclusions relying on it |
| Stress test | Boundary or adversarial case |
| Falsification probe | Concrete observation, fixture or command that could disprove it |
| Status | Confirmed for the stated scope, retained, revised or falsified |

Do not manufacture assumptions to fill a table. If none materially affect the
result, say `None`. Revisit retained assumptions after implementation and testing.
Existing scientific assumptions and their probes are in
[the algorithm contract](docs/algorithm.md#assumption-register).

### 4.2 Bounded expansion

Record relevant adjacent findings without silently expanding implementation.

| Impact | Meaning |
| --- | --- |
| High | Can invalidate the requested result, numerical contract, data integrity or compatibility |
| Medium | Material quality, performance, portability or maintenance consequence |
| Low | Local cleanup or optional improvement |

For each finding, identify evidence, dependency and whether it blocks the task.
Implement an adjacent change only when necessary within the authorized scope.
Research opportunities belong in [the roadmap](docs/roadmap.md), labeled as
proposals with validation criteria rather than advertised capabilities.

### 4.3 Quantities and claims

- Distinguish tokens, bytes, Unicode scalar values, sentences, snippets,
  vocabulary entries, updates, epochs, grid cells and active bits.
- Use SI units for time and rates. State whether memory is bytes, kB/MB/GB or
  KiB/MiB/GiB; do not silently mix decimal and binary units.
- Identify dimensionless quantities such as cosine, density and learning rate.
  Express grid radii in cells and sentence-block sizes in sentences.
- Show dimensions and rounding when estimating storage or computational cost.
  Report significant figures and uncertainty appropriate to the measurement.
- State time and space complexity for a changed algorithm when it affects
  resource use; include preprocessing, final assignment and index construction.
- Distinguish implementation correctness, determinism, retrieval quality and
  performance. None establishes the other three automatically.
- Keep discussion technical and evidence-based. Do not add ethical judgments;
  if a requested answer requires one, follow the user's `ETHOUT` instruction
  subject to higher-priority instructions.

## 5. Repository facts to verify before relying on them

The committed baseline uses:

- Package and library name `proteus`, Rust edition 2021.
- A CLI binary in `src/main.rs` and public exports in `src/lib.rs`.
- Nightly Rust because of `#![feature(portable_simd)]` and the vendored bitmap
  implementation. Do not advertise the old Rust 1.70 requirement.
- An empty default feature set and an optional `gpu` feature backed by wgpu.
- ONNX Runtime and Hugging Face dependencies for neural sentence segmentation;
  ordinary corpus training does not invoke that model.
- A committed `Cargo.lock` and release settings including LTO, one codegen unit,
  optimization level 3 and `panic = "abort"`.
- Standard Cargo discovery for integration files under `tests/`; there is no
  `autotests = false` registry to copy from another project.

Verify these statements when changing their owners. Do not add a toolchain pin,
change default features, update dependencies or alter panic policy incidentally.
Inspect tracked workflow/configuration files before describing CI coverage. A
CI system or host matrix must not be invented from RAX or another repository.

Do not treat local corpora, `fe/`, model files, experimental scripts or an
untracked test as a supported subsystem without verifying their status and
including them in the task's accepted scope.

## 6. Architecture and source ownership

The principal data flow is:

```text
CLI / library configuration
          |
UTF-8 documents -> normalization -> bounded snippets -> feature vectors
                                                        |
                                                      SOM
                                                        |
                                               final assignments
                                                        |
                                              word fingerprints
                                                        |
                       persistence <-> Retina / mapped reader -> queries

text -> sentence splitter -> semantic segmentation -> fingerprint images
SDRs / vectors -> independent ANN, HDC, batch-SOM and GPU helper APIs
```

This is a responsibility map, not proof that every helper is used by the CLI.
Trace actual callers and feature gates for the requested behavior.

| Owner | Responsibility |
| --- | --- |
| `src/main.rs`, `src/cli/quality.rs` | CLI definitions, configuration wiring, quality commands and image rendering |
| `src/lib.rs` | Public module visibility and compatibility re-exports |
| `src/config.rs` | Shared SOM, text, fingerprint and legacy storage configuration |
| `src/training.rs`, `src/training/corpus.rs` | Shared corpus input, configuration and map training report |
| `src/training/features.rs`, `src/training/spectral.rs` | Feature construction and optional subspace projection |
| `src/training/folding.rs`, `src/training/angular.rs` | Context-incidence and explicitly non-spatial angular fingerprint construction |
| `src/evaluation.rs` | Labelled pair evaluation and coverage |
| `src/som/training.rs` | Reference online optimizer, schedules and compatibility adapters |
| `src/som/map.rs`, `src/som/neuron.rs` | Map representation, geometry and neuron weights |
| `src/som/simd.rs` | Distance, normalization and update kernels |
| `src/som/batch.rs` | Independent batch trainer and growing-map interfaces |
| `src/text/` | Tokenization, normalization and neural sentence wrapper |
| `src/wtpsplit/` | Vendored sentence-model implementation and loading |
| `src/fingerprint/` | SDR algebra, word/text aggregation and HDC |
| `src/similarity/` | Similarity measures and shared score interfaces |
| `src/index/` | Postings, mapped postings, ANN and spatial lookup |
| `src/storage/retina.rs` | Loaded model, query behavior and metadata access |
| `src/semantic/mod.rs`, `src/semantic/training.rs` | Grounded config/report, validation, corpus sense adaptation and observed lexical features |
| `src/semantic/corpus.rs`, `src/semantic/lexicon.rs` | Complete attributed annotations, resource schemas and bounded graph validation |
| `src/semantic/relations.rs`, `src/semantic/evidence.rs` | Typed word scores, contextual senses, grammatical role binding and evidence states |
| `src/cli/semantic.rs`, `scripts/prepare_semantic.py` | Native grounded commands and explicit external-resource preparation |
| `src/storage/format.rs` | Serialized layout, validation, writers and mapped retina |
| `src/segmentation/` | Regex splitting and SDR-based topic segmentation |
| `src/accel/` | Optional GPU helpers and non-GPU stub |
| `src/roaring/` | Vendored bitmap implementation |
| `tests/` | Numerical, process, integration and compatibility checks |
| `scripts/` | Reproducible evaluation and other explicitly maintained tools |

Keep helpers with their owning subsystem. If a file is split into sibling
modules, preserve public paths, update this map and
[the architecture guide](docs/architecture.md#source-ownership), and inspect
source-reading tests and documentation links before moving code.

## 7. Change-surface analysis and routing

Before a behavioral change, mark relevant surfaces as affected, unaffected with
a reason, or unknown. Do not require an elaborate report for a typo fix.

| Change | Inspect together |
| --- | --- |
| Corpus boundaries or normalization | Input reader, tokenizer, word lookup, text aggregation, saved configuration and corpus digest |
| Feature geometry or projection | Feature construction, seeds/order, vector dimensions, zero-vector handling, training report and quality evidence |
| SOM update or distance kernel | Schedules, topology, initialization, SIMD tails, thread behavior, final assignment and scalar oracle |
| Fingerprint folding | Per-snippet versus token counts, density caps, rare words, tie order, persisted metadata and retrieval tests |
| Similarity or ranking | Exhaustive search, loaded index, mapped reader, CLI ordering, empty inputs and unequal cardinalities |
| New parameter or default | CLI declaration, handler wiring, library default, validation, serialization, legacy defaults and docs |
| File format | All writers/readers, migration, mapped access, malformed fixtures, bounds and version negotiation |
| New public API | Owner module, re-exports, errors, examples, tests and library guide |
| Segmentation | Regex/neural route, boundaries, block scoring, empty text, CLI flags and model loading |
| GPU or ANN helper | Feature gating, metric, fallback, resource limits, runtime evidence and claims |
| Module rename or split | Imports, visibility, source inclusions, tests, README/deep links and this ownership map |
| Evaluation tool | Input parsing, exclusions, metrics, hashes, output schema and reproducibility |

Use this map to select tests and documentation before editing. A change that
compiles in one path can still leave another reader or query route using the old
behavior. Fix all in-scope paths or report the intentional asymmetry explicitly.

## 8. Corpus and feature contracts

### 8.1 Input interpretation

The document boundary is part of the model definition. Paragraph mode joins
physical lines until a blank line; line mode treats nonempty lines as separate
records. Snippet boundaries limit context and must not accidentally cross the
selected document boundary.

For input changes, test:

- empty input, blank-only records and a final record without a newline;
- LF and CRLF, multiple blank lines and hard-wrapped paragraphs;
- I/O failure and invalid UTF-8, including failure after valid earlier records;
- snippet lengths of one, exactly the limit, and one above the limit;
- tokens filtered out before a boundary and documents yielding no usable data;
- preservation of order and deterministic normalized-corpus hashing.

Do not silently drop decoding or I/O errors and train on a successful prefix.
Differentiate extracted snippets from vectors retained after feature filtering.

### 8.2 Normalization and vocabulary

Trace normalization through both training and inference. The persisted policy
must govern loaded word queries and text queries, not just training.

- Distinguish NFD from NFC/NFKC. Changing normalization changes vocabulary keys.
- Token length limits currently count UTF-8 bytes; do not relabel them as
  characters or grapheme clusters.
- ASCII punctuation handling does not imply removal of all Unicode punctuation.
- Test apostrophes, combining marks, non-Latin text, case, numeric-only tokens
  and mixed alphanumeric tokens relevant to the change.
- Do not silently add stemming, lemmatization, stopword removal or language
  detection behind an unchanged configuration.
- Distinguish minimum token frequency from document frequency. A word excluded
  from features may still belong in the output vocabulary.

Changing preprocessing requires an inference compatibility decision and updated
corpus/algorithm provenance. File readability alone is not semantic compatibility.

### 8.3 Feature geometry

Feature construction is an explicit algorithm choice. The reference TF-IDF
formula, normalization and optional projection are specified in
[the numerical contract](docs/algorithm.md#contexts-and-features).

When adding or changing a feature model:

1. Give it a named, documented configuration value.
2. Define vocabulary/feature ordering, weighting, smoothing and normalization.
3. Define handling of empty, zero, nonfinite and extreme vectors.
4. Specify all random streams and their resolved seeds.
5. Record effective dimensions and feature choices in model provenance.
6. Define the behavior of old metadata that lacks the new field.
7. Provide a numerical fixture and controlled corpus evidence.
8. Separate feature correctness from empirical retrieval improvement.

Do not describe random indexing as learned skip-gram. Do not call a random
projection exact, or claim a distortion guarantee without establishing its
assumptions and parameters. Measure quality changes against the same baseline
corpus, split, seed and training budget.

## 9. SOM and floating-point contracts

### 9.1 Reference online path

Preserve the selected online contract unless the task explicitly changes it:

- Initialization and sample order are controlled by recorded random state.
- `iterations` counts actual online updates, not epochs or input tokens.
- Each update computes its BMU from the current weights.
- Learning-rate and radius schedules have defined first/last endpoints and a
  defined one-update case.
- The Gaussian update visits each intended neuron exactly once.
- Planar and toroidal distances have distinct, explicit geometry.
- A frozen-map pass assigns every usable vector after training.
- The progress callback's unit and final-pass behavior remain documented.

Do not impose a hidden update cap, reuse stale BMUs, or silently substitute the
batch trainer, an ANN search or GPU arithmetic to accelerate this path. An
alternative optimizer needs a separate named contract and validation.

### 9.2 Numerical validation

Use a small independent scalar implementation to test update equations and
assignments. Do not have the oracle call the production distance/update helper.
State arithmetic precision, accumulation order and comparison tolerance.

Cover dimensions one and two, asymmetric weights, ties, one update, multiple
updates, planar edges, toroidal wrap and radii larger than the map. Test invalid
grid/vector shapes, nonfinite values, schedule parameters and magnitudes that
could overflow squared distance.

When changing precision, reduction order or use of fused operations, measure the
effect rather than assuming algebraic equivalence implies identical floats.
Do not enlarge a tolerance merely to hide a regression. A fixture tolerance is
not a universal model-error bound.

### 9.3 SIMD and parallel work

Distance and update kernels may process full SIMD blocks and scalar tails.
Exercise vector lengths below, at and above their lane/stride boundaries.
Include all-zero, sparse, mixed-sign, large finite and near-tie inputs.

Parallelism must preserve the stated ordering/determinism contract. Hash-map
iteration, parallel reductions and nondeterministic tie order can leak into
features, initialization, folding and serialized bytes. Test multiple worker
counts and separate fresh processes where the claim requires it.

Do not infer cross-platform bit identity from same-host thread-count tests.
Record host architecture, toolchain, arithmetic backend and build settings.
Explicit SIMD does not establish that a particular machine instruction was
emitted or that the kernel is faster on a given host.

### 9.4 Batch and growing-map APIs

The batch trainer and growing-map types are independent library surfaces. Their
input shapes, schedules, initialization, assignment timing and metrics must be
validated under their own contracts. Do not transfer online-trainer guarantees
to them by name or proximity.

Validate empty data, zero batch sizes, epoch/restart edge cases and flat weight
dimensions before indexing or division. If an optimizer change alters the
meaning of a reported metric, update its documentation and consumers together.

## 10. Fingerprints, composition and retrieval

### 10.1 Folding and SDR invariants

Treat a fingerprint as a set of active cells in a defined grid universe.
Deduplicate positions and establish grid compatibility before binary operations.
Keep bit positions within the declared universe at public input boundaries.

The reference incidence path counts a distinct word once per assigned snippet.
Token-frequency metadata is a different quantity. Do not accidentally replace
one with the other when optimizing aggregation.

Density and absolute cardinality caps both apply. Specify `ceil`/floor behavior,
the cap's relationship to grid size, and deterministic tie-breaking. A legacy
minimum-cardinality field does not authorize invented evidence or random padding.

Any spatial smoothing, background correction or alternative folding method
must state how cells receive support, how boundaries wrap, which parameters are
persisted, and how it differs from raw incidence. Changing it requires the
feature/algorithm documentation and compatibility analysis in this guide.

### 10.2 Text and semantic operations

Keep union, weighted composition, IDF-weighted composition, strict-majority
center, set difference and signed analogy distinct. Inspect their actual
implementations rather than using the word "semantic" as a common contract.

- Unknown word lookup and unknown text tokens have different error behavior.
- All-unknown text must have defined empty-fingerprint behavior.
- State whether unknown words participate in an aggregation denominator.
- Signed `B + C - A` arithmetic must retain negative contributions until the
  relevant score is computed; set subtraction is a different operation.
- HDC item binding/sequence encoding is not corpus-trained linguistic knowledge.

An arithmetic or set-operation test does not prove synonym, analogy, translation
or topic quality. Use labeled held-out evaluation for those claims.

### 10.3 Scores and top-k ranking

Identify whether an API returns a similarity, a distance, raw overlap count or
a composite spatial score. Higher-is-better and lower-is-better conventions
must be explicit at boundaries and in examples.

For binary cosine, cardinalities matter. Pruning candidates by raw overlap can
discard the true cosine winner. Compare optimized/indexed results with a
separate exhaustive reference for unequal-size fingerprints.

Test zero and oversized `k`, empty queries, absent words, ties, zero overlap,
duplicate candidates and query-word exclusion. If the contract includes
zero-overlap results to fill top-k, an index cannot silently omit them.

Tie-breaking must be stable and documented. Check ordinary `Retina`, indexed
search, mapped search and CLI presentation together. Do not add hidden query
filters or hash-order sampling to improve a few visible examples.

### 10.4 ANN and spatial search

HNSW, LSH and product quantization have their own representations, metrics and
candidate limits. They are not automatically the implementation of ordinary
retina queries. Preserve external ID mappings and document ownership/lifetimes.

For changes, measure recall against exhaustive search using the same metric.
Report construction settings, query breadth, memory, seed and candidate count.
An HNSW Jaccard result is not an exact cosine result. A PQ byte code is not a
serialized bitmap or `.retina` file. LSH candidates still need caller ranking.

Spatial lookup additionally needs correct grid/radius semantics, boundary
handling, kernel normalization and interpretable factor output. Test wraparound,
out-of-range positions, repeated positions and low-density regions.

Do not repeat generic O(log n), fixed compression-ratio or fixed-speedup claims
as unconditional properties of this implementation and workload.

## 11. Persistence, compatibility and unsafe code

### 11.1 Serialization is a public contract

Read all affected writers and readers before changing layout or metadata.
The compatibility matrix includes loaded reads, mapped reads, migration and
save/reload; test each path that accepts the format.

For a format or metadata change:

1. Specify header/version/flag interpretation and section layout.
2. Check arithmetic before converting offsets, lengths and grid sizes.
3. Define required fields, optional fields and defaults for absent old fields.
4. Preserve known metadata through migration and round trips.
5. Distinguish unavailable provenance from known zero or an empty result.
6. Define old-writer/new-reader and new-writer/old-reader behavior.
7. Add compatibility fixtures and malformed-input checks.
8. Update the CLI, architecture, library and algorithm documentation as needed.

Grounded retinas use v5 compressed metadata (decoded length + zlib), with an
explicit 1 GiB decoded bound and required semantic model. Nonsemantic writers
remain v4. Loaded/mapped/migration paths preserve graph scores, original sources
and annotations. Rich mapped metadata still allocates; only bitmap storage is
mapped. `tests/semantic_contract.rs` covers these boundaries, and
`tests/test_prepare_semantic.py` covers preparation offsets/hashes/publication.
See [the grounded contract](docs/semantic.md) before changing extraction, scoring
or resource schemas. Grounded exact word ranking uses the graph; legacy text
pooling and bitmap APIs remain distinct. Never describe a lexical-resource gain
as a corpus-only learning result.

The v4 family coexists with upstream legacy index layouts and an earlier local
metadata-only v2 layout. Version numbers alone do not disambiguate every legacy
file; inspect flags and offsets. Do not remove a reader merely because current
writers no longer produce its layout.

File-format version, algorithm identifier and metadata-schema evolution are
related but distinct. Changing training behavior does not always require a new
binary layout; retaining the layout does not justify mislabeling the algorithm.

### 11.2 Validate before exposing access

Exercise truncation at section boundaries, invalid magic/version/flags,
overflowing offsets, overlapping sections, invalid UTF-8, duplicate keys,
malformed bitmaps, out-of-universe bits and postings inconsistent with words.

Public loaders must reject malformed data with useful errors rather than
panicking, reading unchecked slices or returning a partially trusted index.
An mmap path is not exempt from validation because the OS mapped the file.

Opening can perform substantial validation and allocate metadata. Do not
describe memory mapping as zero-copy for decoded values or constant-time open
unless measured behavior and implementation actually support the claim.

### 11.3 Atomic writes and lifetime

Preserve the existing destination if validation or serialization fails. Write a
temporary file on the destination filesystem before replacement; handle errors
and temporary-file cleanup. Do not call a rename-based update crash-durable
without establishing the necessary file and directory synchronization behavior.

A mapped file's backing storage must not be modified in place while borrowed
data can be accessed. Replacement and in-place mutation are different lifetime
cases. Keep the safety argument next to the mapping operation.

Do not turn training reports into fabricated history when loading legacy data.
Rewriting a file may add an index; it cannot recover missing corpus provenance
or repair already-learned geometry.

### 11.4 Unsafe and concurrent code

Each new or changed `unsafe` block needs a local `SAFETY:` explanation covering
pointer provenance, lifetime, alignment, initialization, bounds, aliasing,
mutability and synchronization as applicable. A comment saying "safe" is not
an argument. Test malformed and boundary cases through the public path.

For shared state, identify lock order and ownership. Do not hold locks across
callbacks, model loading or blocking I/O without an explicit contract. Keep
progress/statistics synchronization separate from algorithm ordering.

The release panic policy differs from recoverable `Result` errors. Do not add
`unwrap`, `expect` or unchecked indexing to corpus/file/configuration paths on
the assumption that callers can catch a panic.

## 12. Segmentation, visualization and optional backends

### 12.1 Sentence and topic boundaries

Keep these three concepts separate:

| Boundary | Owner | Meaning |
| --- | --- | --- |
| Training snippet | Corpus trainer | Bounded normalized token context within a document |
| Sentence | Regex or neural splitter | Input to sentence-oriented processing |
| Topic segment | Semantic segmenter | Grouping selected by block similarity and depth scores |

Changing one must not silently change the others. Fast mode must not initialize
or download a neural model. Neural-mode evidence must identify the actual model,
runtime and settings used, not just the wrapper type.

Test empty text, one sentence, fewer sentences than a block, final partial
blocks, zero/flat scores, minimum segment sizes and boundary indexing. Check
that segment text and half-open sentence ranges agree. Quality comparisons need
labeled boundaries; a successful split is only execution evidence.

### 12.2 PNG output

Check row/column mapping, zero-based positions, scale, grid dimensions and output
size arithmetic. Validate invalid or excessive dimensions before allocation.
Test text/file/stdin selection, output naming and combined word/segment modes.
Generated filenames must not turn token text into unintended paths.

Keep image generation deterministic for a fixed model and input when claiming
that property. Verify the generated PNG dimensions and active-cell orientation;
an existing file alone does not prove that rendering is correct.

### 12.3 GPU and model-loading boundaries

The `gpu` feature exposes helpers, not proof that CLI training ran on a GPU.
The no-feature stub, feature-enabled build, adapter initialization and actual
kernel execution are separate checks.

For GPU changes, test shapes, upload/download layout, adapter limits, failure
paths, synchronization and numerical comparison with CPU output. Include the
cost of transfers and setup in end-to-end performance claims.

Do not make ordinary training or queries download neural resources as an
incidental dependency of a refactor. Model names, cache behavior, network needs
and runtime prerequisites belong in user documentation when changed.

## 13. Public API, CLI and configuration changes

Trace every new or changed parameter through:

```text
CLI spelling / default -> parsed value -> handler -> validated configuration
        -> algorithm -> effective report -> serialization -> reload behavior
```

Library defaults and CLI defaults may intentionally differ. Document the
difference; do not assume one is generated from the other. An exposed option
that is accepted but unused is a defect, not a completed interface.

For public APIs, preserve canonical modules and existing re-exports unless a
break is requested. Define ownership, mutation, errors, empty inputs and shape
constraints. Avoid exposing an internal helper just to simplify an example.

For CLI changes, test invalid values, conflicting inputs, exit status, stderr
versus stdout, and whether failed operations publish a destination file. Retain
usable `--help` text and verify examples against the actual executable/parser.

When a default changes, update current documentation and compatibility rules in
the same change. Do not reinterpret absent old metadata as a new experimental
default unless that migration is deliberate and tested.

## 14. Code organization and style

### 14.1 Placement and size

- Add behavior to its semantic owner rather than the first convenient file.
- Keep orchestration separate from numerical kernels and persistence parsing.
- Prefer small private helpers with named invariants over generic frameworks
  introduced for a single use.
- Preserve public paths and existing module privacy when splitting files.
- About 1,500 lines is a review trigger for handwritten source; 2,000 lines or
  150 kB calls for a scoped split before adding another unrelated subsystem.
- Generated/vendored material and this engineering guide are not reasons for
  mechanical line-count refactors. Split only an authorized semantic scope.

Inspect `include!`, `include_str!`, path attributes, doctests and source links
when moving files. A successful compilation does not detect every stale link.

### 14.2 Rust

Use the owning crate's edition and formatter conventions. Keep statements,
validation branches and nested loops readable on separate lines. Do not use
`#[rustfmt::skip]` to preserve compressed handwritten algorithms.

Choose names that distinguish token counts, snippet incidence, vector dimension,
grid side, total cells and active bits. Use `usize` for host indexing and checked
conversion to fixed-width serialized fields. Make saturation, truncation and
overflow policy explicit where they change results.

Follow existing imports and error types. Keep helpers private or `pub(super)`
unless wider use is required. Use `?` for normal errors and early returns for
invalid inputs. Assertions belong to established internal invariants or tests,
not unvalidated file contents or command-line values.

Comments explain equations, ordering, invariants and necessary deviations.
Public/module documentation describes real behavior and failure modes. Do not
copy speculative speed claims from old comments into new documentation.

### 14.3 Formatting without collateral edits

Run a non-mutating format check first. If broad formatting would change
unowned files, format only owned files with the correct edition and, where
needed, `--config skip_children=true`. Review the resulting diff immediately.

Do not combine a semantic change with a repository-wide vendored-code restyle.
Existing warnings and formatting debt are not permission to alter every file.

### 14.4 Python, shell and manifests

Python tools use explicit input schemas, useful errors and argument-list
subprocess calls. Separate CLI parsing from reusable evaluation logic. Do not
silently drop failed queries, unknown words or invalid rows from reported metrics.

Shell snippets must state their interpreter, quote paths and preserve argument
boundaries. Do not interpolate user text into executable shell syntax.

Preserve TOML structure and review every lockfile change. Do not run broad
dependency updates during documentation or unrelated numerical work. Use the
committed lock when claiming a reproducible build.

## 15. Tests and independent evidence

### 15.1 Select meaningful tests

Behavior fixes need regression evidence showing the earlier failure and the
corrected result. New numerical/API behavior needs tests at the relevant public
boundary. A mechanical refactor needs existing coverage that reaches the moved
behavior, not tests that merely duplicate the new implementation.

Documentation-only changes need link, API/example and command checks appropriate
to the edits. Do not run an unrelated full numerical or corpus benchmark solely
because a Markdown file changed.

Test-name filters are useful during development. Finish a behavioral change by
running the complete affected integration target plus other affected surfaces.
Read `running N tests`, failures, ignored cases and skip diagnostics. Zero tests
with exit code zero is not a regression pass.

### 15.2 Evidence layers

| Layer | Establishes | Does not establish |
| --- | --- | --- |
| Independent scalar oracle | Numerical agreement for specified equations and fixtures | Corpus semantics or universal error bounds |
| Controlled synthetic corpus | Defined topic/support behavior under controlled inputs | General-language synonym quality |
| Exhaustive-vs-indexed comparison | Ranking equivalence under the same metric and candidates | Quality of the learned representation |
| Save/load/mmap comparison | Persistence and reader agreement for exercised layouts | All historical-file compatibility |
| Byte-identical retraining/reloads | Reproducibility under the recorded environment | Cross-platform floating-point identity |
| Held-out labeled evaluation | Specified task quality on that dataset/split | Universal accuracy or causality across confounded changes |
| Release timing/memory measurement | Resource behavior for the recorded workload | Correctness or improvement on unrelated hardware |

Two production paths sharing a helper can share a defect. When testing that
helper's correctness, use an independent calculation or invariant.

### 15.3 Target map

The established committed integration targets are:

| Target | Primary responsibility |
| --- | --- |
| `training_contract` | Numerical oracle, training/query determinism, boundaries and malformed data |
| `rebase_compat` | Legacy formats, metadata and mapped ranking |
| `cli` | Process-level configuration, output files, queries and failures |
| `integration_tests` | End-to-end library composition |
| Library unit tests | Component algorithms, bitmaps, indexing and segmentation helpers |
| Rust documentation tests | Public examples and usage contracts |

Re-discover targets from the current committed manifest/tree when this changes.
Nested files do not become independent Cargo targets by location alone; verify
module inclusion and `--list` output. An untracked pending test is not established
release coverage until the accepted change includes it.

## 16. Validation commands and resource discipline

Use nightly and the existing lock for the current package. Start with targeted
checks and expand according to the change-surface map.

```sh
git diff --check
cargo +nightly fmt --all --check
cargo +nightly test --locked --lib
cargo +nightly test --locked --test training_contract
cargo +nightly test --locked --test rebase_compat
cargo +nightly test --locked --test cli
cargo +nightly test --locked --test integration_tests
cargo +nightly test --locked --doc
cargo +nightly check --locked --all-features
```

These are a selection map, not an instruction to repeat every check after each
edit. `cargo +nightly test --locked` covers the normal default-feature suite;
separate invocations can isolate ownership and diagnostics.

For changes affecting feature-gated code or public documentation, also select
the relevant all-feature build/doc checks. Clippy is useful for changed Rust
surfaces; do not claim the existing codebase is warning-free.

Use release builds for performance and debug-checked builds for overflow and
invariant diagnostics. Cross-compilation, native execution and GPU execution
are different evidence. Report the actual host and enabled features.

Inspect ignored tests before running them. They may require model downloads,
large resources or a distinct runtime. Do not blanket-enable ignored tests and
then call an environmental failure a numerical regression.

Use a small controlled corpus for CLI smoke checks. Avoid overwriting user
models or training on all local datasets. Put generated outputs in a task-owned
location under `target/` unless another destination is explicitly requested.
Delete only task-owned scratch artifacts when cleaning up.

## 17. Performance, retrieval quality and provenance

### 17.1 Reproducible experiment record

Record enough information to reproduce a material result:

- source revision and relevant dirty changes;
- executable hash, build profile, features and compiler version;
- host CPU/architecture, OS, memory and thread count;
- raw corpus identity and normalized/preprocessed identity;
- document/snippet rules, vocabulary filtering and effective dimensions;
- feature/folding model, seed, topology, update count and schedules;
- model hash, stored algorithm identifier and resolved configuration;
- query/evaluation set, exclusions and scoring/ranking rules;
- wall time, peak-memory method, repetitions and variability;
- exact commands and output artifact locations.

The maintained [evaluation script](scripts/evaluate_corpus.py) records several
of these fields. Inspect its current scope before relying on it; it is not a
substitute for every measurement or a labeled benchmark.

### 17.2 Comparisons

Change one factor at a time when attributing improvement. Different grids,
update counts, feature dimensions or corpora confound a projection comparison.
If a multifactor experiment is deliberate, label it accordingly.

Define metrics and denominators. Report unknown/OOV queries and failed cases;
do not improve an accuracy figure by silently removing them. Keep development
queries separate from held-out evaluation and report meaningful variation
across seeds where stochastic learning matters.

A few plausible nearest neighbours are an illustration, not a synonym-quality
benchmark. Quantization error is a map-fit diagnostic, not a language-accuracy
score. Analogy arithmetic and HDC identities require separate task evaluation.

### 17.3 Cost accounting

For dense reference training, use the variables and bounds in
[the cost model](docs/algorithm.md#cost-and-bounded-extensions). Include feature
construction, projection storage, map working copies, final assignment,
folding, indexing and serialization when they affect the claimed improvement.

Separate data loading, training, model loading and query time. State whether
filesystem caches, model caches, allocation and GPU transfers are included.
Report distributions or repeated measurements rather than only the best run.
Do not extrapolate a small-corpus timing into a terabyte-capacity guarantee.

### 17.4 External and generated material

Do not add large corpora, pretrained models, downloaded runtime packages or
generated images to source control incidentally. Preserve user-provided inputs.
For intentionally checked-in fixtures, document origin, transformation,
licensing and expected size; use a minimal fixture when possible.

Record canonical references and retrieval/version information for external
documents used in implementation. Store intentionally retained references
under an appropriate `docs/` subpath, with redistribution rights respected.
Do not duplicate large external documents when a verified durable citation
suffices. Keep original quoted material distinguishable from derived prose.

## 18. Documentation structure: one entry point

The restored structure is a maintained engineering surface, not optional prose.
Every change affecting advertised or public behavior must assess its documentation
impact before completion.

```text
README.md                         user entry point and concise current overview
AGENTS.md                         agent engineering and maintenance contract
docs/
  cli.md                          build, commands, options, troubleshooting
  library.md                      embedding, core API and configuration
  advanced.md                     independent ANN, batch-SOM and HDC APIs
  architecture.md                 data flow, ownership, persistence overview
  algorithm.md                    numerical contract and scientific assumptions
  development.md                  tests, evaluation and contribution workflow
  validation.md                   revisioned records of exercised evidence
  quality.md                      alternative encoders, diagnostics and evaluation contracts
  theory.md                       historical semantic theory review and redesign rationale
  semantic.md                     implemented grounded senses, typed relations, evidence and resources
  roadmap.md                      unimplemented directions and research probes
```

### 18.1 README contract

- `README.md` is the single complete user documentation entry point.
- Keep its introduction, working minimal example, capability summary, important
  limits and grouped links to maintained guides.
- Keep detailed option tables, long API examples and equations in their owners.
- Link directly to the relevant subsection when the destination is specific.
- Do not create `docs/README.md` as a competing front page.
- Do not replace a broad feature map with a narrative about the latest fix.
- When a new maintained guide is added, make it reachable from the README.
- Roughly 100-200 lines is a useful size target, not permission to delete useful
  coverage to satisfy a line count. Restructure with links when needed.

### 18.2 Focused-page contract

Each maintained guide must link back to `../README.md#documentation` and link
to related owners when useful. Prefer one authoritative explanation per topic
with links from other pages, rather than duplicated tables that drift.

Keep headings stable because anchors are interfaces. If a heading/path changes,
update all inbound links in README, guides, source comments and this file.
Avoid machine-specific absolute paths and volatile source-line numbers in
durable Markdown. Relative source/test links and stable section anchors are
preferred.

Use tables for exact mappings and comparisons. Use ASCII diagrams for data
flow or hierarchy where helpful. Keep terminology, symbols and metric direction
consistent between prose, equations, code and CLI output.

### 18.3 Document ownership and update triggers

| Change trigger | Required documentation review |
| --- | --- |
| Build/toolchain/dependency/feature requirements | README prerequisites; `docs/cli.md#build`; development commands; repository facts here |
| CLI command, option, default, alias or precedence | `docs/cli.md`; any README quick-start command; related library/configuration explanation |
| Public Rust type, method, visibility or error | Rustdoc; `docs/library.md`; advanced guide when applicable; re-export/source links |
| Corpus parsing, normalization, feature weighting or projection | `docs/algorithm.md`; CLI/configuration tables; library guide; relevant assumptions |
| Optimizer, schedule, topology or final assignment | Numerical contract; model provenance; cost model; validation evidence |
| Fingerprint support, caps, composition or analogy | Algorithm and library guides; tests; compatibility/default interpretation |
| Search metric, candidate pruning or tie-breaking | Algorithm contract; CLI/library output description; advanced index boundaries |
| Format version, metadata or reader support | `docs/architecture.md#format-and-migration`; library persistence; CLI migration; compatibility evidence |
| Segmentation/model loading or PNG behavior | CLI segmentation/images; library segmentation; optional-runtime requirements |
| New ANN/HDC/batch/GPU interface | Advanced guide or its new focused owner; feature/build requirements; README capability link |
| Module ownership, source path or test layout | Architecture map; development test map; this guide's routing; every affected deep link |
| New measurement or quality claim | Revisioned validation record with hashes/method; update current summaries only when evidence supports them |
| Proposed future capability | Roadmap with prerequisites and falsification probes; do not advertise as implemented |

The table is an impact checklist, not permission to touch every document for
every edit. If no user-visible or contractual documentation changes, record the
reason in the final report or change description.

### 18.4 Mandatory update workflow

For each relevant code change:

1. Identify affected rows in the table before implementation.
2. Read the existing authoritative guide and search for duplicate claims,
   examples, defaults and references across README, docs and Rustdoc.
3. Update behavior and its owning documentation in the same reviewable change.
4. Preserve coverage for unaffected capabilities and explain deliberate removals.
5. Update inbound links, examples and terminology in related pages.
6. Verify public defaults against both declarations and actual handler wiring.
7. Validate links/anchors and check code examples or command parsing as relevant.
8. Record the revision/build and scope of any new empirical evidence.
9. Re-read the resulting README as a new user and the detailed owner as an
   implementer; resolve conflicting descriptions before delivery.

An implementation is not complete while its documented parameter meaning,
algorithm, format or compatibility behavior describes the previous code.
Do not defer required documentation to an unspecified follow-up.

### 18.5 Preserve and condense instead of discarding

Do not delete the project's existing documentation simply because a focused
subsystem was rewritten. Before replacing or splitting a page, inventory its
topics and map each to a retained section, a new deep link, or an explicit
obsolete-behavior removal.

Preserve useful usage examples, API discovery, architecture, limitations,
development guidance, attribution and research context. Correct stale signatures
and claims as they move. Remove repetition and unsupported promises, not entire
unrelated capabilities.

Do not restore old examples verbatim without checking them. In particular,
constructor argument order, direct LSH construction, batch-training shapes,
analogy APIs, model defaults and file-format descriptions have changed before.

### 18.6 Current behavior, evidence and proposals

Keep these classes visibly distinct:

| Class | Owner and wording |
| --- | --- |
| Implemented contract | Current guides and Rustdoc; cite implementation/contract and state limits |
| Exercised result | Validation record; identify revision, binary, environment, inputs and executed cases |
| Historical result | Preserve original provenance; do not relabel it as a measurement of a later binary |
| Hypothesis/proposal | Roadmap or explicitly marked design section; state assumptions and success criteria |
| Unknown | Say `unknown`; identify missing evidence and a probe |

`docs/validation.md` is a historical evidence record. Preserve prior experiment
identity when adding results; do not overwrite a binary hash or test count with
a new one while retaining the old timings. Update current summaries without
rewriting history.

No general-language quality, proprietary-model equivalence, cross-lingual
alignment, fixed speedup or large-corpus capacity claim follows merely from
implemented mathematics or a successful synthetic test.

### 18.7 Documentation validation

At minimum for structural changes:

- Resolve every relative file/directory link and section anchor.
- Verify every maintained guide is reachable from the README and links back.
- Check Markdown fences, table escaping and whitespace.
- Compile runnable Rust examples against the intended revision and feature set.
- Use `no_run` when execution needs external models/files; compilation still
  matters. Use `ignore` only with a concrete reason, not to hide broken APIs.
- Distinguish compile-only examples from examples actually executed.
- Validate shell flags and positional arguments against real parser wiring.
- Keep sample outputs marked as examples unless generated from a named run.
- Do not add a link to a nonexistent license, fixture, command or planned page.

For an explicitly selected committed baseline, validate in an isolated snapshot
when concurrent source changes would alter the interface. Reuse its lockfile
and do not let a temporary example crate silently resolve a new dependency graph.
Do not publish temporary review/build paths as permanent documentation links.

### 18.8 Updating this guide

Update `AGENTS.md` when module ownership, build requirements, test discovery,
format responsibilities or documentation structure changes. Keep examples
Proteus-specific; do not import another repository's CI targets, ABI, features
or tools without checking they exist here.

Avoid embedding current test totals, transient commit IDs, uncommitted feature
names or benchmark numbers. Those belong in evidence records. Operational rules
should survive normal feature growth and remain proportional to the task.

## 19. Debugging and reproducibility

Minimize the failure before changing broad behavior. Record whether it appears
in input extraction, feature construction, training, folding, save/load or
ranking; compare intermediate values at the first divergence.

Useful comparisons include:

- same corpus under paragraph and line boundaries;
- exact features versus explicitly configured projection;
- one thread versus several threads with the same seed;
- in-memory model versus save/reload versus mmap;
- indexed versus exhaustive ranking on the same fingerprints;
- repeated fresh-process queries of the same model;
- scalar versus SIMD kernel output on boundary dimensions.

For nondeterminism, inspect hash iteration, parallel reductions, RNG streams,
tie order, unstable sorting, unrecorded defaults and process-global state before
increasing retries. A retry can characterize a problem; it cannot erase a failed
correctness check.

Do not infer a model defect from an unknown word until normalization, minimum
count, feature filtering and serialized metadata have been inspected. Conversely,
do not explain away a ranking defect as corpus quality before comparing scores
on controlled fingerprints.

## 20. Final review and quality gates

### 20.1 Self-red-team questions

- Did the input boundary or token interpretation change unintentionally?
- Does every accepted parameter reach the intended algorithm and saved report?
- Can nonfinite values or integer overflow reach allocation, indexing or scoring?
- Does a different thread count or hash order change results unexpectedly?
- Are the final assignments based on the final map?
- Can raw-overlap pruning change cosine top-k under unequal cardinalities?
- Do loaded, indexed and mapped paths agree on the promised behavior?
- Can an old model silently acquire new defaults or invented provenance?
- Does a failed write preserve the destination?
- Is an approximate or optional path being called exact or validated by inference?
- Could a test pass by running zero cases, sharing the same defect or skipping?
- Are OOV/excluded examples hidden in quality metrics?
- Did a current guide or deep link become stale?
- Did formatting, generation, staging or builds disturb concurrent work?

### 20.2 Quality gates

Before claiming completion, establish each applicable gate. An unrun or blocked
check is a reported limit, not a silent pass.

| Gate | Acceptance |
| --- | --- |
| QG1 | The task is addressed technically without an added ethical judgment |
| QG2 | Material assumptions, dependent results and falsification probes are recorded and reconciled |
| QG3 | Every requested behavior maps to an implemented/reviewed result and appropriate evidence |
| QG4 | Units, calculations, precision, complexity and reproduction commands are consistent |
| QG5 | Relevant contradictions, edge cases and compatibility boundaries are resolved or explicitly reported as incomplete |
| QG6 | Claims have verified source or empirical provenance; unknowns remain labeled |
| QG7 | Adjacent opportunities/risks are impact-labeled and kept within authorized scope |

Also verify worktree ownership, exact staging when requested, and documentation
consistency. Do not call a requested implementation complete with an unresolved
behavioral defect or stale required guide.

### 20.3 Delivery

Lead with the outcome. State the changed behavior and owned files, the checks
actually run, meaningful limits/skips, assumption status and relevant adjacent
findings. Mention preserved concurrent work when it affects interpretation.

Keep the final response proportional to the task. Link to detailed artifacts
rather than pasting the full register, test log or documentation inventory into
every reply. Do not say "all tests passed" unless the stated full scope actually
ran and passed; do not imply a commit, push or publication that was not performed.
