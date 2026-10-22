# Development and validation

[Documentation index](../README.md#documentation) · [Recorded validation](validation.md)

## Local checks

```sh
cargo +nightly test
cargo +nightly test --test training_contract
cargo +nightly test --test rebase_compat
cargo +nightly test --test cli
cargo +nightly test --test integration_tests
cargo +nightly test --doc
cargo +nightly check --all-features
```

`cargo +nightly fmt --check` and `cargo +nightly clippy --all-targets` inspect
formatting and lints. Existing optional and vendored code has lint findings;
the recorded validation does not claim a warning-free tree.

| Evidence | Coverage |
| --- | --- |
| [`training_contract.rs`](../tests/training_contract.rs) | Scalar numerical oracle, schedules, topology, final assignment, reproducibility, corpus boundaries and malformed inputs |
| [`rebase_compat.rs`](../tests/rebase_compat.rs) | Legacy file variants, metadata preservation and mapped search compatibility |
| [`quality_contract.rs`](../tests/quality_contract.rs) | Feature modes, spectral projection, parameter validation and reader reproducibility |
| [`cli.rs`](../tests/cli.rs) | Process-level training, saving, reload queries, argument and input failures |
| [`integration_tests.rs`](../tests/integration_tests.rs) | End-to-end library composition |
| Inline module tests and Rust examples | Component behavior and public usage |

The default suite leaves neural-model and large bitmap examples ignored.
An all-feature compile does not establish GPU execution. Existing tests do not
establish cross-platform bit identity or general-English retrieval quality.
The [validation record](validation.md#rebase-integration) records the exercised
baseline and separates earlier corpus timings from the integrated binary.

## Measure changes

[`scripts/evaluate_corpus.py`](../scripts/evaluate_corpus.py) records corpus,
executable and model hashes, command parameters, training diagnostics, time and
repeated query output. Example, with a prepared UTF-8 corpus:

```sh
python3 scripts/evaluate_corpus.py corpus.txt target/evaluation \
  --dimension 32 --iterations 20000 --projection-dim 0
```

Hold corpus, preprocessing, grid, update count and seed fixed when isolating a
projection or algorithm change. Report measured time and memory together with
held-out retrieval quality. Repeat-load agreement tests reproducibility; it is
not an accuracy score. No dedicated benchmark target is currently checked in.

Labelled evaluation is provided by `proteus evaluate`; paired comparisons use
[`scripts/compare_quality.py`](../scripts/compare_quality.py). The
[quality guide](quality.md#evaluate-a-model) specifies schemas, split assignment,
coverage handling and resampling limits. Keep failed comparisons in the
validation record; do not choose defaults from a repeatedly inspected test set.

## Contributing and documentation

Agents must read [the engineering guide](../AGENTS.md) before making changes;
its [documentation contract](../AGENTS.md#18-documentation-structure-one-entry-point)
defines page ownership and the updates required with future code changes.

Changes to algorithms or serialization need the relevant numerical or
compatibility checks. Keep source/API changes and runnable examples consistent.
Generated models, corpora and local performance results do not belong in a
documentation-only change unless explicitly part of its evidence.

The root [README](../README.md) is the documentation entry point. Focused pages
link back there and link directly to relevant sections, source or tests; there
is no competing `docs/README.md`. Keep implementation behavior, measured results
and [research proposals](roadmap.md) distinct. Include assumptions and
falsification probes when making new semantic-quality claims.

## Attribution and license

The package declares the MIT license in [`Cargo.toml`](../Cargo.toml); this
checkout does not contain a standalone `LICENSE` file. Dependencies and vendored
components retain their own licensing terms.

The implementation draws on Semantic Folding research, Kohonen maps, TextTiling,
Roaring bitmaps, wtpsplit and Rayon. Algorithm sources are linked from the
[numerical contract](algorithm.md#primary-references),
[segmentation guide](library.md#sentence-and-topic-segmentation) and
[advanced API guide](advanced.md).
