#!/usr/bin/env python3
"""Compare two Proteus evaluation reports on identical covered word pairs.

The development split contains pairs whose SHA-256 digest, interpreted as an
integer, is divisible by three. The remaining pairs form the test split. The
hashed input is word1 + TAB + word2, encoded as UTF-8.

The paired percentile bootstrap is descriptive: pairs sharing a word need not
be independent. Undefined correlations and bootstrap draws remain explicit.
This script requires only the Python standard library.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def average_ranks(values):
    """Return zero-based average ranks, assigning equal ranks to tied values."""
    sorted_indices = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    group_start = 0

    while group_start < len(sorted_indices):
        group_end = group_start + 1
        while (
            group_end < len(sorted_indices)
            and values[sorted_indices[group_start]] == values[sorted_indices[group_end]]
        ):
            group_end += 1

        average_rank = (group_start + group_end - 1) / 2
        for index in sorted_indices[group_start:group_end]:
            ranks[index] = average_rank
        group_start = group_end

    return ranks


def spearman(reference_scores, predicted_scores):
    """Compute rank correlation, or None for constant or insufficient inputs."""
    if len(reference_scores) != len(predicted_scores):
        raise ValueError("Reference and prediction lengths differ")
    if len(reference_scores) < 2:
        return None

    reference_ranks = average_ranks(reference_scores)
    predicted_ranks = average_ranks(predicted_scores)
    mean_rank = (len(reference_ranks) - 1) / 2
    reference_variance = sum((rank - mean_rank) ** 2 for rank in reference_ranks)
    predicted_variance = sum((rank - mean_rank) ** 2 for rank in predicted_ranks)
    if reference_variance == 0 or predicted_variance == 0:
        return None

    covariance = sum(
        (reference_rank - mean_rank) * (predicted_rank - mean_rank)
        for reference_rank, predicted_rank in zip(reference_ranks, predicted_ranks)
    )
    return covariance / math.sqrt(reference_variance * predicted_variance)


def belongs_to_split(word1, word2, split):
    pair_digest = hashlib.sha256((word1 + "\t" + word2).encode("utf-8")).hexdigest()
    is_development_pair = int(pair_digest, 16) % 3 == 0
    return split == "all" or is_development_pair == (split == "dev")


def load_report(path, split):
    raw_report = path.read_bytes()
    document = json.loads(raw_report)
    selected_pairs = {}

    for observation_index, prediction in enumerate(document["predictions"]):
        pair = prediction["word1"], prediction["word2"]
        if not belongs_to_split(*pair, split):
            continue
        for field in ("score", "predicted"):
            value = prediction[field]
            if value is None and field == "predicted":
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {field} for pair {pair}")
        selected_pairs[(observation_index, *pair)] = prediction

    return selected_pairs, hashlib.sha256(raw_report).hexdigest()


def paired_bootstrap(reference_scores, baseline_scores, candidate_scores, draws, seed):
    generator = random.Random(seed)
    differences = []

    for _ in range(draws):
        indices = [generator.randrange(len(reference_scores)) for _ in reference_scores]
        sampled_references = [reference_scores[index] for index in indices]
        baseline_correlation = spearman(
            sampled_references, [baseline_scores[index] for index in indices]
        )
        candidate_correlation = spearman(
            sampled_references, [candidate_scores[index] for index in indices]
        )
        if baseline_correlation is not None and candidate_correlation is not None:
            differences.append(candidate_correlation - baseline_correlation)

    differences.sort()
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.resamples < 1:
        parser.error("--resamples must be positive")

    baseline, baseline_digest = load_report(arguments.baseline, arguments.split)
    candidate, candidate_digest = load_report(arguments.candidate, arguments.split)
    if baseline.keys() != candidate.keys() or any(
        baseline[pair]["score"] != candidate[pair]["score"] for pair in baseline
    ):
        raise ValueError("Reports must contain identical ordered reference observations and scores")

    common_pairs = sorted(
        pair
        for pair in baseline
        if baseline[pair]["predicted"] is not None
        and candidate[pair]["predicted"] is not None
    )
    reference_scores = [baseline[pair]["score"] for pair in common_pairs]
    baseline_scores = [baseline[pair]["predicted"] for pair in common_pairs]
    candidate_scores = [candidate[pair]["predicted"] for pair in common_pairs]
    baseline_correlation = spearman(reference_scores, baseline_scores)
    candidate_correlation = spearman(reference_scores, candidate_scores)
    bootstrap_differences = paired_bootstrap(
        reference_scores,
        baseline_scores,
        candidate_scores,
        arguments.resamples,
        arguments.seed,
    )
    interval = None
    if bootstrap_differences:
        interval = [
            bootstrap_differences[int((len(bootstrap_differences) - 1) * quantile)]
            for quantile in (0.025, 0.975)
        ]

    difference = None
    if baseline_correlation is not None and candidate_correlation is not None:
        difference = candidate_correlation - baseline_correlation

    report = {
        "split": arguments.split,
        "reference_pairs": len(baseline),
        "common_covered_pairs": len(common_pairs),
        "baseline_covered_pairs": sum(
            prediction["predicted"] is not None for prediction in baseline.values()
        ),
        "candidate_covered_pairs": sum(
            prediction["predicted"] is not None for prediction in candidate.values()
        ),
        "baseline_spearman": baseline_correlation,
        "candidate_spearman": candidate_correlation,
        "delta_spearman": difference,
        "paired_bootstrap_95_percentile_interval": interval,
        "valid_bootstrap_draws": len(bootstrap_differences),
        "requested_bootstrap_draws": arguments.resamples,
        "seed": arguments.seed,
        "baseline_report_sha256": baseline_digest,
        "candidate_report_sha256": candidate_digest,
        "scope": (
            "Common-coverage pair ranking; pair resampling is descriptive and "
            "does not establish domain generalization."
        ),
    }
    print(json.dumps(report, indent=4, allow_nan=False))


if __name__ == "__main__":
    main()
