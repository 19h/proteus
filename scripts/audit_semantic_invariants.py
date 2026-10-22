#!/usr/bin/env python3
"""Reproduce representation limits without changing the training implementation.

The generated corpora are controlled counterexamples, not language benchmarks.
An exit status of zero means the documented mathematical and implementation
properties were reproduced; it does not mean semantic accuracy was established.
Only the Python standard library and an existing Proteus executable are needed.
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


SENTENCES = (
    "dogs chase cats",
    "doctors treat patients",
    "teachers guide students",
    "kings govern kingdoms",
    "pilots guide sailors",
    "editors review authors",
    "farmers feed horses",
    "guards watch prisoners",
)


def sha256(path):
    """Hash a file without loading a potentially large executable into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, description):
    """Fail explicitly if the observed implementation differs from this audit."""
    if not condition:
        raise RuntimeError(description)


def binary_cosine(left, right):
    if not left or not right:
        raise ValueError("This mathematical probe requires nonempty bit sets")
    return len(left & right) / math.sqrt(len(left) * len(right))


def match_signed_evidence(query_positive, query_negative, positive, negative):
    """Evaluate unit-weight, already-canonicalized assertions in one scope.

    This checks a proposed scoring algebra. It does not extract assertions,
    resolve referents or establish the truth of an input statement.
    """
    if query_positive & query_negative or positive & negative:
        raise ValueError("Contradictory assertions require separate source scopes")
    required = query_positive | query_negative
    if not required:
        raise ValueError("At least one query assertion is required")

    supported = (query_positive & positive) | (query_negative & negative)
    conflicted = (query_positive & negative) | (query_negative & positive)
    unknown = required - supported - conflicted
    return {
        "score": (len(supported) - len(conflicted)) / len(required),
        "supported": len(supported),
        "conflicted": len(conflicted),
        "unknown": len(unknown),
    }


def mathematical_probes():
    """Check algebraic counterexamples independently of the Rust implementation."""
    left, middle, right = {0}, {0, 1}, {1}
    direct_distance = 1.0 - binary_cosine(left, right)
    indirect_distance = (
        1.0 - binary_cosine(left, middle)
        + 1.0 - binary_cosine(middle, right)
    )
    require(direct_distance > indirect_distance, "Cosine counterexample changed")

    permutation = {0: 7, 1: 2, 2: 6, 3: 0}
    original_sets = ({0, 1, 3}, {1, 2})
    permuted_sets = tuple(
        {permutation[position] for position in positions}
        for positions in original_sets
    )
    original_cosine = binary_cosine(*original_sets)
    permuted_cosine = binary_cosine(*permuted_sets)
    require(original_cosine == permuted_cosine, "Permutation invariance changed")

    # Independent closed-form calculation for A = [[3, 1], [1, 2]].
    # A^T A = [[10, 5], [5, 5]]; inverse = [[0.2, -0.2], [-0.2, 0.4]].
    matrix = ((3.0, 1.0), (1.0, 2.0))
    inverse_gram = ((0.2, -0.2), (-0.2, 0.4))
    whitened_gram = [
        [
            sum(
                matrix[row][inner] * inverse_gram[inner][column] * matrix[other][column]
                for inner in range(2)
                for column in range(2)
            )
            for other in range(2)
        ]
        for row in range(2)
    ]
    require(abs(whitened_gram[0][1]) < 1e-14, "Whitening counterexample changed")

    universe_size = 128 * 128
    cardinality = math.ceil(0.02 * universe_size)
    density = cardinality / universe_size
    overlap_variance = (
        cardinality * density * (1.0 - density)
        * (universe_size - cardinality) / (universe_size - 1)
    )
    query_assertion = {"chase(agent=dogs,patient=cats)"}
    signed_cases = {
        "same_assertion": match_signed_evidence(
            query_assertion, set(), query_assertion, set(),
        ),
        "reversed_arguments": match_signed_evidence(
            query_assertion, set(), {"chase(agent=cats,patient=dogs)"}, set(),
        ),
        "explicit_negation": match_signed_evidence(
            query_assertion, set(), set(), query_assertion,
        ),
        "missing_evidence": match_signed_evidence(
            query_assertion, set(), set(), set(),
        ),
    }
    require(signed_cases["same_assertion"]["score"] == 1.0, "Agreement score changed")
    require(signed_cases["reversed_arguments"]["score"] == 0.0, "Role matching changed")
    require(
        signed_cases["explicit_negation"]["score"] == -1.0,
        "Conflict score changed",
    )
    require(signed_cases["missing_evidence"]["unknown"] == 1, "Missing became negative")
    return {
        "unequal_cardinality_cosine_distance": {
            "direct": direct_distance,
            "via_middle": indirect_distance,
            "triangle_inequality_violated": True,
            "scope": (
                "Unequal cardinalities; fixed-cardinality cosine distance "
                "is scaled Hamming"
            ),
        },
        "global_bit_permutation": {
            "before": original_cosine,
            "after": permuted_cosine,
        },
        "union_collision": {
            "different_constituents": [[0], [1]],
            "same_union_as_single_set": [0, 1],
        },
        "full_rank_whitening": {
            "initial_row_cosine": 5.0 / math.sqrt(10.0 * 5.0),
            "whitened_row_gram": whitened_gram,
        },
        "independent_uniform_bit_sets": {
            "universe_size": universe_size,
            "cardinality": cardinality,
            "expected_overlap": cardinality * density,
            "expected_cosine": density,
            "cosine_standard_deviation": math.sqrt(overlap_variance) / cardinality,
            "scope": (
                "Independent uniformly selected sets, not trained word fingerprints"
            ),
        },
        "proposed_signed_evidence": {
            "cases": signed_cases,
            "scope": (
                "Manually canonicalized assertions; "
                "no semantic extraction or accuracy claim"
            ),
        },
    }


def implementation_probes(binary, output_directory):
    commands = []

    def execute(arguments):
        command = [str(binary), *map(str, arguments)]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        commands.append({
            "arguments": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
        })
        return result.stdout

    original = output_directory / "original.txt"
    reversed_corpus = output_directory / "reversed.txt"
    original.write_text("\n\n".join(SENTENCES) + "\n", encoding="utf-8")
    reversed_corpus.write_text(
        "\n\n".join(
            " ".join(reversed(sentence.split())) for sentence in SENTENCES
        ) + "\n",
        encoding="utf-8",
    )
    vocabulary = sorted({word for sentence in SENTENCES for word in sentence.split()})
    observations = {}

    for backend in ("train", "train-distributional"):
        models = []
        for corpus in (original, reversed_corpus):
            model = output_directory / f"{backend}-{corpus.stem}.retina"
            arguments = [
                backend, "-i", corpus, "-o", model, "-d", "16",
                "--min-count", "1", "--max-document-frequency", "1",
                "--seed", "42", "--threads", "1",
            ]
            if backend == "train":
                arguments.extend(["-n", "2000"])
            else:
                arguments.extend(["--embedding-dimension", "8"])
            execute(arguments)
            models.append(model)

        fingerprints = [
            {
                word: json.loads(execute(["fingerprint", "-r", model, word]))
                for word in vocabulary
            }
            for model in models
        ]
        require(
            fingerprints[0] == fingerprints[1],
            f"{backend}: reversing the controlled corpus changed word fingerprints",
        )
        forward = json.loads(execute(["fingerprint", "-r", models[0], SENTENCES[0]]))
        reverse = json.loads(execute([
            "fingerprint", "-r", models[0], " ".join(reversed(SENTENCES[0].split())),
        ]))
        require(
            forward == reverse,
            f"{backend}: text composition became order-sensitive",
        )
        require(bool(forward), f"{backend}: an empty fingerprint invalidates the probe")
        observations[backend] = {
            "identical_word_fingerprints": len(vocabulary),
            "different_input_hashes": sha256(original) != sha256(reversed_corpus),
            "original_model_sha256": sha256(models[0]),
            "reversed_model_sha256": sha256(models[1]),
            "forward_text": SENTENCES[0],
            "reversed_text": " ".join(reversed(SENTENCES[0].split())),
            "identical_text_positions": forward,
        }

    (output_directory / "commands.json").write_text(
        json.dumps(commands, indent=4) + "\n", encoding="utf-8",
    )
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    binary = arguments.binary.resolve(strict=True)
    arguments.output.mkdir(parents=True, exist_ok=False)
    report = {
        "executable_sha256": sha256(binary),
        "mathematical_probes": mathematical_probes(),
        "implementation_probes": implementation_probes(binary, arguments.output),
        "interpretation": (
            "Counterexamples to semantic sufficiency; no accuracy estimate"
        ),
    }
    require(
        sha256(binary) == report["executable_sha256"],
        "Executable changed during audit",
    )
    serialized_report = json.dumps(report, indent=4) + "\n"
    (arguments.output / "report.json").write_text(serialized_report, encoding="utf-8")
    print(serialized_report, end="")


if __name__ == "__main__":
    main()
