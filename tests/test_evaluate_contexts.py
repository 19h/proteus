"""Independent input, resampling and failure contracts for contextual evaluation."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
specification = importlib.util.spec_from_file_location(
    "evaluate_contexts", SCRIPTS / "evaluate_contexts.py"
)
evaluation = importlib.util.module_from_spec(specification)
specification.loader.exec_module(evaluation)


class ContextEvaluationTests(unittest.TestCase):
    def test_source_indices_become_original_utf8_byte_spans(self):
        self.assertEqual(
            evaluation.target_span("café banks .", 1), {"start": 6, "end": 11}
        )
        with self.assertRaises(ValueError):
            evaluation.target_span("bank", 1)
        with self.assertRaises(ValueError):
            evaluation.target_span("bank", -1)

    def test_label_shape_and_pos_errors_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data.txt"
            gold = Path(folder) / "gold.txt"
            data.write_text("bank\tN\t1-0\tcafé banks .\tbank river .\n")
            gold.write_text("F\n")
            pairs, observations = evaluation.read_wic(data, gold)
            self.assertEqual(len(pairs), 1)
            self.assertFalse(pairs[0]["same_sense"])
            self.assertEqual(pairs[0]["left"]["target"], {"start": 6, "end": 11})
            self.assertEqual(observations[0]["word"], "bank")
            gold.write_text("T\nF\n")
            with self.assertRaises(ValueError):
                evaluation.read_wic(data, gold)
            data.write_text("bank\tX\t0-0\tbank\tbank\n")
            with self.assertRaises(ValueError):
                evaluation.read_wic(data)

    def test_cluster_bootstrap_preserves_repeated_observations_by_lemma(self):
        rows = [{"gold": True, "baseline": False, "predicted": True}] * 3
        observations = [{"word": "bank"}, {"word": "bank"}, {"word": "bat"}]
        result = evaluation.paired_cluster_bootstrap(rows, observations, 100, 42)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["delta_accuracy_95_percentile_interval"], [1.0, 1.0])
        self.assertIsNone(evaluation.paired_cluster_bootstrap([], [], 100, 42))

    def test_failed_native_evaluation_preserves_existing_report_and_input_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data.txt"
            retina = root / "model.retina"
            report = root / "report.json"
            data.write_text("bank\tN\t0-0\tbank money\tbank river\n")
            retina.write_bytes(b"fixture")
            report.write_bytes(b"existing report")
            arguments = argparse.Namespace(
                binary=Path(sys.executable),
                retina=retina,
                data=data,
                gold=None,
                lexicon=None,
                output=report,
                bootstrap_draws=100,
                seed=42,
            )
            with self.assertRaises(subprocess.CalledProcessError):
                evaluation.run(arguments)
            self.assertEqual(report.read_bytes(), b"existing report")
            arguments.output = data
            with self.assertRaises(ValueError):
                evaluation.run(arguments)
            self.assertTrue(data.read_text().startswith("bank\tN"))


if __name__ == "__main__":
    unittest.main()
