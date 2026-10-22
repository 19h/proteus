"""Standard-library contracts for source preservation and atomic preparation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_semantic.py"
SPECIFICATION = importlib.util.spec_from_file_location("prepare_semantic", SCRIPT)
PREPARATION = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(PREPARATION)


class PreparationContract(unittest.TestCase):
    def test_paragraphs_preserve_utf8_offsets_wrapping_and_line_endings(self):
        first = "Café\r\nwrapped\r\n"
        separator = "\r\n"
        second = "next π"
        actual = list(PREPARATION.paragraphs(first + separator + second))
        expected = [(first, 0), (second, len((first + separator).encode("utf-8")))]
        self.assertEqual(actual, expected)
        self.assertEqual(list(PREPARATION.paragraphs("\n\r\n")), [])

    def test_failed_output_keeps_existing_destination_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            destination = directory / "prepared.jsonl"
            destination.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                with PREPARATION.output_stream(str(destination)) as stream:
                    stream.write("partial")
                    raise ValueError("incomplete preparation")
            self.assertEqual(destination.read_bytes(), b"original")
            self.assertEqual(list(directory.iterdir()), [destination])
            with PREPARATION.output_stream(str(destination)) as stream:
                stream.write("complete π\n")
            self.assertEqual(destination.read_text(), "complete π\n")

    def test_model_identity_depends_on_relative_names_and_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left = directory / "left"
            right = directory / "right"
            left.mkdir()
            right.mkdir()
            for name, contents in [("weights", b"abc"), ("configuration", b"def")]:
                (left / name).write_bytes(contents)
            for name, contents in [("configuration", b"def"), ("weights", b"abc")]:
                (right / name).write_bytes(contents)
            self.assertEqual(
                PREPARATION.directory_hash(left), PREPARATION.directory_hash(right)
            )
            (right / "weights").rename(right / "different-weights")
            self.assertNotEqual(
                PREPARATION.directory_hash(left), PREPARATION.directory_hash(right)
            )


if __name__ == "__main__":
    unittest.main()
