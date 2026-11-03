"""Every Python file of the repository compiles without a warning.

A warning of the compiler is nearly always a defect that waits for its line to run: two
strings side by side where a + was meant ("'str' object is not callable; perhaps you missed a
comma?") stopped the cluster's true check after its true evaluations, in a line no test ran.
"""

import subprocess
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def python_files():
    """The files git tracks, or every file under the root where there is no git (an export)."""
    try:
        listed = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        files = [ROOT / name for name in listed if name]
    except (OSError, subprocess.CalledProcessError):
        files = []
    return files or [p for p in ROOT.rglob("*.py") if "results" not in p.relative_to(ROOT).parts]


class SourceCompiles(unittest.TestCase):
    def test_no_file_gives_a_warning_or_an_error(self):
        files = python_files()
        self.assertGreater(len(files), 100)
        problems = []
        for file in files:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    compile(file.read_text(), str(file), "exec")
                except (SyntaxError, Warning, ValueError) as problem:
                    problems.append(f"{file.relative_to(ROOT)}: {problem}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
