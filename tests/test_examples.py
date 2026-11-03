"""The settings files shipped in input/ are valid: each run YAML parses and names a likelihood
file that exists, and each profile YAML parses."""

import unittest
from pathlib import Path
from unittest import mock

import yaml

from config.config import Config
from profiles.config import ProfileConfig

ROOT = Path(__file__).resolve().parents[1]


class Examples(unittest.TestCase):
    def test_run_settings(self):
        files = [p for p in sorted((ROOT / "input").glob("*.yaml")) if not p.name.startswith("profiles_")]
        self.assertGreater(len(files), 5)
        for file in files:
            with self.subTest(file=file.name):
                config = Config.from_yaml(file)
                self.assertTrue((ROOT / config.likelihood.input).is_file(), config.likelihood.input)

    def test_profile_settings(self):
        for name in ("profiles_example.yaml", "profiles_2d_example.yaml"):
            with self.subTest(file=name), mock.patch("os.cpu_count", return_value=64):
                ProfileConfig.load(ROOT / "input" / name)

    def test_the_recommended_sets_differ_as_documented(self):
        near = yaml.safe_load((ROOT / "input/recommended_near_gaussian.yaml").read_text())
        far = yaml.safe_load((ROOT / "input/recommended_non_gaussian.yaml").read_text())
        self.assertNotIn("train_domain", near["acquisition"]["boost"])
        self.assertEqual(far["acquisition"]["boost"]["train_domain"], 1.25)
