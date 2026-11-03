"""Which build of classy and camb a cluster job may evaluate with (hpc/boltzmann_build.py): the environment's
own build, unless the run's job.env asks for the patched CLASS++ with CLASSPP_GUARD_DIR."""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "hpc"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "true_profiles"))

import boltzmann_build  # noqa: E402
import cluster_backend  # noqa: E402

PATCH = KIT / "patches" / "classpp_extrap_guard.patch"
ENV = "/codes/envs/client"
GUARD = "/codes/classpp_7cc41a51_guard"
ENV_CLASSY = ENV + "/lib/python3.10/site-packages/classy.cpython-310-x86_64-linux-gnu.so"
GUARD_CLASSY = GUARD + "/lib/classy.cpython-310-x86_64-linux-gnu.so"
ENV_CAMB = ENV + "/lib/python3.10/site-packages/camb/__init__.py"


class LocationRuleTests(unittest.TestCase):
    def rule(self, root, file, guard=""):
        return boltzmann_build.location_problem(root, file, env_prefix=ENV, guard_dir=guard)

    def test_without_the_variable_the_rule_is_the_one_before(self):
        # what launch.py and the profile backend did before the variable existed: inside CLIENT_ENV
        before = lambda file: os.path.realpath(file).startswith(os.path.realpath(ENV))
        for root, file in (("classy", ENV_CLASSY), ("classy", GUARD_CLASSY), ("camb", ENV_CAMB), ("camb", "/elsewhere/camb.py")):
            self.assertEqual(self.rule(root, file) is None, before(file), (root, file))
        self.assertIn("not from the environment", self.rule("classy", GUARD_CLASSY))
        # CLIENT_ENV unset: nothing is checked, as before
        self.assertIsNone(boltzmann_build.location_problem("classy", "/anywhere/classy.so", env_prefix="", guard_dir=""))

    def test_with_the_variable_classy_must_be_the_patched_build(self):
        self.assertIsNone(self.rule("classy", GUARD_CLASSY, GUARD))
        self.assertIn("patched build", self.rule("classy", ENV_CLASSY, GUARD))
        self.assertIn("patched build", self.rule("classy", GUARD_CLASSY, GUARD + "_other"))
        # camb stays the env's
        self.assertIsNone(self.rule("camb", ENV_CAMB, GUARD))
        self.assertIn("not from the environment", self.rule("camb", "/elsewhere/camb.py", GUARD))

    def test_the_defaults_are_read_from_the_environment(self):
        with mock.patch.dict(os.environ, {"CLIENT_ENV": ENV, "CLASSPP_GUARD_DIR": GUARD}):
            self.assertIsNone(boltzmann_build.location_problem("classy", GUARD_CLASSY))
            self.assertIsNotNone(boltzmann_build.location_problem("classy", ENV_CLASSY))
        with mock.patch.dict(os.environ, {"CLIENT_ENV": ENV, "CLASSPP_GUARD_DIR": "  "}):
            self.assertIsNone(boltzmann_build.location_problem("classy", ENV_CLASSY))   # blank: not set


class ProfileBackendTests(unittest.TestCase):
    """true_profiles/cluster_backend.py refuses a Cobaya model evaluating with another build."""

    def check(self, classy_file, guard):
        backend = types.SimpleNamespace(cobaya_model=types.SimpleNamespace(theory={"classy": object()}))
        fake = types.SimpleNamespace(__file__=classy_file)
        environment = {"CLIENT_ENV": ENV}
        if guard:
            environment["CLASSPP_GUARD_DIR"] = guard
        with mock.patch.dict(sys.modules, {"classy": fake}), mock.patch.dict(os.environ, environment):
            if not guard:
                os.environ.pop("CLASSPP_GUARD_DIR", None)
            cluster_backend._check_boltzmann(backend)

    def test_the_backend(self):
        self.check(ENV_CLASSY, None)
        with self.assertRaisesRegex(RuntimeError, "not from the environment"):
            self.check(GUARD_CLASSY, None)
        self.check(GUARD_CLASSY, GUARD)
        with self.assertRaisesRegex(RuntimeError, "patched build"):
            self.check(ENV_CLASSY, GUARD)


class ThePatch(unittest.TestCase):
    def test_it_changes_the_source_extrapolation_and_nothing_else(self):
        text = PATCH.read_text()
        self.assertEqual([l for l in text.splitlines() if l.startswith(("--- ", "+++ "))],
                         ["--- a/source/nonlinear_module.cpp", "+++ b/source/nonlinear_module.cpp"])
        self.assertIn("case extrap_max_scaled", text)
        self.assertEqual(text.count("@@ "), 1)


if __name__ == "__main__":
    unittest.main()
