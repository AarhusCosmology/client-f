"""Which build of classy and camb a cluster job may evaluate with (hpc/boltzmann_build.py): the environment's
own build, unless the run's job.env asks for the patched CLASS++ with CLASSPP_GUARD_DIR or another commit's build
with CLASSPP_BUILD_DIR; and the commits the launcher records."""

import os
import subprocess
import tempfile
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
BUILD = "/codes/classpp_0a57bbc6"
BUILD_CLASSY = BUILD + "/lib/classy.cpython-310-x86_64-linux-gnu.so"
ENV_CAMB = ENV + "/lib/python3.10/site-packages/camb/__init__.py"


class LocationRuleTests(unittest.TestCase):
    def rule(self, root, file, guard="", build=""):
        return boltzmann_build.location_problem(root, file, env_prefix=ENV, guard_dir=guard, build_dir=build)

    def test_without_the_variable_the_rule_is_the_one_before(self):
        # what launch.py and the profile backend did before the variable existed: inside CLIENT_ENV
        before = lambda file: os.path.realpath(file).startswith(os.path.realpath(ENV))
        for root, file in (("classy", ENV_CLASSY), ("classy", GUARD_CLASSY), ("camb", ENV_CAMB), ("camb", "/elsewhere/camb.py")):
            self.assertEqual(self.rule(root, file) is None, before(file), (root, file))
        self.assertIn("not from the environment", self.rule("classy", GUARD_CLASSY))
        # CLIENT_ENV unset: nothing is checked, as before
        self.assertIsNone(boltzmann_build.location_problem("classy", "/anywhere/classy.so", env_prefix="", guard_dir="",
                                                           build_dir=""))

    def test_with_the_variable_classy_must_be_the_patched_build(self):
        self.assertIsNone(self.rule("classy", GUARD_CLASSY, GUARD))
        self.assertIn("patched build", self.rule("classy", ENV_CLASSY, GUARD))
        self.assertIn("patched build", self.rule("classy", GUARD_CLASSY, GUARD + "_other"))
        # camb stays the env's
        self.assertIsNone(self.rule("camb", ENV_CAMB, GUARD))
        self.assertIn("not from the environment", self.rule("camb", "/elsewhere/camb.py", GUARD))

    def test_with_a_build_directory_classy_must_be_that_build(self):
        self.assertIsNone(self.rule("classy", BUILD_CLASSY, build=BUILD))
        for other in (ENV_CLASSY, GUARD_CLASSY):
            self.assertIn("CLASS++ build in", self.rule("classy", other, build=BUILD))
        self.assertIsNone(self.rule("camb", ENV_CAMB, build=BUILD))                     # camb stays the env's
        self.assertIn("both set", self.rule("classy", BUILD_CLASSY, GUARD, BUILD))     # one build per run
        with mock.patch.dict(os.environ, {"CLIENT_ENV": ENV, "CLASSPP_BUILD_DIR": BUILD}):
            os.environ.pop("CLASSPP_GUARD_DIR", None)
            self.assertIsNone(boltzmann_build.location_problem("classy", BUILD_CLASSY))
            self.assertIsNotNone(boltzmann_build.location_problem("classy", ENV_CLASSY))

    def test_the_defaults_are_read_from_the_environment(self):
        with mock.patch.dict(os.environ, {"CLIENT_ENV": ENV, "CLASSPP_GUARD_DIR": GUARD}):
            os.environ.pop("CLASSPP_BUILD_DIR", None)
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
            os.environ.pop("CLASSPP_BUILD_DIR", None)
            cluster_backend._check_boltzmann(backend)

    def test_the_backend(self):
        self.check(ENV_CLASSY, None)
        with self.assertRaisesRegex(RuntimeError, "not from the environment"):
            self.check(GUARD_CLASSY, None)
        self.check(GUARD_CLASSY, GUARD)
        with self.assertRaisesRegex(RuntimeError, "patched build"):
            self.check(ENV_CLASSY, GUARD)


class Commits(unittest.TestCase):
    """What launch.py records in logs/backend_<job>.json."""

    def test_client_s_commit_is_read_from_its_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(boltzmann_build.commits(tmp, {})["client_commit"])          # not a checkout
            git = ["git", "-C", tmp, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
            subprocess.run(git[:3] + ["init", "-q"], check=True)
            (Path(tmp) / "a.py").write_text("a = 1\n")
            subprocess.run(git + ["add", "a.py"], check=True)
            subprocess.run(git + ["commit", "-q", "-m", "a"], check=True)
            head = subprocess.run(git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(boltzmann_build.commits(tmp, {})["client_commit"], head)
            (Path(tmp) / "b.py").write_text("untracked\n")                                 # untracked files do not count
            self.assertEqual(boltzmann_build.commits(tmp, {})["client_commit"], head)
            (Path(tmp) / "a.py").write_text("a = 2\n")
            self.assertEqual(boltzmann_build.commits(tmp, {})["client_commit"], head + " (DIRTY)")

    def test_class_s_commit_is_the_one_the_job_names(self):
        commits = lambda environ: {k: v for k, v in boltzmann_build.commits("/nowhere", environ).items() if k != "client_commit"}
        self.assertEqual(commits({}), dict(classpp_commit=None, classpp_build="environment"))
        self.assertEqual(commits({"CLASSPP_COMMIT": "7cc41a51"}), dict(classpp_commit="7cc41a51", classpp_build="environment"))
        self.assertEqual(commits({"CLASSPP_COMMIT": "7cc41a51", "CLASSPP_GUARD_DIR": GUARD}),
                         dict(classpp_commit="7cc41a51", classpp_build="CLASSPP_GUARD_DIR"))
        # a build of another commit: its own commit, never the environment's (the kit once recorded that one)
        self.assertEqual(commits({"CLASSPP_COMMIT": "7cc41a51", "CLASSPP_BUILD_DIR": BUILD, "CLASSPP_BUILD_COMMIT": "0a57bbc6"}),
                         dict(classpp_commit="0a57bbc6", classpp_build="CLASSPP_BUILD_DIR"))
        self.assertEqual(commits({"CLASSPP_COMMIT": "7cc41a51", "CLASSPP_BUILD_DIR": BUILD}),
                         dict(classpp_commit=None, classpp_build="CLASSPP_BUILD_DIR"))


class ThePatch(unittest.TestCase):
    def test_it_changes_the_source_extrapolation_and_nothing_else(self):
        text = PATCH.read_text()
        self.assertEqual([l for l in text.splitlines() if l.startswith(("--- ", "+++ "))],
                         ["--- a/source/nonlinear_module.cpp", "+++ b/source/nonlinear_module.cpp"])
        self.assertIn("case extrap_max_scaled", text)
        self.assertEqual(text.count("@@ "), 1)


if __name__ == "__main__":
    unittest.main()
