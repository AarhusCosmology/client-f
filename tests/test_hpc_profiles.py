"""The true-profile launcher and its helpers (hpc/true_profiles/)."""

import contextlib
import io
import json
import logging
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc" / "true_profiles"))
import cluster_backend  # noqa: E402
import launch_profiles  # noqa: E402
import make_config  # noqa: E402
import status  # noqa: E402

# launch_profiles puts $CLIENT_REPO first on the path, as on the cluster: here, this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Backend(unittest.TestCase):
    def test_a_logged_error_is_a_refused_point(self):
        from cobaya.log import LoggedError

        def refuse(x):
            raise LoggedError(logging.getLogger("test"), "Not possible to extrapolate to k=3105.2 1/Mpc")

        with mock.patch.object(cluster_backend, "_evaluate", refuse), mock.patch.dict(cluster_backend._kinds, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(cluster_backend.evaluate([0.0]), -np.inf)
            self.assertEqual(cluster_backend.evaluate([1.0]), -np.inf)
            self.assertEqual(list(cluster_backend._kinds.values()), [2])       # one kind, whatever the numbers in it
        self.assertEqual(printed.getvalue().count("refuses a point"), 1)        # said once

    def test_another_error_still_stops(self):
        def broken(x):
            raise KeyError("x")

        with mock.patch.object(cluster_backend, "_evaluate", broken):
            self.assertRaises(KeyError, cluster_backend.evaluate, [0.0])

    def test_turns_are_taken_one_group_at_a_time(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cluster_backend.tempfile, "gettempdir", lambda: tmp):
            first = cluster_backend._turn(1)
            got = []
            waiter = threading.Thread(target=lambda: got.append(cluster_backend._turn(1)))
            waiter.start()
            time.sleep(0.3)
            self.assertEqual(got, [])                                           # the only turn is taken
            first.close()
            waiter.join(timeout=10)
            self.assertEqual(len(got), 1)
            got[0].close()
            pair = [cluster_backend._turn(2), cluster_backend._turn(2)]         # two turns: both at once
            for handle in pair:
                handle.close()

    def test_camb_accuracy_target_is_set_before_the_likelihood_is_built(self):
        import camb

        seen = []
        before = camb.config.AccuracyTarget
        try:
            with mock.patch.object(cluster_backend, "_initialize", lambda *a: seen.append(camb.config.AccuracyTarget)), \
                    mock.patch.object(cluster_backend.evaluation, "_BACKEND", None), \
                    mock.patch.dict(cluster_backend.os.environ, {"CLIENT_CAMB_ACCURACY_TARGET": "0", "CLIENT_LOAD_GROUP": ""}), \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                cluster_backend.initialize()
            self.assertEqual(seen, [0])
            self.assertIn("CAMB AccuracyTarget 0", printed.getvalue())
            camb.config.AccuracyTarget = before
            with mock.patch.object(cluster_backend, "_initialize", lambda *a: seen.append(camb.config.AccuracyTarget)), \
                    mock.patch.object(cluster_backend.evaluation, "_BACKEND", None), \
                    mock.patch.dict(cluster_backend.os.environ, {"CLIENT_CAMB_ACCURACY_TARGET": "", "CLIENT_LOAD_GROUP": ""}), \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                cluster_backend.initialize()
            self.assertEqual(seen, [0, before])                                 # unset: CAMB as it is
            self.assertNotIn("AccuracyTarget", printed.getvalue())
        finally:
            camb.config.AccuracyTarget = before

    def test_install_points_the_profile_code_here(self):
        import profiles.evaluation as evaluation

        with mock.patch.object(evaluation, "initialize_backend"), mock.patch.object(evaluation, "evaluate_backend"):
            cluster_backend.install()
            self.assertIs(evaluation.initialize_backend, cluster_backend.initialize)
            self.assertIs(evaluation.evaluate_backend, cluster_backend.evaluate)


class RegionIsTheYamls(unittest.TestCase):
    """Where a boost network is searched is CLiENT's profile setting `region`
    (tests/test_profile_region.py); the launcher only makes sure nothing else is believed."""

    def launch(self, arguments, environment=None):
        import profiles.__main__ as profiles_main

        ran = []
        with mock.patch.dict(launch_profiles.os.environ, environment or {}), \
                mock.patch.object(launch_profiles.sys, "argv", ["launch_profiles.py", *arguments]), \
                mock.patch.object(cluster_backend, "install"), \
                mock.patch.object(profiles_main, "main", lambda: ran.append(True)), \
                contextlib.redirect_stdout(io.StringIO()):
            launch_profiles.os.environ.pop("CLIENT_PROFILE_NEWTON", None)
            launch_profiles.os.environ.pop("CLIENT_CAMB_ACCURACY_TARGET", None)
            if not (environment or {}).get("CLIENT_PROFILE_REGION"):
                launch_profiles.os.environ.pop("CLIENT_PROFILE_REGION", None)
            launch_profiles.main()
        return ran

    def test_a_new_study_records_its_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            arguments = ["run", "--config", "profile.yaml", "-o", str(Path(tmp) / "study")]
            self.assertEqual(self.launch(arguments), [True])
            self.assertEqual(json.loads((Path(tmp) / "study.newton_settings.json").read_text()), {})

    def test_a_study_of_the_kit_that_applied_the_region_itself_is_not_continued(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            study.mkdir()
            (study / "manifest.json").write_text("{}")
            record = Path(tmp) / "study.newton_settings.json"
            arguments = ["run", "--config", "profile.yaml", "-o", str(study), "--resume"]
            for edge in (1.0, "run"):
                record.write_text(json.dumps(dict(step=0.5, CLIENT_PROFILE_REGION=edge)))
                with self.subTest(edge=edge), self.assertRaisesRegex(SystemExit, "applied the region itself"):
                    self.launch(arguments, {"CLIENT_PROFILE_NEWTON": "step=0.5"})
            # Made with the bare network (0), or before the launcher knew a region: its other settings decide.
            for saved in (dict(CLIENT_PROFILE_REGION=0), {}):
                record.write_text(json.dumps(saved))
                self.assertEqual(self.launch(arguments), [True])
            record.write_text(json.dumps(dict(CLIENT_CAMB_ACCURACY_TARGET=0)))
            with self.assertRaisesRegex(SystemExit, "was made with the settings"):
                self.launch(arguments)

    def test_the_old_variable_stops_the_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            arguments = ["run", "--config", "profile.yaml", "-o", str(Path(tmp) / "study")]
            for value in ("1", "run"):
                with self.subTest(value=value), self.assertRaisesRegex(SystemExit, "not read any more"):
                    self.launch(arguments, {"CLIENT_PROFILE_REGION": value})
            self.assertFalse((Path(tmp) / "study.newton_settings.json").exists())

    def test_make_config_writes_the_setting(self):
        from profiles.config import ProfileConfig

        for text, expected in [("run", "run"), ("RUN", "run"), ("1", "1"), ("1.50", "1.5"), ("none", "null"), ("null", "null")]:
            self.assertEqual(make_config.region_setting(text), expected)
        for text in ("0", "-1", "inf", "nan", "edge"):
            with self.subTest(text=text):
                self.assertRaises(SystemExit, make_config.region_setting, text)
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "analysis").mkdir(parents=True)
            np.savez(run / "analysis" / "posterior_it_7.npz", x=rng.normal(0.0, 0.1, (5000, 2)), names=np.array(["a", "b"]))
            # A later network's chain, wider: the ranges must not come from it.
            np.savez(run / "analysis" / "posterior_it_9.npz", x=rng.normal(0.0, 0.3, (5000, 2)), names=np.array(["a", "b"]))
            (run / "metadata.json").write_text(json.dumps(dict(parameters=[dict(name=n, lower=-1.0, upper=1.0) for n in "ab"])))
            for words, expected in [([], "run"), (["--region", "1"], 1), (["--region", "none"], None)]:
                out = Path(tmp) / "profile.yaml"
                with mock.patch.object(sys, "argv", ["make_config.py", str(run), "--iteration", "7", "--out", str(out),
                                                     "--workers", "1", *words]), contextlib.redirect_stdout(io.StringIO()):
                    make_config.main()
                self.assertEqual(ProfileConfig.load(out).region, expected)
                self.assertLess(ProfileConfig.load(out).ranges["a"][1], 0.45)    # 0.33 from network 7, 0.99 from 9

    def test_make_config_scan_options_and_budget(self):
        from profiles.config import ProfileConfig

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "analysis").mkdir(parents=True)
            np.savez(run / "analysis" / "posterior_it_7.npz", x=rng.normal(0.0, 0.1, (5000, 2)), names=np.array(["a", "b"]))
            (run / "metadata.json").write_text(json.dumps(dict(parameters=[dict(name=n, lower=-1.0, upper=1.0) for n in "ab"])))
            out = Path(tmp) / "profile.yaml"

            def made(*words):
                with mock.patch.object(sys, "argv", ["make_config.py", str(run), "--iteration", "7", "--out", str(out),
                                                     "--workers", "1", "--points", "25", *words]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    make_config.main()
                return ProfileConfig.load(out)

            default = made()
            self.assertEqual((default.hybrid_hops, default.gtol), (ProfileConfig.hybrid_hops, ProfileConfig.gtol))
            self.assertNotIn("hybrid_hops", out.read_text())
            # 50 nodes of 2 parameters at maxfun 24: four gradient rounds a node (1000) is less than the plan
            # can use, every node and the global best fit at maxfun with the nodes' and audit's calls (2000)
            self.assertEqual(default.maxfun, 24)
            self.assertEqual(default.total_call_budget, 2000)
            chosen = made("--hybrid-hops", "30", "--gtol", "1e-3", "--budget", "1500")
            self.assertEqual((chosen.hybrid_hops, chosen.gtol, chosen.total_call_budget), (30, 1e-3, 1500))


class NewtonSettings(unittest.TestCase):
    def test_fields_are_set_and_the_yaml_keeps_its_own(self):
        import profiles.residual_newton as newton

        with mock.patch.object(newton, "NewtonConfig", newton.NewtonConfig):
            settings = cluster_backend.newton_settings("step=0.5, noise=0.01, flat_central=3, secant=false")
            self.assertEqual(settings, dict(step=0.5, noise=0.01, flat_central=3, secant=False))
            config = newton.NewtonConfig(tol=0.05, maxcalls=300)
            self.assertEqual((config.step, config.noise, config.flat_central, config.secant, config.tol, config.maxcalls),
                             (0.5, 0.01, 3, False, 0.05, 300))
            self.assertEqual(newton.NewtonConfig().trust0, 0.5)                 # what was not named keeps its default

    def test_nothing_set_changes_nothing(self):
        import profiles.residual_newton as newton

        before = newton.NewtonConfig
        self.assertEqual(cluster_backend.newton_settings(""), {})
        self.assertIs(newton.NewtonConfig, before)

    def test_what_the_yaml_carries_and_unknown_names_are_refused(self):
        for text in ("tol=0.5", "maxcalls=10", "stepp=0.5", "step"):
            with self.subTest(text=text):
                self.assertRaises(ValueError, cluster_backend.newton_settings, text)


class Ranges(unittest.TestCase):
    def test_a_range_without_a_width_names_its_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain = Path(tmp) / "posterior_it_7.npz"
            np.savez(chain, x=np.column_stack([np.ones(100), np.linspace(0, 1, 100)]), names=np.array(["a", "b"]))
            with self.assertRaisesRegex(SystemExit, "a: the chain's 3 sigma range has no width .*--range a LOW HIGH"):
                make_config.ranges(chain, ["a", "b"], np.array([[0.0, 2.0], [0.0, 1.0]]), 3.0, ["a", "b"])

    def test_three_sigma_quantiles_cut_at_the_box(self):
        rng = np.random.default_rng(0)
        names = ["a", "b", "c"]
        x = np.column_stack([rng.normal(1.0, 0.1, 200000), np.abs(rng.normal(0.0, 1.0, 200000)), rng.uniform(-5, 5, 200000)])
        bounds = np.array([[0.0, 2.0], [0.0, 10.0], [-5.0, 5.0]])
        with tempfile.TemporaryDirectory() as tmp:
            chain = Path(tmp) / "posterior_it_7.npz"
            np.savez(chain, x=x, names=np.array(names))
            found, notes = make_config.ranges(chain, names, bounds, 3.0, names)
            self.assertEqual(make_config.network_chain(Path(tmp), 7), chain)
            with self.assertRaisesRegex(SystemExit, "no posterior_it_8.npz"):
                make_config.network_chain(Path(tmp), 8)
        self.assertAlmostEqual(found["a"][0], 1.0 - 0.33, delta=0.02)           # 3 sigma and 5% of the span each side
        self.assertAlmostEqual(found["a"][1], 1.0 + 0.33, delta=0.02)
        self.assertEqual(found["b"][0], 0.0)                                    # piled against the lower edge: on it
        self.assertLess(found["b"][1], 4.0)
        self.assertEqual(found["c"], [-5.0, 5.0])                               # fills the box: both edges
        self.assertEqual(notes["c"][2], ["lower edge of the box", "upper edge of the box"])
        for name, (low, high) in found.items():
            k = names.index(name)
            self.assertTrue(bounds[k, 0] <= low < high <= bounds[k, 1])


class Status(unittest.TestCase):
    def test_summary_of_a_study_in_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "study"
            (out / "true").mkdir(parents=True)
            (out / "calls").mkdir()
            (out / "manifest.json").write_text(json.dumps(dict(iteration=25, run="/somewhere/20261008_run", configuration=dict(
                parameters=["mnu", "wa"], profile_points=5, total_call_budget=1000, maxfun=306))))
            (out / "status.txt").write_text("direct and true profiles (corrected)\n")
            events = []
            for call in range(1, 301):
                events += [dict(event="start", call=call, time=1000.0 + call, stage="corrected/node/mnu/0", x=[0.0]),
                           dict(event="return", call=call, time=1013.0 + call, value=-1.0, seconds=13.0, stage="corrected/node/mnu/0")]
            (out / "calls" / "attempt_0000.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n{\"event\": \"sta")
            for (p, j), (reason, calls, gain) in {(0, 2): ("converged", 55, 0.3), (0, 1): ("converged", 110, 0.5), (1, 2): ("budget", 306, 2.0)}.items():
                (out / "true" / f"node_{p:03d}_{j:05d}.json").write_text(json.dumps(dict(
                    logl=-10.0 + gain, direct_logl=-10.0, diagnostics=dict(reason=reason, backend_calls=calls, gradient_rounds=calls // 50))))
            with mock.patch.object(sys, "argv", ["status.py", str(out)]), contextlib.redirect_stdout(io.StringIO()) as printed:
                status.main()
        text = printed.getvalue()
        self.assertIn("2 parameters x 5 points = 10 nodes, from network 25", text)
        self.assertIn("true calls: 300 of the budget 1000; a call takes 13.0 s", text)
        self.assertIn("nodes finished: 3 of 10; ended as converged 2, budget 1", text)
        self.assertIn("median 110", text)
        self.assertIn("most 2.000", text)
        self.assertIn("about 770 calls left", text)                             # 7 nodes at the median cost of 110
        self.assertIn("would pass its total_call_budget", text)                 # 300 + 770 > 1000


if __name__ == "__main__":
    unittest.main()
