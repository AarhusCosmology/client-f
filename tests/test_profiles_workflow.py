"""Cheap analytic end-to-end regression: no CLASS or Planck resources required."""

import contextlib, io, json, os, tempfile, unittest
from types import SimpleNamespace
from pathlib import Path
from concurrent.futures import Future
import numpy as np
import yaml
from profiles.config import ProfileConfig
from profiles.evaluation import WorkflowLedger, aggregate_counts
from profiles.workflow import canonical_points, run_workflow


class Pool:
    def __call__(self, x):
        return -float(np.asarray(x) @ x)

    def submit(self, x):
        f = Future()
        f.set_result(self(x))
        return f


class ProfilesUnitTests(unittest.TestCase):
    def test_configuration(self):
        for kwargs in [
            dict(workers=(os.cpu_count() or 1) + 1),  # more local processes than CPUs
            dict(chain_length=-1),
            dict(chain_length=1.5),
            dict(parameters=["x", "x"]),
            dict(added_points=-1),
            dict(ranges={"x": [1, 0]}),
            dict(corrected_tol=-1e-3),
        ]:
            c = ProfileConfig(parameters=["x"])
            c.__dict__.update(kwargs)
            with self.assertRaises(ValueError):
                c.validate()
        c = ProfileConfig(parameters=["x"])
        self.assertEqual(c.corrected_tolerance(), 1e-2)
        self.assertEqual(c.chain_length, 2)
        with tempfile.TemporaryDirectory() as d:
            for removed in ("true_optimizer", "corrected_method", "corrected_probes",
                            "surface_methods", "audit_rows", "radius", "final_radius"):
                p = Path(d) / "old.yaml"
                p.write_text(f"parameters: [x]\n{removed}: 1\n")
                with self.assertRaisesRegex(ValueError, "unknown profile settings"):
                    ProfileConfig.load(p)
        with self.assertRaisesRegex(ValueError, "surrogate_optimizer"):
            ProfileConfig(parameters=["x"], surrogate_optimizer="bfgs").validate()
        self.assertFalse(c.retrain)
        self.assertEqual(c.added_points, 0)
        self.assertTrue(ProfileConfig.load(parameters=["x"], added_points=12).retrain)
        # In Python the two are said together (the example of docs/create_profiles.md): the
        # rule above is the YAML reader's, and the example without retrain stopped at once.
        ProfileConfig(parameters=["x"], added_points=1000, retrain=True).validate()
        with self.assertRaisesRegex(ValueError, "retrain=false requires added_points=0"):
            ProfileConfig(parameters=["x"], added_points=1000).validate()
        self.assertIn("added_points=1000, retrain=True)",
                      (Path(__file__).resolve().parents[1] / "docs" / "create_profiles.md").read_text())
        with self.assertRaises(ValueError):
            ProfileConfig.load(parameters=["x"], added_points=12, retrain=False)
        with self.assertRaises(ValueError):
            ProfileConfig.load(parameters=["x"], added_points="invalid")
        with tempfile.TemporaryDirectory() as d:
            # PyYAML reads 1e-2 as a string; it used to be refused as "must be finite"
            p = Path(d) / "numbers.yaml"
            p.write_text("parameters: [x]\ncorrected_tol: 1e-2\ngtol: 1e-4\n")
            c = ProfileConfig.load(p)
            self.assertEqual((c.corrected_tol, c.gtol), (1e-2, 1e-4))
            p.write_text("parameters: [x]\ncorrected_tol: small\n")
            with self.assertRaisesRegex(ValueError, "corrected_tol must be finite"):
                ProfileConfig.load(p)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.yaml"
            p.write_text("paramters: [x]\n")
            with self.assertRaises(ValueError):
                ProfileConfig.load(p)

    def test_label_audit_checks_the_best_rows(self):
        # Labels saved on another machine, whose backend differs by 1e-5 relative in
        # chi-squared. The design rows come first in the file and have chi-squared ~1e5:
        # there the difference is ~1, a hundred times the absolute tolerance, although the
        # rows near the peak agree to 1e-4. The audit must check the latter.
        from profiles.workflow import audit_labels, audit_row_ids

        sigma = 0.003

        class Narrow(Pool):
            def __call__(self, x):
                return -0.5 * float(np.asarray(x) @ x) / sigma**2

        rng = np.random.default_rng(3)
        x = np.vstack([rng.uniform(-1, 1, (40, 2)), rng.normal(0, sigma, (10, 2))])
        x = x.astype(np.float32).astype(float)  # as the workflow reads the training CSV
        here = np.array([Narrow()(row) for row in x])
        elsewhere = here * (1 + 1e-5)
        original = np.column_stack([x, elsewhere.astype(np.float32).astype(float)])
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        self.assertGreater(-2 * here[0], 1e4)

        c = ProfileConfig(parameters=["x"])
        with tempfile.TemporaryDirectory() as d:
            ledger = WorkflowLedger(Narrow(), bounds, d, 10)
            best = audit_labels(ledger, original, bounds, c)
            self.assertEqual(aggregate_counts(d)["backend_attempts"], 3)
        # The design row would not have passed: there the backends differ by ~1 in chi-squared.
        self.assertGreater(2 * abs(here[0] - original[0, -1]), 0.1)
        np.testing.assert_array_equal(best["indices"], np.sort(np.argsort(-original[:, -1])[:3]))
        self.assertTrue(np.all(best["indices"] >= 40))
        self.assertTrue(best["passed"])
        self.assertLess(max(best["delta_chi2_errors"]), 1e-3)
        self.assertEqual(best["rows"], "best")
        # A wrong objective is caught at the best rows too: here a constant offset of 1.
        c = ProfileConfig(parameters=["x"])
        with tempfile.TemporaryDirectory() as d:
            shifted = np.column_stack([x, original[:, -1] + 1.0])
            self.assertFalse(audit_labels(WorkflowLedger(Narrow(), bounds, d, 10), shifted, bounds, c)["passed"])

        # The selection breaks ties by file order and never asks for more rows than there are.
        for n, k in [(1, 3), (2, 3), (3, 3), (80, 3), (80, 1), (7, 5), (5, 9)]:
            self.assertEqual(len(audit_row_ids(np.arange(n), k)), min(k, n))
        np.testing.assert_array_equal(audit_row_ids([1.0, 5.0, 5.0, 5.0, 2.0, 5.0], 3), [1, 2, 3])
        np.testing.assert_array_equal(audit_row_ids([3.0, -1.0, 9.0, 0.0], 2), [0, 2])

    def test_quantization(self):
        x = canonical_points([[0.1, 0.2]], [[0.1, 0.9], [0.1, 0.9]], [])
        self.assertTrue(np.all(x >= 0.1))
        np.testing.assert_array_equal(x, x.astype(np.float32).astype(float))
        with self.assertRaises(ValueError):
            canonical_points([[0.4, 0.5], [0.4 + 1e-12, 0.5]], [[0, 1], [0, 1]], [])
        with self.assertRaises(ValueError):
            canonical_points(x, [[0.1, 0.9], [0.1, 0.9]], x)

    def test_batch_resume_budget(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = WorkflowLedger(Pool(), [[-1, 1], [-1, 1]], d, 3)
            ledger.stage = "batch"
            np.testing.assert_allclose(
                ledger.evaluate_many([[0, 0], [0.5, 0], [0.5, 0]]), [0, -0.25, -0.25]
            )
            self.assertEqual(ledger.counts["backend_calls"], 2)
            ledger2 = WorkflowLedger(Pool(), [[-1, 1], [-1, 1]], d, 3)
            self.assertEqual(ledger2([0.5, 0]), -0.25)
            self.assertEqual(ledger2.counts["backend_calls"], 0)
            ledger2([1, 0])
            from profiles.evaluation import CallBudgetExceeded

            with self.assertRaises(CallBudgetExceeded):
                ledger2([-0.5, 0])
            self.assertEqual(aggregate_counts(d)["backend_attempts"], 3)
            ledger3 = WorkflowLedger(Pool(), [[-1, 1], [-1, 1]], d, 3)
            self.assertEqual(ledger3([1, 0]), -1.0)
            with self.assertRaises(CallBudgetExceeded):
                ledger3([0, 1])

    def test_failed_batch_drains_and_journals_all_attempts(self):
        class FailingPool(Pool):
            def submit(self, x):
                f = Future()
                if x[0] > 0:
                    f.set_exception(RuntimeError("backend failed"))
                else:
                    f.set_result(0.0)
                return f

        with tempfile.TemporaryDirectory() as d:
            ledger = WorkflowLedger(FailingPool(), [[-1, 1], [-1, 1]], d, 10)
            with self.assertRaises(RuntimeError):
                ledger.evaluate_many([[0.5, 0], [-0.5, 0]])
            counts = aggregate_counts(d)
            self.assertEqual(counts["backend_attempts"], 2)
            self.assertEqual(counts["events"]["exception"], 1)
            self.assertEqual(counts["events"]["return"], 1)
            self.assertEqual(counts["unclosed_attempts"], 0)

    def test_a_refused_point_is_minus_infinity_and_asked_again_only_on_a_resume(self):
        # The return the cluster backend gives for an error. The callers decide what it means.
        asked = []

        class RefusingPool(Pool):
            def submit(self, x):
                asked.append(tuple(x))
                f = Future()
                f.set_result(-np.inf if x[0] > 0 else 0.0)
                return f

        with tempfile.TemporaryDirectory() as d:
            ledger = WorkflowLedger(RefusingPool(), [[-1, 1], [-1, 1]], d, 10)
            np.testing.assert_array_equal(ledger.evaluate_many([[0.5, 0], [-0.5, 0], [0.5, 0]]), [-np.inf, 0.0, -np.inf])
            self.assertEqual(aggregate_counts(d)["backend_attempts"], 2)     # the same point once in a batch
            again = WorkflowLedger(RefusingPool(), [[-1, 1], [-1, 1]], d, 10)
            np.testing.assert_array_equal(again.evaluate_many([[0.5, 0], [-0.5, 0]]), [-np.inf, 0.0])
            self.assertEqual(asked, [(0.5, 0.0), (-0.5, 0.0), (0.5, 0.0)])       # the finite one came from the journal

    def test_a_backend_that_returns_nan_is_refused_and_one_that_raises_still_raises(self):
        class NanPool(Pool):
            def submit(self, x):
                f = Future()
                f.set_result(float("nan"))
                return f

        with tempfile.TemporaryDirectory() as d:
            np.testing.assert_array_equal(WorkflowLedger(NanPool(), [[-1, 1], [-1, 1]], d, 10).evaluate_many([[0.5, 0]]), [-np.inf])

    def test_record_refused_lists_points_and_stops_above_half(self):
        from profiles.evaluation import record_refused

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            points = np.arange(8.0).reshape(4, 2)
            np.testing.assert_array_equal(record_refused(d, "s", points, [0.0, 1.0, 2.0, 3.0]), [True] * 4)
            self.assertFalse((Path(d) / "refused_points.json").exists())       # nothing refused, no file
            np.testing.assert_array_equal(record_refused(d, "s", points, [0.0, -np.inf, 2.0, -np.inf]), [True, False, True, False])
            record_refused(d, "s", points, [0.0, -np.inf, 2.0, 3.0])            # a resume does not list a point twice
            self.assertEqual(json.loads((Path(d) / "refused_points.json").read_text()), dict(s=[[2.0, 3.0], [6.0, 7.0]]))
            with self.assertRaisesRegex(RuntimeError, "refused 3 of the batch's 4 points"):
                record_refused(d, "t", points, [0.0, -np.inf, -np.inf, -np.inf])
            self.assertIn("t", json.loads((Path(d) / "refused_points.json").read_text()))

    def test_validation_pairs_are_renumbered_without_refused_rows(self):
        from profiles.training import without_refused

        rows = lambda *values: np.column_stack([np.zeros(len(values)), values])
        g, r, profile, nuisance = without_refused(
            rows(1.0, -np.inf, 3.0), rows(0.0, 1.0, -np.inf, 3.0, 4.0, 5.0),
            [[0, 3], [0, 2]], [[0, 1], [3, 4], [2, 5]])
        np.testing.assert_array_equal(g[:, -1], [1.0, 3.0])
        np.testing.assert_array_equal(r[:, -1], [0.0, 1.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(profile, [[0, 2]])                         # 3 is now 2; the pair with 2 has gone
        np.testing.assert_array_equal(nuisance, [[0, 1], [2, 3]])
        # Nothing refused: the arrays as they were.
        g, r, profile, nuisance = without_refused(rows(1.0), rows(0.0, 1.0, 2.0), [[0, 1]], [[1, 2]])
        np.testing.assert_array_equal(profile, [[0, 1]])
        np.testing.assert_array_equal(nuisance, [[1, 2]])
        with self.assertRaisesRegex(RuntimeError, "none of profile_pairs is left"):
            without_refused(rows(1.0), rows(-np.inf, 1.0, 2.0), [[0, 1], [0, 2]], [[1, 2]])

    def test_the_label_audit_stops_on_a_refused_row_and_its_record_can_be_written(self):
        from profiles.io import save
        from profiles.workflow import audit_labels

        ledger = SimpleNamespace(stage=None, evaluate_many=lambda points: np.array([-np.inf, -1.0]))
        original = np.array([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0]])
        result = audit_labels(ledger, original, np.array([[-1.0, 1.0]] * 2), SimpleNamespace(audit_points=2, audit_tolerance=0.01))
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["refused_rows"]), 1)
        self.assertEqual(sum(e is None for e in result["delta_chi2_errors"]), 1)
        with tempfile.TemporaryDirectory() as d:
            save(Path(d) / "failed_audit.json", result)

    def test_small_targeted_budget_is_exact(self):
        from dataset.profile_augmentation import allocate_counts

        counts = allocate_counts(
            [dict(weight=1.0) for _ in range(20)], 3, np.random.default_rng(1)
        )
        self.assertEqual(sum(counts), 3)
        self.assertEqual(sum(counts > 0), 3)

    def test_submission_failure_drains_prior_work(self):
        class BrokenPool(Pool):
            def submit(self, x):
                if x[0] > 0:
                    raise RuntimeError("executor unavailable")
                return super().submit(x)

        with tempfile.TemporaryDirectory() as d:
            ledger = WorkflowLedger(BrokenPool(), [[-1, 1], [-1, 1]], d, 10)
            with self.assertRaises(RuntimeError):
                ledger.evaluate_many([[-0.5, 0], [0.5, 0], [0, 0]])
            counts = aggregate_counts(d)
            self.assertEqual(counts["backend_attempts"], 2)
            self.assertEqual(counts["events"]["exception"], 1)
            self.assertEqual(counts["events"]["return"], 1)
            self.assertEqual(counts["unclosed_attempts"], 0)


def make_toy_run(root):
    """A saved two-parameter CLiENT iteration whose truth is a unit Gaussian."""
    import tensorflow as tf
    from likelihood.cobaya import CobayaLikelihood
    from likelihood.surrogate import SurrogateMetadata

    run = root / "run"
    for name in ["likelihood_input", "trained_models", "training_data"]:
        (run / name).mkdir(parents=True)
    likelihood = dict(
        likelihood={
            "gaussian_mixture": dict(
                means=[[0.0, 0.0]],
                covs=[[[1.0, 0.0], [0.0, 1.0]]],
                input_params_prefix="x",
            )
        },
        params={f"x_{i}": dict(prior=dict(min=-4.0, max=4.0)) for i in range(2)},
    )
    path = run / "likelihood_input/toy.yaml"
    path.write_text(yaml.safe_dump(likelihood, sort_keys=False))
    (run / "run.yaml").write_text(
        yaml.safe_dump(dict(likelihood=dict(wrapper="cobaya", input="input/toy.yaml")))
    )
    truth = CobayaLikelihood(str(path))
    SurrogateMetadata.from_likelihood(truth).save(run / "metadata.json")
    x = np.random.default_rng(91).uniform(-3, 3, (80, 2)).astype(np.float32)
    y = np.array([truth.loglkl(p) for p in x])
    peak = truth.loglkl([0, 0])
    truth.cobaya_model.close()
    np.savetxt(
        run / "training_data/data_it_2.csv",
        np.column_stack([x, y]),
        delimiter=",",
        header="x_0,x_1,loglkl",
        comments="",
    )
    inputs = tf.keras.Input((2,))
    norm = tf.keras.layers.Normalization(mean=[0.0, 0.0], variance=[1.0, 1.0])(inputs)
    hidden = tf.keras.layers.Dense(2)(norm)
    q = tf.keras.layers.Dot(axes=1)([hidden, hidden])
    output = tf.keras.layers.Dense(1)(q)
    model = tf.keras.Model(inputs, output)
    model.layers[2].set_weights([np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)])
    model.layers[-1].set_weights([np.array([[-0.5]], np.float32), np.array([peak], np.float32)])
    model.save(run / "trained_models/model_it_2.keras")
    return run


def toy_config(**overrides):
    c = ProfileConfig(
        parameters=["x_0", "x_1"],
        retrain=True,
        added_points=24,
        workers=2,
        first_epochs=2,
        polish_epochs=2,
        validation_global=8,
        validation_nodes=3,
        scan_points=9,
        profile_points=5,
        ranges={"x_0": [-2.0, 2.0], "x_1": "prior"},
        total_call_budget=3000,
        **overrides,
    )
    c.validate()
    return c


class ProfilesIntegrationTests(unittest.TestCase):
    def check_run(self, run, out, c):
        from profiles.workflow import sha

        original_hash = sha(run / "trained_models/model_it_2.keras")
        run_workflow(run, out, c, dry_run=True)
        self.assertFalse(out.exists())
        run_workflow(run, out, c)
        data = np.load(out / "augmented_data.npy")
        self.assertEqual(len(data), 104)
        val = dict(np.load(out / "validation.npz"))
        self.assertEqual(len(val["global_data"]) + len(val["ridge_data"]), 26)
        self.assertFalse(
            set(map(tuple, data[:, :-1]))
            & set(map(tuple, np.vstack([val["global_data"][:, :-1], val["ridge_data"][:, :-1]])))
        )
        records = json.loads((out / "true_profiles.json").read_text())
        for n, r in records["profiles"].items():
            self.assertEqual(len(r["grid"]), 5)
            np.testing.assert_allclose(
                2 * (records["best_logl"] - np.array(r["logl"])),
                np.array(r["grid"]) ** 2,
                atol=0.03,
            )
            self.assertTrue(all(d.get("flag") == 0 for d in r["optimization"]))
        self.assertTrue((out / "direct_profiles.json").exists())
        self.assertTrue((out / "profiles.png").exists())
        self.assertTrue((out / "profiles.csv").exists())
        self.assertEqual(sha(run / "trained_models/model_it_2.keras"), original_hash)
        # Resume after the last stage re-evaluates nothing.
        before = aggregate_counts(out / "calls")["backend_attempts"]
        (out / "complete.json").unlink()
        run_workflow(run, out, c, resume=True)
        self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], before)
        return records

    def test_a_retraining_study_with_part_of_the_box_refused(self):
        # A real study through the backend processes. x_0 > 1.5 is refused: design and validation
        # points there are left out, and the nodes at x_0 = 2 and beyond have no value.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "study"
            path = run / "likelihood_input/toy.yaml"
            info = yaml.safe_load(path.read_text())
            info["prior"] = dict(refuses="lambda x_0: -float('inf') if x_0 > 1.5 else 0.0")
            path.write_text(yaml.safe_dump(info, sort_keys=False))
            c = toy_config()
            with contextlib.redirect_stdout(io.StringIO()):
                run_workflow(run, out, c)
            listed = json.loads((out / "refused_points.json").read_text())
            refused = [x for points in listed.values() for x in points]
            self.assertGreater(len(refused), 0)
            self.assertTrue(all(x[0] > 1.5 for x in refused), refused)
            data = np.load(out / "augmented_data.npy")
            self.assertTrue(np.isfinite(data).all())
            self.assertEqual(len(data), 104 - sum(k.startswith("augmentation") and len(v) for k, v in listed.items()))
            val = dict(np.load(out / "validation.npz"))
            for key in ("global_data", "ridge_data"):
                self.assertTrue(np.isfinite(val[key]).all())
            for key in ("profile_pairs", "nuisance_pairs"):
                self.assertLess(val[key].max(), len(val["ridge_data"]))
            self.assertTrue((out / "training/training.json").exists())
            self.assertTrue((out / "complete.json").exists())
            warnings = json.loads((out / "diagnostics.json").read_text())["warnings"]
            self.assertTrue(any("design or validation point(s)" in w for w in warnings), warnings)
            # A resume spends nothing more.
            before = aggregate_counts(out / "calls")["backend_attempts"]
            (out / "complete.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                run_workflow(run, out, c, resume=True)
            self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], before)

    def test_a_backend_that_refuses_the_best_saved_rows_stops_at_the_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "study"
            path = run / "likelihood_input/toy.yaml"
            info = yaml.safe_load(path.read_text())
            info["prior"] = dict(refuses="lambda x_0: -float('inf')")
            path.write_text(yaml.safe_dump(info, sort_keys=False))
            with self.assertRaisesRegex(RuntimeError, "refuses saved training row"), contextlib.redirect_stdout(io.StringIO()):
                run_workflow(run, out, ProfileConfig(parameters=["x_0"], workers=1, profile_points=3, ranges={"x_0": [-1.0, 1.0]}))
            self.assertIn("refused_rows", json.loads((out / "failed_audit.json").read_text()))

    def test_saved_network_default_without_training_and_resume(self):
        from profiles.workflow import sha
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "profiles"
            c = ProfileConfig(parameters=["x_0"], ranges={"x_0": [-2., 2.]},
                              profile_points=3, scan_points=5, workers=2)
            self.assertFalse(c.retrain)
            self.assertEqual(c.added_points, 0)
            before_hash = sha(run / "trained_models/model_it_2.keras")
            run_workflow(run, out, c)
            for name in ["training", "validation.npz", "augmented_data.npy",
                         "baseline_profiles.json"]:  # every range explicit: no full-range scan
                self.assertFalse((out / name).exists())
            ranges = json.loads((out / "ranges.json").read_text())
            self.assertEqual(ranges["x_0"]["actual"], [-2.0, 2.0])
            self.assertIsNone(ranges["x_0"]["surrogate_1sigma_crossings"])
            result = json.loads((out / "true_profiles.json").read_text())
            row = result["profiles"]["x_0"]
            np.testing.assert_allclose(2 * (result["best_logl"] - np.array(row["logl"])),
                                       np.array(row["grid"])**2, atol=.03)
            self.assertFalse(json.loads((out / "complete.json").read_text())["training_performed"])
            self.assertEqual(sha(run / "trained_models/model_it_2.keras"), before_hash)
            before = aggregate_counts(out / "calls")["backend_attempts"]
            (out / "complete.json").unlink()
            run_workflow(run, out, c, resume=True)
            self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], before)

    def test_a_study_is_resumed_after_its_run_was_moved_and_not_after_it_changed(self):
        # The manifest held the run's files by their absolute paths, and a study could not be
        # resumed once the run directory had been moved or synced elsewhere. What a study
        # depends on is what the files hold.
        import contextlib, io, shutil

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "profiles"
            c = ProfileConfig(parameters=["x_0"], ranges={"x_0": [-2., 2.]}, profile_points=3, scan_points=5, workers=2)
            run_workflow(run, out, c)
            before = aggregate_counts(out / "calls")["backend_attempts"]
            manifest = (out / "manifest.json").read_text()
            moved = root / "elsewhere" / "the_run"
            moved.parent.mkdir()
            shutil.move(str(run), str(moved))
            (out / "complete.json").unlink()
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                run_workflow(moved, out, c, resume=True)
            self.assertIn("with the same files", said.getvalue())
            self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], before)      # nothing evaluated again
            self.assertEqual((out / "manifest.json").read_text(), manifest)                    # the record is as it was
            # the same place, another network: refused as before
            (out / "complete.json").unlink()
            data = moved / "training_data/data_it_2.csv"
            data.write_text(data.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "source/configuration changed"):
                run_workflow(moved, out, c, resume=True)

    def test_a_refusal_says_where_it_was_raised(self):
        import contextlib, io
        from unittest import mock
        from profiles.__main__ import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_toy_run(root)
            settings = root / "profiles.yaml"
            settings.write_text(yaml.safe_dump(dict(parameters=["x_0"], ranges={"x_0": [-2.0, 2.0]}, profile_points=3)))
            arguments = ["profiles", str(run), "-o", str(run), "--config", str(settings)]
            said = io.StringIO()
            with mock.patch("sys.argv", arguments), contextlib.redirect_stderr(said), self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertRegex(said.getvalue(), r"output cannot be the source run or its parent  \[raised in workflow\.py:\d+\]")
            with mock.patch("sys.argv", arguments), mock.patch.dict(os.environ, {"CLIENT_PROFILE_TRACEBACK": "1"}), \
                    self.assertRaisesRegex(ValueError, "output cannot be the source run"):
                main()

    def test_an_anchor_that_did_not_converge_is_in_the_report(self):
        from profiles.reporting import anchor_warning

        self.assertEqual(anchor_warning(dict(anchor_diagnostics=dict(flag=0, reason="converged"))), [])
        self.assertEqual(anchor_warning(dict(anchor_diagnostics=None)), [])
        self.assertEqual(anchor_warning({}), [])                              # a study of a method without the record
        (said,) = anchor_warning(dict(anchor_diagnostics=dict(flag=1, reason="budget")))
        self.assertIn("did not converge (solver flag 1, budget)", said)
        self.assertIn("normalization may still rise", said)

    def test_label_audit_rows_in_the_workflow(self):
        # The saved labels differ from this backend in proportion to their depth below the
        # peak, as labels made on another machine do. Checked at rows spread over the file
        # the study would stop; it checks the best rows and runs.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "profiles"
            csv = run / "training_data/data_it_2.csv"
            table = np.loadtxt(csv, delimiter=",", skiprows=1)
            depth = table[:, -1].max() - table[:, -1]
            table[:, -1] -= 0.01 * depth
            np.savetxt(csv, table, delimiter=",", header="x_0,x_1,loglkl", comments="")
            best = np.sort(np.argsort(-table[:, -1])[:3])
            self.assertGreater(0.02 * depth[0], 0.05)        # row 0: five times the tolerance
            self.assertLess(0.02 * depth[best].max(), 0.008)  # the best rows: inside it

            def config(**overrides):
                return ProfileConfig(parameters=["x_0"], ranges={"x_0": [-2., 2.]},
                                     profile_points=3, scan_points=5, workers=2, **overrides)

            run_workflow(run, out, config())
            audit = json.loads((out / "audit.json").read_text())
            self.assertEqual((audit["rows"], audit["indices"], audit["passed"]), ("best", best.tolist(), True))
            self.assertTrue((out / "complete.json").exists())

    def test_full_range_scan_only_when_it_chooses_a_range(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_toy_run(root)
            # No explicit range: the baseline scan sets it from the surrogate 1-sigma crossings.
            auto = ProfileConfig(parameters=["x_0"], profile_points=3, scan_points=5, workers=2)
            run_workflow(run, root / "auto", auto)
            self.assertTrue((root / "auto/baseline_profiles.json").exists())
            ranges = json.loads((root / "auto/ranges.json").read_text())
            self.assertEqual(ranges["x_0"]["kind"], "surrogate 1-sigma widths")
            self.assertEqual(len(ranges["x_0"]["surrogate_1sigma_crossings"]), 2)
            # A study that already has a baseline scan keeps using it when resumed.
            out = root / "explicit"
            explicit = ProfileConfig(parameters=["x_0"], ranges={"x_0": [-2., 2.]},
                                     profile_points=3, scan_points=5, workers=2)
            run_workflow(run, out, explicit)
            first = json.loads((out / "true_profiles.json").read_text())
            before = aggregate_counts(out / "calls")["backend_attempts"]
            shutil.copy(root / "auto/baseline_profiles.json", out / "baseline_profiles.json")
            (out / "complete.json").unlink()
            run_workflow(run, out, explicit, resume=True)
            self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], before)
            self.assertEqual(json.loads((out / "true_profiles.json").read_text()), first)
            ranges = json.loads((out / "ranges.json").read_text())
            self.assertEqual(ranges["x_0"]["actual"], [-2.0, 2.0])
            self.assertEqual(len(ranges["x_0"]["surrogate_1sigma_crossings"]), 2)

    def test_default_hybrid_surrogate_and_corrected_true_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "profiles"
            c = toy_config()
            records = self.check_run(run, out, c)
            self.assertEqual(records["method"], "surrogate-corrected residual Newton")
            surrogate = json.loads((out / "surrogate_profiles.json").read_text())
            self.assertTrue(surrogate["optimization_quality"]["converged"])
            self.assertEqual(surrogate["optimization_quality"]["method"],
                             "hybrid basin hopping + projected BFGS")
            c.profile_points = 7
            with self.assertRaises(ValueError):
                run_workflow(run, out, c, resume=True)

    def test_legacy_scan_with_the_newton_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, out = make_toy_run(root), root / "profiles"
            self.check_run(run, out, toy_config(surrogate_optimizer="legacy"))
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["configuration"]["surrogate_optimizer"], "legacy")


if __name__ == "__main__":
    unittest.main()
