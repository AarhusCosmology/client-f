"""The true check (hpc/analysis/true_check.py): a network's entry when
the likelihood refused some of its points or failed at them, and the guard that ends an MPI job
when one rank raises. The check itself, with a real likelihood, runs on the cluster.

Also the launcher's refusal of a prior.n_sigma (hpc/launch.py). launch.py starts CLiENT
when it is imported, so its function is read out of the file and nothing else of it is run."""

import ast
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

KIT = Path(__file__).resolve().parents[1] / "hpc"
sys.path.insert(0, str(KIT / "analysis"))
import true_check  # noqa: E402

NAMES = ["a", "b", "c"]
BEST = -1000.0
REFUSED_VALUE = -30700.0


def sample(n=400, seed=2):
    """Posterior points, the network's values there and true values with a spread of 0.3."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3)).astype(np.float32)
    network = BEST - 0.5 * np.sum(x.astype(float) ** 2, axis=1)
    true = network + 0.1 + 0.3 * rng.standard_normal(n)
    return x, network, true


def entry(x, network, true, *values):
    return true_check.network_entry(7, x, network, true, BEST, NAMES, true_check.refused_points(true, *values),
                                    values[0] if values else None)


class RefusedPoints(unittest.TestCase):
    def test_minus_infinity_and_the_refused_value(self):
        true = np.array([-1001.5, -np.inf, REFUSED_VALUE, np.nan, np.inf, REFUSED_VALUE + 1e-6, -250000.0])
        np.testing.assert_array_equal(true_check.refused_points(true), [0, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(true_check.refused_points(true, None), [0, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(true_check.refused_points(true, REFUSED_VALUE), [0, 1, 1, 0, 0, 0, 0])
        # the run's value and another one this job's environment gave: a point with either
        np.testing.assert_array_equal(true_check.refused_points(true, REFUSED_VALUE, -250000.0), [0, 1, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(true_check.refused_points(true, REFUSED_VALUE, None), [0, 1, 1, 0, 0, 0, 0])


class Entry(unittest.TestCase):
    def test_without_refused_points(self):
        x, network, true = sample()
        found = entry(x, network, true, REFUSED_VALUE)
        d = true - network
        w = np.exp(d - d.max())
        self.assertEqual((found["iteration"], found["points"], found["not_finite"], found["refused"]), (7, 400, 0, 0))
        self.assertEqual(found["refused_value"], REFUSED_VALUE)
        self.assertAlmostEqual(found["truth_minus_network"]["mean"], d.mean(), places=12)
        self.assertAlmostEqual(found["truth_minus_network"]["sd"], d.std(), places=12)
        self.assertAlmostEqual(found["truth_minus_network"]["rms"], np.sqrt(np.mean(d ** 2)), places=12)
        self.assertAlmostEqual(found["truth_minus_network"]["largest"], np.abs(d).max(), places=12)
        self.assertAlmostEqual(found["effective_points"], w.sum() ** 2 / np.sum(w ** 2), places=9)
        self.assertAlmostEqual(found["effective_points"] / 400, np.exp(-0.09), delta=0.03)     # exp(-s^2) for a spread s
        self.assertEqual(found["best_true_here"], true.max())
        self.assertEqual(found["above_best_training_value"], int(np.sum(true > BEST)))
        self.assertAlmostEqual(found["median_depth_below_best"], np.median(BEST - true), places=12)
        self.assertEqual(list(found["corrected_shift"]), NAMES)
        self.assertEqual(found["largest_shift"], max(abs(v) for v in found["corrected_shift"].values()))
        # what the readers of true_check.json take from an entry is all there, and it is JSON
        for key in ("iteration", "points", "not_finite", "truth_minus_network", "effective_points", "median_depth_below_best",
                    "above_best_training_value", "best_true_here", "corrected_shift", "corrected_width", "largest_shift",
                    "largest_shift_parameter"):
            self.assertIn(key, found)
        self.assertEqual(json.loads(json.dumps(found)), found)
        self.assertIsNone(entry(x, network, true)["refused_value"])                           # a run without one

    def test_refused_and_failed_points_are_counted_and_left_out(self):
        x, network, true = sample()
        spoiled = true.copy()
        spoiled[[3, 50, 51, 200]] = REFUSED_VALUE          # refused, CLIENT_REFUSED_VALUE in the job's environment
        spoiled[[10, 300]] = -np.inf                       # refused, without it
        spoiled[[77]] = np.nan                             # an evaluation that failed
        spoiled[[78]] = np.inf
        kept = np.ones(len(true), bool)
        kept[[3, 50, 51, 200, 10, 300, 77, 78]] = False
        found = entry(x, network, spoiled, REFUSED_VALUE)
        clean = entry(x[kept], network[kept], true[kept], REFUSED_VALUE)
        self.assertEqual((found["points"], found["not_finite"], found["refused"]), (400, 8, 6))
        self.assertEqual(found["points"] - found["not_finite"], clean["points"])       # the points behind the numbers
        for key in ("truth_minus_network", "median_depth_below_best", "above_best_training_value", "best_true_here"):
            self.assertEqual(found[key], clean[key], key)        # every number is that of the points with a true value
        self.assertLess(found["truth_minus_network"]["largest"], 2.0)
        for key in ("effective_points", "corrected_shift", "corrected_width", "largest_shift"):
            self.assertIsNone(found[key])
        self.assertEqual(found["failed"], 2)

    def test_refusal_changes_the_full_surrogate_distribution(self):
        x = np.linspace(-1, 1, 1000)[:, None]
        network = np.zeros(len(x))
        truth = np.where(x[:, 0] >= 0, 0., -np.inf)
        found = true_check.network_entry(1, x, network, truth, 0., ["x"], np.isneginf(truth))
        self.assertAlmostEqual(found["corrected_shift"]["x"], np.sqrt(3) / 2, places=5)
        self.assertAlmostEqual(found["corrected_width"]["x"], -.5, places=5)
        self.assertAlmostEqual(found["effective_points"], 500.)
        self.assertIsNone(found["correction_unavailable"])

    def test_the_same_numbers_whichever_way_the_refused_points_came_back(self):
        # with CLIENT_REFUSED_VALUE in the true check's environment they carry the value, without it they are -inf
        x, network, true = sample()
        with_value, without = true.copy(), true.copy()
        with_value[[3, 50, 51, 200]] = REFUSED_VALUE
        without[[3, 50, 51, 200]] = -np.inf
        self.assertEqual(entry(x, network, with_value, REFUSED_VALUE), entry(x, network, without, REFUSED_VALUE))

    def test_what_the_refused_value_did_as_a_true_value(self):
        # the fault this guards against: the value taken as a likelihood, 29 700 below the network
        x, network, true = sample()
        true[[3, 50, 51, 200]] = REFUSED_VALUE
        taken = entry(x, network, true)                     # the value not known: only -inf is a refusal
        self.assertEqual((taken["not_finite"], taken["refused"]), (0, 0))
        self.assertGreater(taken["truth_minus_network"]["largest"], 29000.0)
        self.assertGreater(taken["truth_minus_network"]["rms"], 2900.0)
        known = entry(x, network, true, REFUSED_VALUE)
        self.assertEqual((known["not_finite"], known["refused"]), (4, 4))
        self.assertLess(known["truth_minus_network"]["rms"], 0.4)

    def test_a_network_without_any_true_value_is_recorded_and_says_so(self):
        x, network, true = sample(20)
        true[:15], true[15:] = REFUSED_VALUE, np.nan
        found = entry(x, network, true, REFUSED_VALUE)
        self.assertIn("none of its 20 points has a true value (15 refused, 5 NaN or +inf)", found["correction_unavailable"])
        self.assertIsNone(found["effective_points"])
        self.assertIsNone(found["truth_minus_network"])
        self.assertEqual((found["points"], found["not_finite"], found["refused"], found["failed"]), (20, 20, 15, 5))
        json.dumps(found, allow_nan=False)
        self.assertIn("0 of 20 with a true value (15 refused, 5 NaN or +inf)", true_check.report_line(found, 2))

class ReportLine(unittest.TestCase):
    """The line the job prints for every network. Its pieces once stood side by side without a
    +, which calls a string: the job stopped at the first network with a correction, after
    the true evaluations and before true_check.json, and no test ran the line."""

    def test_a_network_with_a_correction(self):
        x, network, true = sample()
        line = true_check.report_line(entry(x, network, true), ndim=3)
        self.assertRegex(line, r"^network 7: 400 of 400 with a true value; truth minus network: mean \+0\.\d\d, spread 0\.\d\d, "
                               r"rms 0\.\d\d, largest \d\.\d; \d+ effective points; largest corrected mean shift "
                               r"\d\.\d\d sigma \([abc]\); median depth below the best training value \d\.\d \(n / 2 = 1\.5\)$")

    def test_refused_points_failed_evaluations_and_no_width(self):
        x, network, true = sample()
        true[:30], true[30:50] = REFUSED_VALUE, -np.inf
        line = true_check.report_line(entry(x, network, true, REFUSED_VALUE), ndim=3)
        self.assertIn("350 of 400 with a true value (50 refused, 0 NaN or +inf)", line)
        self.assertIn("effective points; largest corrected mean shift", line)
        true[60:62] = np.nan
        line = true_check.report_line(entry(x, network, true, REFUSED_VALUE), ndim=3)
        self.assertIn("348 of 400 with a true value (50 refused, 2 NaN or +inf)", line)
        self.assertIn("correction unavailable: 2 unresolved true evaluations", line)
        x, network, true = sample()
        line = true_check.report_line(entry(np.zeros_like(x), network, true), ndim=3)      # every point the same: no width
        self.assertIn("effective points; no corrected mean shift (zero widths)", line)

    def test_main_prints_it(self):
        source = (KIT / "analysis" / "true_check.py").read_text()
        self.assertIn("print(report_line(entry, run.ndim), flush=True)", source)


class ChainLine(unittest.TestCase):
    """What the job says of a network's chain before its true evaluations. A chain whose own
    record marks the weight of a cell as not reliable gave its true values and no correction,
    and the log said so only after the evaluations."""

    def test_a_chain_that_can_carry_the_corrections(self):
        self.assertEqual(true_check.chain_line(7, 200, 85504), "network 7: 200 of its 85504 posterior samples")
        self.assertEqual(true_check.chain_line(7, 200, 85504, None), true_check.chain_line(7, 200, 85504, ""))

    def test_a_chain_that_cannot(self):
        line = true_check.chain_line(7, 200, 85504, "unreliable region-cell mixture weights (cells [1])")
        first, warning = line.split("\n")
        self.assertEqual(first, "network 7: 200 of its 85504 posterior samples")
        self.assertRegex(warning, r"^  WARNING, network 7: unreliable region-cell mixture weights \(cells \[1\]\)\. Its true values "
                                  r"and the pointwise errors will be saved; the corrected moments, the effective points and the "
                                  r"evidence correction will not\.")

    def test_main_prints_it_before_the_evaluations(self):
        source = (KIT / "analysis" / "true_check.py").read_text()
        said = "print(chain_line(it, len(x), len(samples), sampling_correction_problem(info)), flush=True)"
        self.assertLess(source.index(said), source.index("broadcast_and_evaluate(samples=points"))


class Box(unittest.TestCase):
    def test_a_box_narrower_than_the_run_s_is_named(self):
        # A domain file made again, narrower, after the run: the samples beyond it got -inf
        # from the prior and were counted as refused points.
        names, lower, upper = ["a", "b", "c"], np.array([0.0, -1.0, 10.0]), np.array([1.0, 1.0, 20.0])
        same = dict(a=(0.0, 1.0), b=(-1.0, 1.0), c=(10.0, 20.0))
        self.assertEqual(true_check.narrower_than_the_run(same, names, lower, upper), [])
        self.assertEqual(true_check.narrower_than_the_run(dict(same, a=(-5.0, 5.0), b=(None, None), c=(10.0, None)),
                                                          names, lower, upper), [])       # wider, or unbounded
        self.assertEqual(true_check.narrower_than_the_run(dict(same, b=(-0.5, 1.0), c=(10.0, 19.0)), names, lower, upper), ["b", "c"])
        self.assertEqual(true_check.narrower_than_the_run(dict(same, a=(0.0, 1.0 - 1e-13)), names, lower, upper), [])   # rounding

    def test_main_checks_it_before_the_evaluations(self):
        source = (KIT / "analysis" / "true_check.py").read_text()
        self.assertLess(source.index("narrower_than_the_run(truth.prior_bounds"), source.index("broadcast_and_evaluate(samples=points"))


class Guard(unittest.TestCase):
    """stop_peers_on_error, on a communicator that only records: the real one would end the tests."""

    def run_guarded(self, module, error, size=2):
        aborted = []
        comm = SimpleNamespace(size=size, Abort=aborted.append)

        @module.stop_peers_on_error
        def work():
            if error is not None:
                raise error
            return "done"

        printed = io.StringIO()
        with mock.patch.object(module, "MPI", SimpleNamespace(COMM_WORLD=comm)), contextlib.redirect_stderr(printed):
            try:
                return work(), aborted, printed.getvalue()
            except BaseException as raised:                # SystemExit and KeyboardInterrupt too
                self.assertIs(raised, error)               # the error itself goes on, for a run without MPI
                return None, aborted, printed.getvalue()

    def test_an_error_on_one_rank_stops_the_others(self):
        for error in (RuntimeError("the run's parameters are not the likelihood's"), AssertionError("a check"),
                      KeyboardInterrupt(), SystemExit("--fix x: not a sampled parameter"), SystemExit(2)):
            result, aborted, printed = self.run_guarded(true_check, error)
            self.assertEqual((result, aborted), (None, [1]), repr(error))
            self.assertIn(type(error).__name__, printed)                 # the traceback, before the others are stopped

    def test_an_exit_that_is_not_a_failure_aborts_nothing(self):
        for error in (SystemExit(0), SystemExit(), SystemExit(None)):
            self.assertEqual(self.run_guarded(true_check, error), (None, [], ""))

    def test_nothing_to_stop(self):
        self.assertEqual(self.run_guarded(true_check, None), ("done", [], ""))
        self.assertEqual(self.run_guarded(true_check, RuntimeError("alone"), size=1), (None, [], ""))   # one process: no peers

    def test_main_is_guarded(self):
        self.assertIsNot(getattr(true_check.main, "__wrapped__", None), None)


def launcher_function(name):
    """A function of launch.py by name, compiled on its own: importing launch.py would start CLiENT."""
    tree = ast.parse((KIT / "launch.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(KIT / "launch.py"), "exec"), namespace)
    return namespace[name]


class LauncherSettings(unittest.TestCase):
    def setUp(self):
        self.problem = launcher_function("n_sigma_problem")

    def settings(self, folder, n_sigma, name="client.yaml"):
        file = Path(folder) / name
        file.write_text("likelihood:\n  wrapper: cobaya\nprior:\n  n_samples: 2000\n" + n_sigma + "sampling:\n  temperature: 7\n")
        return file

    def test_a_new_run_with_n_sigma_is_refused_with_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text, shown in (("  n_sigma: 6\n", "6"), ("  n_sigma: 4.5\n", "4.5"), ("  n_sigma: 0\n", "0")):
                file = self.settings(tmp, text)
                message = self.problem(file, "runs/x/domain_bounds.json")
                self.assertIn(f"{file}: prior.n_sigma is {shown};", message)
                self.assertIn("after the launcher has applied the box of runs/x/domain_bounds.json", message)
                self.assertIn("not in the domain file's", message)
                self.assertIn("Set it to null", message)
            for text in ("  n_sigma: null\n", "  n_sigma: ~\n", "  n_sigma:\n", ""):          # null, or not there at all
                self.assertIsNone(self.problem(self.settings(tmp, text), "d.json"), text)

    def test_a_continued_run_is_judged_by_the_settings_in_its_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20261011_120000_run"
            run.mkdir()
            self.assertIsNone(self.problem(run, "d.json"))                    # no settings: CLiENT says so itself
            self.settings(run, "  n_sigma: null\n")
            self.assertIsNone(self.problem(run, "d.json"))
            file = self.settings(run, "  n_sigma: 30.0\n")
            message = self.problem(run, "d.json")
            self.assertIn(f"{file}: prior.n_sigma is 30.0;", message)
            self.assertIn("The run was started with it", message)
            self.assertNotIn("Set it to null", message)                       # a run's copy of its settings is not rewritten

    def test_what_is_not_a_settings_file_is_left_to_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self.problem(Path(tmp) / "no_such.yaml", "d.json"))
            for name, text in (("broken.yaml", "prior: [1, 2\n"), ("list.yaml", "- 1\n- 2\n"), ("flat.yaml", "prior: 3\n"), ("empty.yaml", "")):
                (Path(tmp) / name).write_text(text)
                self.assertIsNone(self.problem(Path(tmp) / name, "d.json"), name)

    def test_the_kit_s_own_runs_pass(self):
        files = sorted((KIT / "runs").glob("*/client.yaml"))
        self.assertTrue(files)
        for file in files:
            self.assertIsNone(self.problem(file, "d.json"), str(file))

    def test_the_launcher_asks_before_it_imports_client(self):
        body = ast.parse((KIT / "launch.py").read_text()).body
        asked = next(k for k, node in enumerate(body) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                     and getattr(node.value.func, "id", None) == "n_sigma_problem")
        stops = body[asked + 1]
        self.assertTrue(isinstance(stops, ast.If) and isinstance(stops.body[0], ast.Raise))
        imported = next(k for k, node in enumerate(body) if isinstance(node, ast.Import) and node.names[0].name == "client")
        self.assertLess(asked, imported)




class MixtureConsumers(unittest.TestCase):
    """Exercise the producer and both its cached and evidence-summary consumers."""

    def test_old_cache_is_resampled_and_unreliable_corrections_stay_withheld(self):
        import report
        import evidence_summary
        from sampling.provenance import target_identity, save_samples, sampler_identity
        from contextlib import ExitStack

        for reliable in (False, True):
            with self.subTest(reliable=reliable), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'trained_models').mkdir()
                (root / 'trained_models/model_it_4.keras').write_bytes(b'controlled network')
                (root / 'metadata.json').write_text('{}')
                (root / 'analysis').mkdir()
                region = [0, 1]
                target = target_identity(root, 4, SimpleNamespace(param_names=['x'], bounds={'x': [-10., 10.]}),
                                         {'file': 'boost/region_it_4.npz', 'sha256': 'same region', 'edge': 2.})
                cache = root / 'analysis/posterior_it_4.npz'
                # Legacy producer's file: target provenance but no sampling metadata.
                old_x = np.linspace(-3, 3, 200).reshape(-1, 1)
                save_samples(cache, old_x, ['x'], target)
                run = SimpleNamespace(dir=root, names=['x'], ndim=1, lower=np.array([-10.]), upper=np.array([10.]),
                                      models={4: None}, boost=None, best=1., mixtures={}, linear_prior=(),
                                      target_identity=lambda it: target,
                                      surrogate=lambda it, region: SimpleNamespace(region=[0, 1]),
                                      predict=lambda it, x, region: -.5*x[:, 0]**2)
                truth = SimpleNamespace(param_names=['x'], prior_bounds={'x': (-10., 10.)}, refused_value=None,
                                        evaluate_with_derived=lambda x: None, split_evaluations=lambda x: (x, None))
                info = dict(sampler_identity(region), cells=[dict(cell=0, efficiency=.8 if reliable else .001)],
                            unreliable_cells=[] if reliable else [0])

                def sample(*args):
                    run.mixtures[4] = info
                    return np.linspace(-2, 4, 200).reshape(10, 20, 1), .6

                def evaluate(samples, evaluator):
                    return samples, -.5*samples[:, 0]**2 + np.where(samples[:, 0] > 0, np.log(2), 0)

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(true_check, 'MPI', SimpleNamespace(COMM_WORLD=SimpleNamespace(rank=0, size=1))))
                    stack.enter_context(mock.patch('loading.build_in_turn', return_value=truth))
                    stack.enter_context(mock.patch('refused.refuse_logged_errors'))
                    stack.enter_context(mock.patch('refused.read_record', return_value=None))
                    stack.enter_context(mock.patch.object(report, 'Run', return_value=run))
                    stack.enter_context(mock.patch('utils.mpi_utils.broadcast_and_evaluate', side_effect=evaluate))
                    stack.enter_context(mock.patch.object(sys, 'argv', ['true_check.py', str(root), '--cobaya', 'unused', '--networks', '4', '--points', '200']))
                    printed = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    with mock.patch.object(report, 'sample_posterior', side_effect=sample) as sampled:
                        true_check.main.__wrapped__()
                        sampled.assert_called_once()
                    # said before the true evaluations, not only in the result after them
                    said = printed.getvalue()
                    if reliable:
                        self.assertNotIn('WARNING', said)
                    else:
                        self.assertLess(said.index('WARNING, network 4: unreliable region-cell'), said.index('true evaluations in'))
                    self.assertEqual(len(list((root / 'analysis').glob('posterior_it_4.previous-*.npz'))), 1)
                    # A new Run object has no in-memory diagnostics. They must come from the cache.
                    run.mixtures.clear()
                    with mock.patch.object(report, 'sample_posterior', side_effect=AssertionError('must reuse cache')):
                        true_check.main.__wrapped__()
                entry = json.loads((root / 'analysis/true_check.json').read_text())['networks'][0]
                self.assertEqual(entry['sampling'], info)
                self.assertEqual(entry['target'], target)
                self.assertEqual(entry['failed'], 0)
                with np.load(root / 'analysis/true_check_it_4.npz') as saved:
                    self.assertEqual(json.loads(str(saved['sampling_json'].item())), info)
                    self.assertEqual(json.loads(str(saved['target_json'].item())), target)
                    self.assertEqual(len(saved['truth']), 200)
                correction = evidence_summary.correction(root / 'analysis/true_check_it_4.npz', np.random.default_rng(1), repeats=20)
                if reliable:
                    self.assertIsNone(entry['correction_unavailable'])
                    self.assertIsNotNone(entry['corrected_shift'])
                    self.assertIsNotNone(correction['value'])
                else:
                    self.assertIn('unreliable region-cell', entry['correction_unavailable'])
                    self.assertIsNone(entry['corrected_shift'])
                    self.assertIsNone(entry['effective_points'])
                    self.assertIn('correction unavailable', true_check.report_line(entry, 1))
                    self.assertIsNone(correction['value'])
                    self.assertIn('unreliable region-cell', correction['unavailable'])



class LinearPrior(unittest.TestCase):
    """--linear-prior: the check of a posterior with a prior flat in 10^p (report.py --linear-prior). The added
    ln(10) p is the same in the truth and the network, so their difference is the network's error as before; the
    samples are those of the new posterior, and every file of it goes to analysis/linear_prior_<names>/."""

    def test_its_chain_and_results_go_to_their_own_folder(self):
        import report
        from sampling.provenance import sampler_identity, target_identity
        from contextlib import ExitStack

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'trained_models').mkdir()
            (root / 'trained_models/model_it_4.keras').write_bytes(b'controlled network')
            (root / 'metadata.json').write_text('{}')
            target = dict(target_identity(root, 4, SimpleNamespace(param_names=['x'], bounds={'x': [-10., 10.]}), None),
                          linear_prior=['x'])
            run = SimpleNamespace(dir=root, names=['x'], ndim=1, lower=np.array([-10.]), upper=np.array([10.]),
                                  models={4: None}, boost=None, best=1., mixtures={}, linear_prior=('x',),
                                  target_identity=lambda it: target, surrogate=lambda it, region: SimpleNamespace(region=None),
                                  predict=lambda it, x, region: -.5 * x[:, 0] ** 2)
            truth = SimpleNamespace(param_names=['x'], prior_bounds={'x': (-10., 10.)}, refused_value=None,
                                    evaluate_with_derived=lambda x: None, split_evaluations=lambda x: (x, None))

            def evaluate(samples, evaluator):
                return samples, -.5 * samples[:, 0] ** 2 + 0.1

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(true_check, 'MPI', SimpleNamespace(COMM_WORLD=SimpleNamespace(rank=0, size=1))))
                stack.enter_context(mock.patch('loading.build_in_turn', return_value=truth))
                stack.enter_context(mock.patch('refused.refuse_logged_errors'))
                stack.enter_context(mock.patch('refused.read_record', return_value=None))
                made = stack.enter_context(mock.patch.object(report, 'Run', return_value=run))
                stack.enter_context(mock.patch('utils.mpi_utils.broadcast_and_evaluate', side_effect=evaluate))
                stack.enter_context(mock.patch.object(report, 'sample_posterior',
                                                      return_value=(np.linspace(0.5, 3.5, 200).reshape(10, 20, 1), .6)))
                stack.enter_context(mock.patch.object(sys, 'argv', ['true_check.py', str(root), '--cobaya', 'unused',
                                                                    '--networks', '4', '--points', '200', '--linear-prior', 'x']))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                true_check.main.__wrapped__()
            self.assertEqual(made.call_args.kwargs.get('linear_prior'), ['x'])
            folder = root / 'analysis' / 'linear_prior_x'
            self.assertEqual(sorted(p.name for p in (root / 'analysis').iterdir()), ['linear_prior_x'])
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ['posterior_it_4.npz', 'true_check.json', 'true_check_it_4.npz'])
            with np.load(folder / 'true_check_it_4.npz') as saved:
                np.testing.assert_allclose(saved['network'], -.5 * saved['x'][:, 0].astype(float) ** 2, rtol=1e-6)
                np.testing.assert_allclose(saved['truth'] - saved['network'], 0.1, atol=1e-6)
                self.assertEqual(json.loads(str(saved['target_json'].item()))['linear_prior'], ['x'])
            found = json.loads((folder / 'true_check.json').read_text())
            self.assertEqual(found['linear_prior'], ['x'])
            self.assertAlmostEqual(found['networks'][0]['truth_minus_network']['mean'], 0.1, places=5)
            self.assertAlmostEqual(found['networks'][0]['effective_points'], 200.0, places=3)


if __name__ == "__main__":
    unittest.main()
