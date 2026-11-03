"""The centre of a training box (hpc/analysis/find_centre.py) and the box
made from it (make_box.py), on a correlated Gaussian with prior edges. The real likelihood is
fitted on the cluster."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

KIT = Path(__file__).resolve().parents[1] / "hpc"
sys.path.insert(0, str(KIT / "analysis"))
import find_centre  # noqa: E402
import make_box  # noqa: E402

NAMES = ["a", "b", "c", "d"]
TRUTH = np.array([1.0, -2.0, 0.5, 3.0])
SD = np.array([0.1, 0.5, 0.02, 1.0])
CORR = np.array([[1.0, -0.98, 0.0, 0.3], [-0.98, 1.0, 0.0, -0.3], [0.0, 0.0, 1.0, 0.0], [0.3, -0.3, 0.0, 1.0]])
COV = CORR * np.outer(SD, SD)
PRECISION = np.linalg.inv(COV)


def logpost(points, centre=TRUTH):
    d = np.atleast_2d(points) - centre
    return -0.5 * np.einsum("ij,jk,ik->i", d, PRECISION, d)


class Inputs(unittest.TestCase):
    def test_covmat_and_priors(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "x.covmat"
            file.write_text("# " + " ".join(NAMES) + "\n" + "\n".join(" ".join(f"{v:.12e}" for v in row) for row in COV) + "\n")
            names, matrix = find_centre.read_covmat(file)
        self.assertEqual(names, NAMES)
        np.testing.assert_allclose(matrix, COV, rtol=1e-10)
        self.assertEqual(find_centre.prior_of({"prior": {"min": 0, "max": 2}}), ("flat", 0, 2, None, None))
        self.assertEqual(find_centre.prior_of({"prior": {"dist": "norm", "loc": 1.0, "scale": 0.1}}), ("gauss", None, None, 1.0, 0.1))
        self.assertEqual(find_centre.reference({"prior": {"min": 0, "max": 2}, "ref": {"dist": "norm", "loc": 0.7, "scale": 0.1}}), 0.7)
        self.assertEqual(find_centre.reference({"prior": {"min": 0, "max": 2}}), 1.0)
        self.assertEqual(find_centre.reference({"prior": {"dist": "norm", "loc": 3.0, "scale": 0.1}}), 3.0)

    def test_a_parameter_the_covariance_lacks_gets_its_own_axis(self):
        infos = {"a": {"prior": {"min": 0, "max": 2}}, "c": {"prior": {"min": 0, "max": 1}},
                 "new": {"prior": {"min": 0, "max": 10}, "proposal": 0.25}, "wide": {"prior": {"min": 0, "max": 40}}}
        matrix, widths, missing = find_centre.scaling(["a", "new", "c", "wide"], infos, (NAMES, COV))
        self.assertEqual(missing, ["new", "wide"])
        np.testing.assert_allclose(widths, [0.1, 0.25, 0.02, 2.0])
        self.assertEqual(matrix[0, 2], COV[0, 2])
        self.assertEqual((matrix[1, 0], matrix[1, 2], matrix[3, 1]), (0.0, 0.0, 0.0))


class Fit(unittest.TestCase):
    def problem(self, start, lower, upper):
        return find_centre.Problem(NAMES, np.array(start, float), np.array(lower, float), np.array(upper, float),
                                   np.linalg.cholesky(COV), SD)

    def test_finds_a_correlated_optimum_far_from_the_start(self):
        problem = self.problem([1.8, 4.0, 0.9, -9.0], [-np.inf] * 4, [np.inf] * 4)     # 8 to 20 widths off
        calls = []
        def evaluate(points):
            calls.append(len(points))
            return logpost(points)
        z, value, state, values, result = find_centre.fit(problem, evaluate, 0.1, 100, log=lambda *_: None)
        theta = problem.clipped(z)[0][0]
        np.testing.assert_allclose(theta, TRUTH, atol=2e-3 * SD.max())
        np.testing.assert_allclose((theta - TRUTH) / SD, 0.0, atol=0.02)
        self.assertLess(value, 1e-3)
        self.assertTrue(all(n == 9 for n in calls))                      # every batch is 2 n + 1 points
        self.assertLess(len(calls), 40)                                  # unit curvature: few iterations
        curvature = (values[1:5] + values[5:] - 2 * values[0]) / 0.1 ** 2
        np.testing.assert_allclose(curvature, 1.0, atol=1e-6)            # the covariance is the posterior's

    def edges(self, start, lower, upper, centre=TRUTH):
        asked = []
        def evaluate(points):
            asked.append(np.array(points))
            return logpost(points, centre)
        done = find_centre.fit_with_edges(NAMES, np.array(start, float), np.array(lower, float), np.array(upper, float),
                                          COV, SD, evaluate, 0.1, 100, log=lambda *_: None)
        return done, np.concatenate(asked)

    def test_an_optimum_beyond_a_prior_edge_ends_on_the_edge(self):
        # d would like to be 3 but its prior ends at 2: it is held at 2, the others settle given that,
        # and the likelihood is never asked beyond the edge
        lower, upper = [-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, 1.0, 2.0]
        done, asked = self.edges([1.0, -2.0, 0.5, 1.0], lower, upper)
        theta = done["theta"]
        self.assertEqual(theta[3], 2.0)
        self.assertEqual({NAMES[k]: side for k, side in done["held"].items()}, {"d": "upper"})
        self.assertLessEqual(asked[:, 3].max(), 2.0)
        expected = TRUTH[:3] + COV[:3, 3] / COV[3, 3] * (2.0 - TRUTH[3])
        np.testing.assert_allclose((theta[:3] - expected) / SD[:3], 0.0, atol=0.02)
        self.assertTrue(done["converged"])
        self.assertAlmostEqual(done["value"], float(-logpost(theta[None])[0]), places=9)

    def test_a_start_on_an_edge_is_released_when_the_optimum_is_inside(self):
        # as a reference value that sits on its prior's upper limit: c starts at 1, the optimum is 0.5
        done, _ = self.edges([1.0, -2.0, 1.0, 3.0], [-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, 1.0, 9.0])
        self.assertEqual(done["held"], {})
        np.testing.assert_allclose((done["theta"] - TRUTH) / SD, 0.0, atol=0.02)

    def test_two_edges_at_once(self):
        # the optimum lies outside in c (below 0) and in d (above 2)
        centre = np.array([1.0, -2.0, -0.03, 3.0])
        done, asked = self.edges([1.0, -2.0, 0.5, 1.0], [-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, 1.0, 2.0], centre)
        self.assertEqual({NAMES[k]: side for k, side in done["held"].items()}, {"c": "lower", "d": "upper"})
        np.testing.assert_array_equal(done["theta"][2:], [0.0, 2.0])
        self.assertGreaterEqual(asked[:, 2].min(), 0.0)

    def test_a_parameter_at_an_edge_gets_the_distance_of_a_drop_of_one(self):
        # held at d = 2 with the optimum at 3: along d alone -log-posterior is (d - 3)^2 / (2 s^2) with the
        # conditional width s, so it is 1 above its value at the edge where (3 - d)^2 = 1 + 2 s^2
        lower, upper = np.array([-np.inf, -np.inf, 0.0, 0.0]), np.array([np.inf, np.inf, 1.0, 2.0])
        theta = np.array([1.0, -2.0, 0.5, 2.0])
        value = float(-logpost(theta[None])[0])
        got = find_centre.one_sided_widths(theta, value, {3: "upper"}, lower, upper, SD, lambda points: logpost(points))
        s2 = 1.0 / PRECISION[3, 3]
        gradient = PRECISION[3] @ (theta - TRUTH)              # d(-log-posterior)/dd at the edge, others fixed
        exact = s2 * (gradient + np.sqrt(gradient ** 2 + 2 / s2))     # root of g x + x^2 / (2 s2) = 1 going inside
        self.assertAlmostEqual(got[3], exact, delta=0.15 * exact)     # read off a ladder of distances
        self.assertEqual(find_centre.one_sided_widths(theta, value, {}, lower, upper, SD, None), {})

    def test_a_start_the_likelihood_cannot_evaluate_is_an_error(self):
        problem = self.problem(TRUTH, [-np.inf] * 4, [np.inf] * 4)
        with self.assertRaises(RuntimeError):
            find_centre.fit(problem, lambda points: np.full(len(points), -np.inf), 0.1, 5, log=lambda *_: None)


class Box(unittest.TestCase):
    def centre(self):
        flat = lambda lo, hi: dict(kind="flat", lower=lo, upper=hi, loc=None, scale=None)
        return dict(parameters={
            "h": dict(value=67.4, fixed=True, prior=flat(60, 80)),
            "a": dict(value=1.0, fixed=False, at_prior_edge=None, conditional_width=0.02, prior=flat(0.0, 5.0)),
            "edge": dict(value=0.0, fixed=False, at_prior_edge="lower", conditional_width=None, prior=flat(0.0, 50.0)),
            "g": dict(value=1.0, fixed=False, at_prior_edge=None, conditional_width=0.004,
                      prior=dict(kind="gauss", lower=None, upper=None, loc=1.0, scale=0.01)),
            "lacking": dict(value=2.0, fixed=False, at_prior_edge=None, conditional_width=0.05, prior=flat(0.0, 10.0)),
            "narrow": dict(value=3.0, fixed=False, at_prior_edge=None, conditional_width=0.9, prior=flat(0.0, 10.0))})

    def test_centre_plus_and_minus_the_widths_cut_at_the_priors(self):
        widths = {"h": 0.4, "a": 0.1, "edge": 0.5, "g": 0.007, "narrow": 0.3}
        rows = {r["name"]: r for r in make_box.box(self.centre(), widths, half=6.0)}
        np.testing.assert_allclose((rows["h"]["lower"], rows["h"]["upper"]), (65.0, 69.8))
        np.testing.assert_allclose((rows["a"]["lower"], rows["a"]["upper"]), (0.4, 1.6))
        self.assertEqual((rows["edge"]["lower"], rows["edge"]["upper"], rows["edge"]["cut_at_prior"]), (0.0, 3.0, ["lower"]))
        np.testing.assert_allclose((rows["g"]["lower"], rows["g"]["upper"]), (0.958, 1.042))          # no prior range to cut at
        np.testing.assert_allclose((rows["lacking"]["lower"], rows["lacking"]["upper"]), (1.4, 2.6))  # 2 x conditional
        self.assertIn("conditional", rows["lacking"]["width_from"])
        # the log-posterior is wider along `narrow` than the covariance says: the measured width is used
        self.assertEqual(rows["narrow"]["width"], 0.9)
        np.testing.assert_allclose((rows["narrow"]["lower"], rows["narrow"]["upper"]), (0.0, 8.4))

    def test_a_parameter_at_an_edge_that_the_covariance_lacks(self):
        centre = self.centre()
        centre["parameters"]["edge"]["one_sided_width"] = 0.04
        rows = {r["name"]: r for r in make_box.box(centre, {"h": 0.4, "a": 0.1, "g": 0.007, "narrow": 0.3}, half=6.0)}
        np.testing.assert_allclose((rows["edge"]["lower"], rows["edge"]["upper"]), (0.0, 0.48))     # 6 x 2 x 0.04 inside the edge
        self.assertIn("drop of 1", rows["edge"]["width_from"])

    def test_a_parameter_without_any_width_stops_it(self):
        centre = self.centre()
        centre["parameters"]["edge"]["conditional_width"] = None
        with self.assertRaises(SystemExit):
            make_box.box(centre, {"h": 0.4, "a": 0.1, "g": 0.007, "narrow": 0.3}, half=6.0)


class Guard(unittest.TestCase):
    """A rank that raises stops the others (stop_peers_on_error), on a communicator that only
    records. That both ranks of a real job then end was tried with mpirun, not here."""

    def guarded(self, error, size=2):
        aborted = []

        @find_centre.stop_peers_on_error
        def work():
            if error is not None:
                raise error
            return "done"

        fake = SimpleNamespace(COMM_WORLD=SimpleNamespace(size=size, Abort=aborted.append))
        printed = io.StringIO()
        with mock.patch.object(find_centre, "MPI", fake), contextlib.redirect_stderr(printed):
            try:
                return work(), aborted, printed.getvalue()
            except BaseException as raised:
                self.assertIs(raised, error)
                return None, aborted, printed.getvalue()

    def test_an_error_in_a_worker_s_likelihood_stops_the_job(self):
        for error in (KeyError("a parameter the likelihood does not know"), SystemExit("--fix x: not a sampled parameter")):
            result, aborted, printed = self.guarded(error)
            self.assertEqual((result, aborted), (None, [1]))
            self.assertIn(type(error).__name__, printed)
        self.assertEqual(self.guarded(KeyError("alone"), size=1), (None, [], ""))      # without peers: the error alone

    def test_an_exit_that_is_not_a_failure_aborts_nothing(self):
        for error in (None, SystemExit(0), SystemExit()):
            self.assertEqual(self.guarded(error)[1:], ([], ""))
        self.assertIsNot(getattr(find_centre.main, "__wrapped__", None), None)          # main is guarded


if __name__ == "__main__":
    unittest.main()
