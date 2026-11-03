"""Numerical regression checks; run with python -m unittest discover -s tests."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiles.metric import confidence_set, crossings, interval, dpm, resolution_floor
from profiles.surrogate import ProfileOptimizer, scan_profiles
from run.run import _start_iteration


def profile(g, d):
    return dict(names=['x'], profiles={'x': dict(grid=list(g), dchi2=list(d))})


class MetricTests(unittest.TestCase):
    def test_endpoint_roots_and_plateau_no_duplicates(self):
        self.assertEqual(crossings([0, 1, 2], [1, 0, 1], 1), [0, 2])
        self.assertEqual(crossings([0, 1, 2, 3], [0, 1, 1, 2], 1), [1, 2])
        self.assertEqual(confidence_set([0, 1, 2, 3], [0, 1, 1, 2], 1), [(0, 2)])

    def test_empty_full_and_tangent(self):
        self.assertEqual(interval([-1, 1], [2, 2], 1, -1, 1)[2], 'empty')
        self.assertEqual(interval([-1, 1], [0, 0], 1, -1, 1), (-1, 1, 'unconstrained'))
        self.assertEqual(interval([-1, 0, 1], [2, 1, 2], 1, -1, 1), (0, 0, 'point'))
        self.assertEqual(interval([-1, 1], [2, 1], 1, -1, 1), (1, 1, 'point'))

    def test_disconnected_is_not_an_envelope(self):
        self.assertEqual(confidence_set([0, 1, 2, 3, 4], [2, 0, 2, 0, 2], 1),
                         [(0.5, 1.5), (2.5, 3.5)])
        self.assertEqual(interval([0, 1, 2, 3, 4], [2, 0, 2, 0, 2], 1, 0, 4)[2], 'disconnected')

    def test_one_sided_and_known_endpoint_score(self):
        self.assertEqual(interval([0, 1, 2], [0, 1, 2], 1, 0, 2), (0, 1, 'upper limit'))
        self.assertEqual(interval([0, 1, 2], [2, 1, 0], 1, 0, 2), (1, 2, 'lower limit'))
        ref, em = profile([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4]), profile([-2, -1, 0, 1, 2], [8, 2, 0, 2, 8])
        self.assertAlmostEqual(dpm(em, ref)['1 sigma']['x'][0], .5)
        self.assertEqual(dpm(ref, ref)['1 sigma']['x'], (0., 'two-sided'))

    def test_spurious_constraints_are_visible(self):
        with self.assertWarns(RuntimeWarning):
            scores = dpm(profile([-1, 0, 1], [2, 0, 2]), profile([-1, 0, 1], [0, 0, 0]))
        self.assertEqual(scores['1 sigma']['x'], (.5, 'spurious constraint'))

    def test_undefined_interval_is_flagged(self):
        with self.assertWarns(RuntimeWarning):
            scores = dpm(profile([-1, 0, 1], [5, 5, 5]), profile([-1, 0, 1], [2, 0, 2]))
        self.assertTrue(np.isnan(scores['1 sigma']['x'][0]))
        self.assertEqual(scores['1 sigma']['x'][1], 'emulator empty')

    def test_invalid_grids_and_values(self):
        for g, d in [([0, 0], [1, 2]), ([1, 0], [1, 2]), ([0, 1], [0, np.nan]), ([0], [0])]:
            with self.assertRaises(ValueError):
                confidence_set(g, d, 1)
        with self.assertRaises(ValueError):
            dpm(profile([1, 1.0001, 1.0002], [2, 0, 2]),
                profile([1.000001, 1.000101, 1.000201], [2, 0, 2]))

    def test_off_grid_gaussian_floor(self):
        g = np.linspace(-5, 5, 61)
        d = (g - .08)**2
        roots = crossings(g, d, 1)
        actual = abs(roots[-1] - roots[0] - 2)/2
        self.assertAlmostEqual(resolution_floor(profile(g, d), 'x', 1), actual, places=12)
        self.assertLess(actual, .004)

    def test_floor_rejects_nonquadratic_or_unconstrained(self):
        g = np.linspace(-3, 3, 31)
        self.assertIsNone(resolution_floor(profile(g, g**4), 'x', 1))
        self.assertIsNone(resolution_floor(profile(g, g*0), 'x', 1))

    def test_unreliable_optimizer_metadata_warns(self):
        p = profile([-2, 0, 2], [4, 0, 4])
        p['optimization_quality'] = dict(converged=False)
        with self.assertWarns(RuntimeWarning):
            dpm(p, p)


class OptimizerTests(unittest.TestCase):
    @staticmethod
    def quadratic(u, i, value):
        x = np.empty(2)
        mask = np.ones(2, bool)
        if i is not None:
            x[i], mask[i] = value, False
        x[mask] = u
        # nuisance optimum y=0.5x; exact profile negative logL=0.5x^2.
        grad = np.array([1.25*x[0] - .5*x[1], x[1] - .5*x[0]])
        return .5*(x[0]**2 + (x[1]-.5*x[0])**2), grad[mask]

    def test_active_bound_projected_gradient(self):
        def objective(u, i, v):
            return float((u[0]-2)**2), np.array([2*(u[0]-2)])
        v, x, d = ProfileOptimizer(objective, [-1], [1])(None, None, [[0]])
        self.assertAlmostEqual(x[0], 1)
        self.assertTrue(d['stationary'])
        self.assertEqual(d['projected_gradient_inf'], 0)

    def test_failure_keeps_better_seed_and_reports_status(self):
        def objective(u, i, v):
            return float(u@u), 2*u
        fake = OptimizeResult(x=np.array([.9]), success=False, status=1,
                              message='budget exhausted', nit=1, nfev=2)
        with patch('profiles.surrogate.minimize', return_value=fake):
            v, x, d = ProfileOptimizer(objective, [-1], [1], retry_starts=0)(None, None, [[.1]])
        self.assertAlmostEqual(x[0], .1)
        self.assertFalse(d['stationary'])
        self.assertFalse(d['attempts'][-1]['optimizer_success'])
        self.assertEqual(d['attempts'][-1]['status'], 1)

    def test_nonfinite_result_does_not_replace_seed(self):
        fake = OptimizeResult(x=np.array([np.nan]), success=False, status=2,
                              message='failure', nit=1, nfev=1)
        with patch('profiles.surrogate.minimize', return_value=fake):
            v, x, d = ProfileOptimizer(lambda u,i,v:(float(u@u),2*u), [-1], [1], retry_starts=0)(None,None,[[.2]])
        self.assertAlmostEqual(x[0], .2)
        self.assertFalse(d['attempts'][-1]['finite'])

    def test_retry_budget_is_bounded(self):
        fake = OptimizeResult(x=np.array([.9]), success=False, status=1,
                              message='failure', nit=1, nfev=1)
        with patch('profiles.surrogate.minimize', return_value=fake) as solve:
            _, _, d = ProfileOptimizer(lambda u,i,v:(float(u@u),2*u), [-1], [1], retry_starts=2)(None,None,[[.2]])
        self.assertEqual(solve.call_count, 3)
        self.assertEqual(d['retries'], 2)

    def test_one_dimensional_fixed_profile(self):
        opt = ProfileOptimizer(lambda u,i,v:(v*v, np.empty(0)), [-1], [1])
        value, x, diag = opt(0, .5, [[0]])
        self.assertEqual(value, -.25)
        self.assertTrue(diag['stationary'])

    def test_invalid_bounds_rejected(self):
        for lo, hi in [([0], [0]), ([0], [np.inf]), ([1], [0])]:
            with self.assertRaises(ValueError):
                ProfileOptimizer(self.quadratic, lo, hi)

class ResumeTests(unittest.TestCase):
    def test_augmented_arms_do_not_break_iteration_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'trained_models').mkdir(); (p/'training_data').mkdir()
            for n in ['model_it_2.keras', 'model_it_2_pa2000.keras']:
                (p/'trained_models'/n).touch()
            for n in ['data_it_3.csv', 'data_it_5_pa1000.csv']:
                (p/'training_data'/n).touch()
            self.assertEqual(_start_iteration(p), 3)


if __name__ == '__main__':
    unittest.main()
