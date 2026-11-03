"""Bounded numerical-failure recovery; synthetic labels never become true data."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from config.config import AuditConfig
from training.audit_failures import evaluate_with_recovery, active_penalties, load_penalties


class FailureRecovery(unittest.TestCase):
    def recover(self, values, budget=4, **options):
        batches = []
        def evaluate(points):
            batches.append(points.copy())
            y = np.asarray(values[len(batches)-1], float)
            return points, y, np.full((len(points), 2), np.nan)
        result = evaluate_with_recovery(
            np.ones((1, 2)), evaluate, AuditConfig(failure_policy='penalize', **options),
            -10., np.zeros(2), np.ones(2), np.random.default_rng(7), budget)
        return result, batches

    def test_three_failed_neighbours_give_one_separate_penalty(self):
        (x, y, d, penalties, failures, primary), batches = self.recover([[np.nan], [np.nan, np.inf, -np.inf]])
        self.assertEqual([len(b) for b in batches], [1, 3])
        self.assertEqual(penalties, [dict(x=[1., 1.], penalty_loglkl=-1010.)])
        self.assertEqual(failures[0]['outcome'], 'penalized')
        self.assertEqual(d.shape, (4, 2))
        self.assertFalse(np.isfinite(y).any())
        self.assertTrue(np.all((x >= .999) & (x <= 1)))
        self.assertEqual(len({row.tobytes() for row in x}), 4)

    def test_finite_neighbour_is_real_and_prevents_penalty(self):
        result, batches = self.recover([[np.nan], [-9., np.nan, -np.inf]])
        self.assertEqual(result[3], [])
        self.assertEqual(result[4][0]['outcome'], 'finite_neighbour')
        np.testing.assert_array_equal(result[1][np.isfinite(result[1])], [-9.])

    def test_insufficient_budget_does_not_label_or_probe(self):
        result, batches = self.recover([[np.nan]], budget=3)
        self.assertEqual(len(batches), 1)
        self.assertEqual(result[3], [])
        self.assertEqual(result[4][0]['outcome'], 'insufficient_budget')

    def test_invalid_policy_settings(self):
        for key, values in dict(failure_probes=[0,5,1.5,True], failure_radius=[0,.1,np.nan],
                                failure_penalty=[0,np.inf], failure_training_fraction=[-1,2,np.nan],
                                failure_policy=['ignore']).items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    AuditConfig.from_dict({key:value})
        self.assertEqual(AuditConfig.from_dict({}).failure_policy, 'skip')

    def test_persistence_is_separate_and_real_observation_supersedes(self):
        from training.audit import save_audit, load_audit
        p = dict(x=[1., 1.], penalty_loglkl=-1010.)
        with tempfile.TemporaryDirectory() as directory:
            save_audit(directory, 3, ['a','b'], [[0.,0.]], [-10.], dict(penalties_added=[p]))
            x,y,d = load_audit(directory, 3, ['a','b'])
            np.testing.assert_array_equal(y.ravel(), [-10.])
            self.assertEqual(load_penalties(directory, 3, ['a','b']), [])
            loaded = load_penalties(directory, 4, ['a','b'])
            self.assertEqual(loaded, [p])
            self.assertEqual(active_penalties(loaded, [[1.,1.]], np.zeros(2), np.ones(2)), [])

    def test_audit_trains_penalty_without_polluting_truth_or_validation(self):
        from training.audit import audit_optima
        model = mock.Mock()
        state = [50.]
        model.get_weights.side_effect = lambda: state.copy()
        model.set_weights.side_effect = lambda weights: state.__setitem__(0, weights[0])
        def search(*args, **kwargs):
            kwargs['counts'].update(excess=[state[0]], hits=[1], starts=1)
            return [dict(x=np.ones(2), excess=state[0], distance=1., at_bound=2, starts=1, hits=1)]
        def tune(model, x, y, loss, validation, *args):
            self.assertEqual(len(x), 10)  # nine real training rows, one constraint
            self.assertEqual(y[-1], -1000.)
            self.assertEqual(validation[1].shape, (1,1))
            np.testing.assert_array_equal(validation[1], [[0.]])
            state[0] = -1000.
        loss_values = []
        with mock.patch('training.audit.find_maxima', side_effect=search), \
             mock.patch('training.audit.fine_tune', side_effect=tune), \
             mock.patch('training.audit._predict', side_effect=lambda m,x: np.zeros(len(x))):
            x,y,d,report = audit_optima(
                model, np.zeros((10,2)), np.zeros(10), .1, np.zeros(2), np.ones(2),
                lambda x: (x, np.full(len(x), np.nan), None),
                lambda y: loss_values.append(y.copy()),
                AuditConfig(k=1, rounds=4, starts=1, failure_policy='penalize'),
                16, 100, seed=1, log=lambda _: None)
        self.assertEqual(len(y), 0)
        self.assertEqual(report['new_best'], 0)
        self.assertEqual(report['true_calls'], 4)
        self.assertEqual(report['penalized'], 1)
        self.assertTrue(report['clear'])
        np.testing.assert_array_equal(loss_values[0], np.zeros(10))

    def test_constraint_repetition_is_balanced_and_bounded(self):
        from training.audit_failures import penalty_training_rows
        constraints = [dict(x=[i], penalty_loglkl=-1000-i) for i in range(3)]
        x,y = penalty_training_rows(constraints, 100, 1, .1)
        self.assertEqual(len(x), 10)
        np.testing.assert_array_equal(np.bincount(x.astype(int).ravel()), [4,3,3])
        self.assertEqual(len(penalty_training_rows(constraints, 100, 1, 0)[0]), 3)
        self.assertEqual(penalty_training_rows([], 100, 1, .1)[0].shape, (0,1))

    def test_checked_transport_classifies_errors(self):
        from utils.mpi_utils import broadcast_and_evaluate_checked
        def evaluator(x):
            if x[0] == 0: raise FloatingPointError('numerical failure')
            if x[0] == 1: raise TypeError('programming error')
            return [2., 3.]
        x,y,errors = broadcast_and_evaluate_checked(np.arange(3).reshape(-1,1), evaluator, 2)
        self.assertEqual([e['recoverable'] for e in errors], [True,False])
        self.assertTrue(np.isnan(y[:2]).all())
        np.testing.assert_array_equal(y[2], [2.,3.])

    def test_fatal_error_and_penalty_load_error_release_workers(self):
        from client import audit_network
        cfg = SimpleNamespace(acquisition=SimpleNamespace(audit=AuditConfig(failure_policy='penalize')))
        likelihood = SimpleNamespace(n_derived=0, evaluate_with_derived=lambda x: [0.],
                                     split_evaluations=lambda values: (values[:,0], None))
        def audit(evaluate, **kwargs):
            return evaluate(np.zeros((1,2)))
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch('client.bcast_idle') as broadcast, \
             mock.patch('training.audit.audit_optima', side_effect=audit), \
             mock.patch('utils.mpi_utils.broadcast_and_evaluate_checked', return_value=(
                 np.zeros((1,2)), np.full((1,1), np.nan),
                 [dict(x=[0.,0.], recoverable=False, exception='TypeError')])):
            journal = Path(directory)/'calls.jsonl'
            with self.assertRaisesRegex(RuntimeError, 'Fatal audit'):
                audit_network(cfg, likelihood, True, True, evaluation_journal=journal)
            self.assertIn('audit_error', broadcast.call_args.args[0])
            self.assertIn('fatal_exception', journal.read_text())
        with mock.patch('client.bcast_idle') as broadcast, \
             mock.patch('training.audit_failures.load_penalties', side_effect=ValueError('bad penalty file')):
            with self.assertRaisesRegex(ValueError, 'bad penalty file'):
                audit_network(cfg, likelihood, True, True, penalty_source=('unused', 1, ['x']))
            self.assertIn('audit_error', broadcast.call_args.args[0])

    def test_a_long_error_reaches_the_workers_trimmed_and_the_master_whole(self):
        from client import audit_network
        cfg = SimpleNamespace(acquisition=SimpleNamespace(audit=AuditConfig()))
        likelihood = SimpleNamespace(n_derived=0)
        with mock.patch('client.bcast_idle') as broadcast, \
             mock.patch('training.audit.audit_optima', side_effect=RuntimeError('x' * 100_000)):
            with self.assertRaises(RuntimeError) as raised:
                audit_network(cfg, likelihood, True, True)
        self.assertEqual(len(str(raised.exception)), 100_000)
        self.assertLess(len(broadcast.call_args.args[0]['audit_error']), 32_768 // 4)


if __name__ == '__main__':
    unittest.main()
