"""Regression cases from the release review; no external likelihood required."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


class ReleaseSafety(unittest.TestCase):
    def test_boost_residence_and_duplicate_alignment(self):
        from config.config import BoostConfig
        from dataset.boost import draw_boost, finish_boost
        x = np.r_[np.zeros((90, 1)), np.ones((10, 1))].astype('float32')
        mode = dict(x=x, f=np.zeros(100), weight=1., log_evidence=0.,
                    log_evidence_spread=0., efficiency=1., walkers=10, peak=np.zeros(1),
                    peak_value=0., acceptance=.5)
        # Sampling all row indices must preserve the 90:10 residence ratio, not 1:1.
        with patch('dataset.boost.posterior_modes', return_value=([mode], {})):
            points, draw = draw_boost(lambda y: np.zeros(len(y)), np.zeros(1), np.ones(1),
                                     x, 7., BoostConfig(n=100, start_iteration=0), seed=3, log=lambda _: None)
        self.assertEqual(int((points[:, 0] == 0).sum()), 90)
        with tempfile.TemporaryDirectory() as td:
            report = finish_boost(Path(td), 0, ['x'], draw, points, np.arange(100.), log=lambda _: None)
            self.assertEqual(report['failed'], 0)
            import pandas as pd
            np.testing.assert_array_equal(pd.read_csv(Path(td)/'boost_it_0.csv').loglkl, np.arange(100))

    def test_boost_zero_density_and_unresolved_returns(self):
        from dataset.boost import boost_report
        draw = dict(predicted=np.zeros(4), label=np.array([0,0,1,1]), modes=[dict(weight=.5)]*2)
        # A point without any value: what the mode's mean would have been is not known.
        for truth in ([0,np.nan,0,0], [0,np.inf,0,0], [0,-np.inf,0,0]):
            report = boost_report(draw, truth)
            self.assertFalse(report['evidence_correction_valid'])
            self.assertIn('without a value', report['correction_withheld_reason'])
            self.assertTrue(all('corrected_weight' not in m for m in report['modes']))
            self.assertEqual((report['failed'], report['refused']), (1, 0))
        # A point the likelihood refused, by a marked -inf or by the launcher's refused value:
        # zero weight, in the denominator, as hpc/analysis/true_check.py counts it.
        for report in (boost_report(draw, [0,-np.inf,0,0], zero_density=[False,True,False,False]),
                       boost_report(draw, [0,-999,0,0], ignore=-999)):
            self.assertTrue(report['evidence_correction_valid'])
            self.assertIsNone(report['correction_withheld_reason'])
            self.assertAlmostEqual(report['modes'][0]['corrected_weight'], 1/3)
            self.assertAlmostEqual(report['modes'][1]['corrected_weight'], 2/3)
            self.assertEqual(report['failed'], 0)
            self.assertEqual(report['zero_density'] + report['refused'], 1)
            self.assertEqual(report['modes'][0]['effective_points'], 1.0)
        self.assertEqual(boost_report(draw, [0,-999,0,0], ignore=-999)['modes'][0]['refused'], 1)
        # refused and without a value together: withheld
        self.assertFalse(boost_report(draw, [0,-999,np.nan,0], ignore=-999)['evidence_correction_valid'])

    def test_a_boost_with_refused_points_keeps_its_corrected_weights(self):
        from dataset.boost import finish_boost
        import pandas as pd

        points = np.arange(8, dtype=np.float32)[:, None]
        draw = dict(points=points, predicted=np.zeros(8), label=np.repeat([0, 1], 4), modes=[dict(weight=.5)] * 2, info={})
        for bad, ignore, valid in ((-np.inf, None, True), (-30700., -30700., True), (np.nan, None, False)):
            truth = np.zeros(8); truth[1] = bad
            said = []
            with tempfile.TemporaryDirectory() as td:
                order = np.random.default_rng(1).permutation(8)             # evaluations come back in any order
                report = finish_boost(Path(td), 3, ['x'], draw, points[order], truth[order], log=said.append, ignore=ignore)
                saved = json.loads((Path(td) / 'boost_it_3.json').read_text())
                np.testing.assert_array_equal(pd.read_csv(Path(td) / 'boost_it_3.csv').loglkl.fillna(-1).to_numpy()[[0, 2]], [0, 0])
            self.assertEqual(report['evidence_correction_valid'], valid)
            self.assertEqual(saved['evidence_correction_valid'], valid)
            if valid:
                self.assertAlmostEqual(saved['modes'][0]['corrected_weight'], 3 / 7)    # 3 of its 4 points carry weight
                self.assertAlmostEqual(saved['modes'][1]['corrected_weight'], 4 / 7)
                self.assertEqual(saved['refused'] + saved['zero_density'], 1)
                self.assertIn('3 true values (1 refused: zero weight, counted in the mean)', said[0])
                self.assertIn('-> 42.9% with the true values', said[0])
            else:
                self.assertEqual(saved['failed'], 1)
                self.assertIn('withheld (objective returns without a value', said[0])
                self.assertIn('3 usable points', said[0])

    def test_an_unknown_mpi_schedule_is_named_at_the_start(self):
        import client

        for fine in ({}, {"CLIENT_MPI_SCHEDULE": "chunked"}, {"CLIENT_MPI_SCHEDULE": " Static "}):
            self.assertIsNone(client.schedule_note(fine))
        for unknown in ("dynamc", "dynamic"):
            self.assertIn(f"'{unknown}' is not a schedule", client.schedule_note({"CLIENT_MPI_SCHEDULE": unknown}))
        from utils.mpi_utils import _requested_schedule
        with patch.dict('os.environ', {"CLIENT_MPI_SCHEDULE": "dynamic"}):
            self.assertIsNone(_requested_schedule())             # which is the default: the note says what happens
        from config.config import SamplingConfig
        with self.assertRaises(ValueError):
            SamplingConfig.from_dict(dict(sampler='aies', temperature=1, n_walkers=8, burn_in=0, n_steps=10,
                                          adaptive=True))

    def test_configuration_rejects_typos_coercions_and_ranges(self):
        from config.config import SamplingConfig, TrainingConfig, PriorConfig, AuditConfig
        for cls, data in ((SamplingConfig,dict(temprature=3)),
                          (AuditConfig,dict(failure_policy='invalid')),
                          (PriorConfig,dict(n_samples=-1,sampling_strategy='lhs')),
                          (TrainingConfig,dict(learning_rate=float('nan'))),
                          (SamplingConfig,dict(n_walkers=3)),
                          (SamplingConfig,dict(temperature=0)),
                          (AuditConfig,dict(quick='false'))):
            with self.subTest(cls=cls, data=data), self.assertRaises((ValueError, KeyError)):
                cls.from_dict(data)
        self.assertEqual(AuditConfig.from_dict(dict(learning_rate='1e-5')).learning_rate, 1e-5)

    def test_settings_that_do_not_fit_the_problem_stop_before_the_design(self):
        # A misspelt loss or activation and too few walkers for the dimension were first met
        # at the first training or the first chain, after the initial design had been paid for.
        from types import SimpleNamespace as S
        from client import check_settings_for
        from config.config import PriorConfig

        def settings(loss="msre", activation="alsing", walkers=16, boost=None):
            return S(training=S(loss=loss), model=S(activation=activation), sampling=S(n_walkers=walkers),
                     acquisition=S(boost=boost))
        check_settings_for(settings(), 8)
        check_settings_for(settings(loss="mse", activation="tanh", boost=S(walkers=64)), 8)
        for bad, said in ((dict(loss="msrre"), "training.loss"), (dict(activation="alsnig"), "model.activation"),
                          (dict(walkers=14), "sampling.n_walkers: 14 walkers for 8 parameters"),
                          (dict(boost=S(walkers=10)), "acquisition.boost.walkers")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, said):
                check_settings_for(settings(**bad), 8)
        with self.assertRaisesRegex(ValueError, "sampling_strategy"):
            PriorConfig.from_dict(dict(n_samples=10, sampling_strategy="lsh", n_sigma=None))

    def test_sampler_rejects_low_rank_ensemble(self):
        from sampling.aies import AIESampler
        import tensorflow as tf
        with self.assertRaises(ValueError):
            AIESampler(2,2,lambda x: tf.zeros(len(x)))
        sampler = AIESampler(4,2,lambda x: tf.zeros(len(x)))
        for x in (np.zeros((4,2)),np.array([[0,0],[1,1],[2,2],[3,3]]),np.zeros((2,4))):
            with self.assertRaises(ValueError):
                sampler._initialize(x)
        sampler._initialize([[0,0],[0,1],[1,0],[1,1]])

    def test_box_checks_shape_order_and_extrapolation(self):
        from evidence.__main__ import sampling_box
        md = SimpleNamespace(param_names=['a','b'],bounds={'a':(0,1),'b':(0,1)})
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'box.json'
            for names,lo,hi in ((['b','a'],[0,0],[1,1]),(['a','b'],[0],[1]),
                                (['a','b'],[-1,0],[1,1]),(['a','b'],[0,0],[0,1])):
                path.write_text(json.dumps(dict(param_names=names, lower=lo, upper=hi)))
                with self.assertRaises(ValueError): sampling_box(md, box_file=path)
            path.write_text(json.dumps(dict(param_names=['a','b'], lower=[.1,.2], upper=[.8,.9])))
            box,_=sampling_box(md, box_file=path)
            np.testing.assert_equal(box, [[.1,.2],[.8,.9]])

    def test_atomic_failure_retains_previous_artifact_and_lock(self):
        from run.storage import atomic_path, RunLock, PhaseStore
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'data.csv';p.write_text('old')
            with self.assertRaises(RuntimeError):
                with atomic_path(p) as q:
                    q.write_text('partial');raise RuntimeError()
            self.assertEqual(p.read_text(),'old')
            lock=RunLock(td)
            try:
                with self.assertRaises(RuntimeError): RunLock(td)
            finally: lock.close()
            RunLock(td).close()
            phases=PhaseStore(td);phases.begin('audit0')
            with self.assertRaisesRegex(RuntimeError,'Incomplete phase'): phases.check()
            phases.complete('audit0',[p]);phases.check()
            p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'artifact changed'): phases.check()

    def test_identity_rejects_domain_or_objective_changes(self):
        from run.identity import verify
        with tempfile.TemporaryDirectory() as td:
            current=dict(parameters=[dict(name='x',lower=0,upper=1,scale=1)],files={'input':'aaa'})
            verify(td,current,new=True);verify(td,current)
            for changed in (dict(current,files={'input':'bbb'}),dict(current,parameters=[])):
                with self.assertRaises(ValueError): verify(td,changed)
        with tempfile.TemporaryDirectory() as td:
            # A continued run without a record: its training values cannot be checked.
            with self.assertRaisesRegex(ValueError,'no objective identity'): verify(td,current)

    def test_torn_tail_preserved_and_intact_cache_reused(self):
        from profiles.evaluation import WorkflowLedger, aggregate_counts
        from profiles.journal import records
        with tempfile.TemporaryDirectory() as td:
            ledger=WorkflowLedger(lambda x: -float(x[0]**2),[[-1,1]],Path(td),10)
            ledger.stage='node';ledger([.1]);p=Path(ledger.journal) if hasattr(ledger,'journal') else next(Path(td).glob('*.jsonl'))
            with p.open('ab') as f:f.write(b'{"event":')
            raw=p.read_bytes()
            with self.assertWarns(RuntimeWarning):
                resumed=WorkflowLedger(lambda x: self.fail('cached point recomputed'),[[-1,1]],Path(td),10)
            self.assertAlmostEqual(resumed([.1]),-.01)
            self.assertEqual(p.read_bytes(),raw)
            self.assertTrue(p.with_suffix('.truncated.json').exists())
            with self.assertWarns(RuntimeWarning):counts=aggregate_counts(td)
            self.assertEqual(counts['backend_attempts'],1)
            self.assertEqual(counts['uncertain_tail_records'],1)
            bad=Path(td)/'middle.jsonl';bad.write_bytes(b'{broken}\n{"event":"start"}\n')
            with self.assertRaises(ValueError):list(records(bad))

    def test_node_budget_survives_interruption_and_keeps_incumbent(self):
        from test_residual_newton import study
        from profiles.residual_newton import NewtonRefiner,NewtonConfig
        from profiles.evaluation import WorkflowLedger,aggregate_counts
        sb,truth,source,_,bounds=study(curvature_error=-.8)
        x=source['profiles']['p0']['xopt'][2];fixed=source['profiles']['p0']['grid'][2]
        with tempfile.TemporaryDirectory() as td:
            calls=0
            def interrupted(x):
                nonlocal calls
                calls+=1
                if calls==9:raise RuntimeError('interrupted')
                return truth(x)
            cfg=NewtonConfig(maxcalls=12,tol=1e-9)
            j=WorkflowLedger(interrupted,bounds,Path(td),100)
            with self.assertRaises(RuntimeError):NewtonRefiner(sb,j,cfg).solve(sb.box(x),0,fixed)
            k=WorkflowLedger(truth,bounds,Path(td),100)
            value,point,diag,_=NewtonRefiner(sb,k,cfg).solve(sb.box(x),0,fixed)
            self.assertLessEqual(aggregate_counts(td)['backend_attempts'],12)
            self.assertEqual(diag['previous_node_attempts'],9)
            self.assertGreaterEqual(value,max(v for _,v in k.prior_values_by_stage['node']))
            self.assertAlmostEqual(value,truth(point))

    def test_nonfinite_interior_probes_return_unverified_incumbent(self):
        from test_residual_newton import study
        from profiles.residual_newton import NewtonRefiner,NewtonConfig
        from profiles.evaluation import WorkflowLedger
        sb,truth,source,_,bounds=study()
        x=np.asarray(source['best_x']);calls=0
        def isolated(point):
            nonlocal calls
            calls+=1
            return truth(point) if calls==1 else -np.inf
        with tempfile.TemporaryDirectory() as td:
            j=WorkflowLedger(isolated,bounds,Path(td),100)
            value,_,diag,_=NewtonRefiner(sb,j,NewtonConfig(maxcalls=30)).solve(sb.box(x))
            self.assertTrue(np.isfinite(value));self.assertNotEqual(diag['flag'],0)
            self.assertEqual(diag['reason'],'unverified_interior')

    def test_stage_accounting_includes_failed_and_injected(self):
        from run.objective_calls import ObjectiveCalls,summarize
        class Backend:
            refused_value=-999
            def evaluate_with_derived(self,x):
                if x[0]==3:raise RuntimeError('backend')
                return [0,-np.inf,-999][int(x[0])]
        with tempfile.TemporaryDirectory() as td:
            journal=ObjectiveCalls(td,0);b=Backend()
            for v in range(3):journal.evaluate(b,[v])
            journal.stage='initial/injection'
            with self.assertRaises(RuntimeError):journal.evaluate(b,[3])
            report=summarize(td)['stages']
            self.assertEqual(report['initial/design'],dict(attempts=3,finite=1,nonfinite=1,refused=1,unfinished=0))
            self.assertEqual(report['initial/injection']['exceptions'],1)

    def test_settings_file_is_the_one_yaml_or_yml_of_the_run(self):
        from run.run import settings_file
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            with self.assertRaisesRegex(FileNotFoundError,'No YAML configuration'):settings_file(run)
            (run/'client.yml').write_text('a: 1')
            self.assertEqual(settings_file(run).name,'client.yml')
            (run/'other.yaml').write_text('a: 1')
            with self.assertRaisesRegex(ValueError,'Multiple run configurations.*client.yml'):settings_file(run)

    def test_call_journal_flushes_every_record_and_limits_its_syncs(self):
        from run.objective_calls import ObjectiveCalls,SYNC_SECONDS
        class Backend:
            def evaluate_with_derived(self,x):return [0.]
        clock=[100.]
        with tempfile.TemporaryDirectory() as td,\
             patch('run.objective_calls.time.monotonic',side_effect=lambda:clock[0]),\
             patch('run.objective_calls.os.fsync') as sync:
            journal=ObjectiveCalls(td,0)
            for v in range(50):journal.evaluate(Backend(),[v])
            self.assertEqual(sync.call_count,1)                       # an instant likelihood
            self.assertEqual(len(journal.path.read_text().splitlines()),100)     # on disk, nothing buffered
            for v in range(3):
                clock[0]+=2*SYNC_SECONDS                              # a slow one: before every call
                journal.evaluate(Backend(),[v])
            self.assertEqual(sync.call_count,4)
            journal._stream.close()

    def test_knn_index_is_built_when_read_and_holds_every_point(self):
        import dataset.dataset as module
        likelihood=SimpleNamespace(param_names=['a','b'],derived_names=[])
        rng=np.random.default_rng(0)
        points=rng.normal(size=(40,2))
        with patch.object(module,'_WhitenedKNN',wraps=module._WhitenedKNN) as built:
            data=module.TrainingDataset(points[:20],-np.sum(points[:20]**2,axis=1),likelihood,3,1.)
            for start in (20,30):
                data.add_data(points[start:start+10],-np.sum(points[start:start+10]**2,axis=1))
            self.assertEqual(built.call_count,0)
            index=data.knn_index
            self.assertIs(data.knn_index,index)
            self.assertEqual(built.call_count,1)
        fresh=module._WhitenedKNN(np.asarray(points,np.float32),3)
        np.testing.assert_array_equal(index.query(points[:5])[0],fresh.query(points[:5])[0])
        data.add_data(points[:1],np.zeros(1))
        self.assertIsNot(data.knn_index,index)

    def test_shared_consumer_region_and_hash_guard(self):
        from test_profiles_workflow import make_toy_run
        from test_profile_region import add_region
        from sampling.targets import load_surrogate, load, resolve
        from profiles.region import resolve as profile_resolve
        with tempfile.TemporaryDirectory() as td:
            run=make_toy_run(Path(td));add_region(run)
            target,entry=load_surrogate(run,2)
            self.assertEqual(entry,profile_resolve(run,2,'run'))
            bare,_=load_surrogate(run,2,None)
            point=np.array([[3.9,3.9]],np.float32)
            self.assertLess(float(target.logpost(point)[0]),float(bare.logpost(point)[0]))
            file=run/entry['file'];file.write_bytes(file.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'region changed'):load(run,entry)

    def test_unknown_tail_reserves_global_budget_only_once(self):
        from test_residual_newton import Quadratic
        from profiles.residual_newton import NewtonRefiner, NewtonConfig
        from profiles.evaluation import WorkflowLedger, aggregate_counts
        bounds=np.array([[-1.,1.]])
        sb=Quadratic(bounds,np.eye(1),np.array([0.]))
        truth=lambda x:float(-.5*(x[0]-.1)**2)
        x=np.array([0.])
        config=NewtonConfig(maxcalls=2,flat_central=0)
        with tempfile.TemporaryDirectory() as td:
            old=WorkflowLedger(truth,bounds,Path(td),100);old.stage='node';old(x)
            with old.journal.open('ab') as stream:stream.write(b'{"event":')
            with self.assertWarns(RuntimeWarning):ledger=WorkflowLedger(truth,bounds,Path(td),100)
            _,_,diag,_=NewtonRefiner(sb,ledger,config).solve(sb.box(x))
            self.assertEqual(diag['uncertain_budget_reservations'],0)
            self.assertEqual(diag['backend_calls'],1)
            self.assertEqual(diag['lifetime_node_attempts'],2)
            self.assertEqual(ledger.prior_calls,2)
            self.assertEqual(ledger.max_calls,98)
            _,_,later,_=NewtonRefiner(sb,ledger,config).solve(sb.box(x),stage='later_node')
            self.assertEqual(later['uncertain_budget_reservations'],0)
            self.assertEqual(later['previous_node_attempts'],0)
            self.assertEqual(ledger.prior_calls,2)
            counts=aggregate_counts(td)
            self.assertEqual(counts['budget_reserved_attempts']-counts['backend_attempts'],1)

    def test_node_miss_estimate_is_an_upper_bound_and_knows_refused_points(self):
        import threading
        from profiles.hybrid import NodeLedger
        ledger=SimpleNamespace(cache={},_key=lambda x:tuple(x))
        shared=SimpleNamespace(ledger=ledger,bounds=[[-1,1]],lock=threading.Lock(),refused=set())
        node=NodeLedger(shared,'node')
        self.assertEqual(node.missing_calls([[0],[0]]),2)
        ledger.cache[(0,)]=1.
        self.assertEqual(node.missing_calls([[0],[0]]),0)
        shared.refused.add((1,))
        self.assertEqual(node.missing_calls([[1],[1]]),0)


if __name__=='__main__':unittest.main()


