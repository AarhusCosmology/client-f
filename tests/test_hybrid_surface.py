"""Joint conditional truth, physical bounds, deterministic scheduling and recovery."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

from profiles.config import ProfileConfig
from profiles.hybrid import reoptimise_anchor
from profiles.hybrid_surface import scan_surface_hybrid, refine_surface_newton, validate_grid
from profiles.evaluation import WorkflowLedger, aggregate_counts
from profiles.residual_newton import NewtonRefiner, NewtonConfig
from profiles.evaluation import EvaluationLedger, CallBudgetExceeded
from test_residual_newton import Quadratic, Pool

quiet = lambda *a, **k: None


def study():
    # f = -.5[(z-x-y)^2 + x^2 + (y-x)^2], genuinely correlated surface.
    A = np.array([[-1, -1, 1], [1, 0, 0], [-1, 1, 0]], float)
    P = A.T @ A
    bounds = np.array([[-4, 4], [-4, 4], [-1.2, 1.2]])
    sb = Quadratic(bounds, P, np.array([0., 0., .35]))
    truth = lambda x: float(-.5 * np.asarray(x) @ P @ np.asarray(x))
    axes = [np.array([-.8, .1, .9]), np.array([-.7, .2, .8])]
    cfg = ProfileConfig(dimension=2, parameters=['x','y'], corrected_tol=1e-7, workers=1)
    rng = np.random.default_rng(3)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], (16, 3))
    data = np.column_stack([X, sb.value(sb.box(X))])
    source = scan_surface_hybrid(None, bounds, data, [0,1], axes, cfg, surrogate=sb, log=quiet)
    x, y = np.meshgrid(*axes, indexing='ij')
    z = np.clip(x+y, *bounds[2])
    exact = -.5*((z-x-y)**2 + x*x + (y-x)**2)
    return sb, truth, bounds, cfg, source, exact


def run(d, sb, truth, bounds, cfg, source, budget=10000):
    ledger = WorkflowLedger(Pool(truth), bounds, Path(d)/'calls', budget)
    result = refine_surface_newton(None, source, bounds, ledger, d, cfg, surrogate=sb, log=quiet)
    return result, ledger


class ReoptimiseAnchor(unittest.TestCase):
    @staticmethod
    def state(f):
        n = len(f)
        res = SimpleNamespace(U=np.arange(n, dtype=float)[:, None], f=np.array(f, float), g=np.zeros((n, 1)), pg=np.zeros(n),
                              status=np.zeros(n, int), iterations=np.zeros(n, int))
        return res, dict(stationary=np.zeros(n, bool))

    def certify_gives(self, *values):
        answers = iter(values)
        def answer(sb, U, fixed, **kwargs):
            res, second = self.state([next(answers)])
            return res, second
        return mock.patch('profiles.hybrid.certify', side_effect=answer)

    def test_a_search_that_does_not_beat_the_anchor_is_not_repeated(self):
        res, second = self.state([1.0, 2.0])
        with self.certify_gives(0.5, 0.5, 0.5) as certify:
            self.assertEqual(reoptimise_anchor(None, res, second, 1, SimpleNamespace(value_tol=1e-6, gtol=1e-4, trust=0.5, gain_tol=1e-6)), 1)
        self.assertEqual(certify.call_count, 1)
        self.assertEqual(res.f[0], 1.0)

    def test_it_goes_on_while_it_improves_up_to_the_limit(self):
        cfg = SimpleNamespace(value_tol=1e-6, gtol=1e-4, trust=0.5, gain_tol=1e-6)
        res, second = self.state([1.0, 5.0])
        with self.certify_gives(2.0, 3.0, 4.0, 4.5) as certify:
            self.assertEqual(reoptimise_anchor(None, res, second, 1, cfg), 3)
        self.assertEqual((certify.call_count, res.f[0]), (3, 4.0))
        res, second = self.state([1.0, 5.0])
        with self.certify_gives(6.0) as certify:
            self.assertEqual(reoptimise_anchor(None, res, second, 1, cfg), 1)
        self.assertEqual(res.f[0], 6.0)


class HybridSurfaceTests(unittest.TestCase):
    def test_a_refused_start_is_recorded_and_the_report_is_still_written(self):
        from profiles.io import save
        from profiles.surface_reporting import write_surface_outputs

        sb, truth, bounds, cfg, source, exact = study()
        start = np.clip(np.array(source['xopt'][1][2], float), bounds[:, 0], bounds[:, 1])
        objective = lambda x: -np.inf if np.allclose(x, start, atol=1e-12, rtol=0) else truth(x)
        with tempfile.TemporaryDirectory() as d:
            result, _ = run(d, sb, objective, bounds, cfg, source)
            self.assertEqual((result['logl'][1][2], result['xopt'][1][2], result['direct_logl'][1][2]), (None, None, None))
            self.assertEqual(result['diagnostics'][1][2]['reason'], 'refused start')
            found = np.array(result['logl'], float)
            np.testing.assert_allclose(found[~np.isnan(found)], exact[~np.isnan(found)], atol=1e-6)
            self.assertEqual(int(np.isnan(found).sum()), 1)
            save(Path(d)/'surrogate_surface.json', source)
            save(Path(d)/'ranges.json', {n: dict(clipped=False) for n in ('x', 'y')})
            save(Path(d)/'counts.json', aggregate_counts(Path(d)/'calls'))
            from scipy.interpolate import RectBivariateSpline
            with mock.patch('scipy.interpolate.RectBivariateSpline', wraps=RectBivariateSpline) as spline:
                write_surface_outputs(d)
            self.assertTrue((Path(d)/'surface.png').exists() and (Path(d)/'isocontours.png').exists())
            # The node's true value here is well inside the contours: nothing may stand in for it.
            self.assertLess(2 * (result['best_logl'] - exact[1, 2]), 25.0)
            self.assertEqual(spline.call_count, 1)                                    # the surrogate's alone
            np.testing.assert_allclose(spline.call_args.args[2], np.minimum(2 * (source['best_logl'] - np.array(source['logl'])), 25.0))
            diagnostics = json.loads((Path(d)/'surface_diagnostics.json').read_text())
            self.assertTrue(any('1 node(s) have no true value' in w for w in diagnostics['warnings']), diagnostics['warnings'])
            self.assertEqual(len(diagnostics['spline_min_delta_chi2']), 1)
            self.assertIn('linear on the nodes with values', diagnostics['isocontour_interpolation'])

    def test_a_surface_without_refused_nodes_is_splined_as_before(self):
        from profiles.io import save
        from profiles.surface_reporting import write_surface_outputs
        from scipy.interpolate import RectBivariateSpline

        sb, truth, bounds, cfg, source, exact = study()
        with tempfile.TemporaryDirectory() as d:
            result, _ = run(d, sb, truth, bounds, cfg, source)
            save(Path(d)/'surrogate_surface.json', source)
            save(Path(d)/'ranges.json', {n: dict(clipped=False) for n in ('x', 'y')})
            save(Path(d)/'counts.json', aggregate_counts(Path(d)/'calls'))
            with mock.patch('scipy.interpolate.RectBivariateSpline', wraps=RectBivariateSpline) as spline:
                write_surface_outputs(d)
            self.assertEqual(spline.call_count, 3)
            diagnostics = json.loads((Path(d)/'surface_diagnostics.json').read_text())
            self.assertFalse(any('no true value' in w for w in diagnostics['warnings']), diagnostics['warnings'])
            self.assertEqual(len(diagnostics['spline_min_delta_chi2']), 3)

    def test_joint_scan_truth_bounds_and_accounting(self):
        sb, truth, bounds, cfg, source, exact = study()
        # Surrogate conditional nuisance differs from truth; both obey the same bounds.
        x,y = np.meshgrid(*source['axes'], indexing='ij')
        np.testing.assert_allclose(np.array(source['xopt'])[:,:,2], np.clip(x+y+.35, *bounds[2]), atol=1e-5)
        self.assertTrue(source['quality']['converged'])
        with tempfile.TemporaryDirectory() as d:
            result, ledger = run(d, sb, truth, bounds, cfg, source)
            np.testing.assert_allclose(result['logl'], exact, atol=1e-6)
            np.testing.assert_array_equal(np.array(result['xopt'])[:,:,0], x)
            np.testing.assert_array_equal(np.array(result['xopt'])[:,:,1], y)
            events = [json.loads(l) for l in (Path(d)/'calls/attempt_0000.jsonl').read_text().splitlines()]
            starts = {e['call']: e for e in events if e['event']=='start'}
            returns = {e['call']: e for e in events if e['event']=='return'}
            self.assertEqual(starts.keys(), returns.keys())
            self.assertEqual(len(starts), ledger.counts['backend_calls'])
            for number, e in starts.items():
                p = np.array(e['x'])
                self.assertTrue(np.all(p>=bounds[:,0]) and np.all(p<=bounds[:,1]))
                self.assertEqual(returns[number]['value'], truth(p))
                if '/surface/' in e['stage']:
                    i,j = map(int, e['stage'].split('/')[-2:])
                    np.testing.assert_array_equal(p[:2], [source['axes'][0][i],source['axes'][1][j]])
            self.assertFalse(result['accuracy_certified'])

    def test_start_a_few_ulps_outside_the_box_is_clipped(self):
        # At (x, y) = (0.9, 0.8) the surrogate optimum sits on the z upper bound; a scan can
        # report it a few ulps beyond, which the ledger would reject.
        sb, truth, bounds, cfg, source, exact = study()
        source = json.loads(json.dumps(source, default=lambda a: np.asarray(a).tolist()))
        self.assertEqual(source['xopt'][2][2][2], 1.2)
        source['xopt'][2][2][2] = float(np.nextafter(1.2, 2.0))
        with tempfile.TemporaryDirectory() as d:
            result, ledger = run(d, sb, truth, bounds, cfg, source)
            self.assertEqual(ledger.counts['invalid_requests'], 0)
            node = json.loads((Path(d)/'true_surface/node_0002_0002.json').read_text())
            self.assertTrue(np.isfinite(node['direct_logl']))
            np.testing.assert_allclose(result['logl'], exact, atol=1e-6)

    def test_workflow_stops_on_an_unconverged_hybrid_scan(self):
        # The hybrid scan returns unconverged results instead of raising; the workflow must
        # stop before the true stage (no backend call) and keep the scan as diagnostics.
        from types import SimpleNamespace
        from unittest import mock
        from profiles.io import save
        from profiles.surface_workflow import execute_surface
        sb, truth, bounds, cfg, source, exact = study()
        cfg = ProfileConfig(dimension=2, parameters=['x', 'y'], profile_points=3,
                            ranges={'x': [-.8, .9], 'y': [-.7, .8]})
        md = SimpleNamespace(param_names=['x', 'y', 'z'],
                             bounds=dict(zip(['x', 'y', 'z'], bounds.tolist())))

        class Refined(Exception):
            pass

        def refine(*a, **k):
            raise Refined

        def ledger(*a, **k):
            raise AssertionError('backend called before the scan was checked')

        for converged in (False, True):
            result = dict(source, quality=dict(source['quality'], converged=converged))

            def scan(model, bounds, data, indices, axes, config, checkpoint=None, **k):
                save(checkpoint, result)
                return result

            with tempfile.TemporaryDirectory() as d, \
                    mock.patch('profiles.hybrid_surface.scan_surface_hybrid', scan), \
                    mock.patch('profiles.hybrid_surface.refine_surface_newton', refine):
                with self.assertRaisesRegex(Refined if converged else RuntimeError,
                                            '' if converged else 'stationarity/stability'):
                    execute_surface(Path(d), cfg, np.zeros((1, 4)), None, md, ledger, quiet)
                self.assertEqual((Path(d)/'failed_surrogate_diagnostics.json').exists(), not converged)
                self.assertEqual((Path(d)/'surrogate_surface.json').exists(), converged)

    def test_serial_parallel_resume_and_changed_inputs(self):
        sb, truth, bounds, cfg, source, exact = study()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            serial, _ = run(a, sb, truth, bounds, cfg, source)
            cfg.workers=3
            parallel, ledger = run(b, sb, truth, bounds, cfg, source)
            for key in ['logl','xopt','best_logl','best_x','diagnostics']:
                self.assertEqual(serial[key], parallel[key])
            before=(Path(b)/'calls/attempt_0000.jsonl').read_bytes()
            _, resumed = run(b, sb, truth, bounds, cfg, source)
            self.assertEqual(resumed.counts['backend_calls'],0)
            self.assertEqual(before, (Path(b)/'calls/attempt_0000.jsonl').read_bytes())
            cfg.corrected_tol *= 2
            with self.assertRaisesRegex(ValueError, 'checkpoint inputs'):
                run(b, sb, truth, bounds, cfg, source)

    def test_short_chains_serial_parallel_and_signature(self):
        sb, truth, bounds, cfg, source, exact = study()
        cfg.chain_length = 0  # every node of every row independent
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            serial, _ = run(a, sb, truth, bounds, cfg, source)
            cfg.workers = 3
            parallel, _ = run(b, sb, truth, bounds, cfg, source)
            for key in ['logl', 'xopt', 'best_logl', 'best_x', 'diagnostics']:
                self.assertEqual(serial[key], parallel[key])
            np.testing.assert_allclose(serial['logl'], exact, atol=1e-5)
            self.assertEqual(json.loads((Path(a)/'true_surface/method.json').read_text())['chain_length'], 0)
            cfg.chain_length = 1  # another plan: its checkpoints may not be reused
            with self.assertRaisesRegex(ValueError, 'checkpoint inputs'):
                run(b, sb, truth, bounds, cfg, source)
        cfg.chain_length = None  # the plan before chain_length leaves the signature as it was
        with tempfile.TemporaryDirectory() as c:
            run(c, sb, truth, bounds, cfg, source)
            self.assertNotIn('chain_length', json.loads((Path(c)/'true_surface/method.json').read_text()))

    def test_interrupted_resume_and_budget(self):
        sb, truth, bounds, cfg, source, exact = study()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            expected,_=run(a,sb,truth,bounds,cfg,source)
            with self.assertRaises(CallBudgetExceeded):
                run(b,sb,truth,bounds,cfg,source,budget=28)
            result, ledger=run(b,sb,truth,bounds,cfg,source)
            np.testing.assert_allclose(result['logl'],expected['logl'],atol=1e-12)
            coords=[]
            for journal in sorted((Path(b)/'calls').glob('attempt_*.jsonl')):
                events=[json.loads(l) for l in journal.read_text().splitlines()]
                completed={e['call'] for e in events if e['event']=='return'}
                coords.extend(tuple(e['x']) for e in events if e['event']=='start' and e['call'] in completed)
            self.assertEqual(len(coords),len(set(coords)))
            self.assertGreater(ledger.counts['cache_hits'],0)

    def test_two_dimensions_no_nuisance_and_bad_constraints(self):
        bounds=np.array([[-2,2]]*2)
        sb=Quadratic(bounds,np.eye(2),np.zeros(2))
        with tempfile.TemporaryDirectory() as d:
            ledger=EvaluationLedger(lambda x:float(-.5*np.dot(x,x)), bounds,
                                    journal=Path(d)/'calls.jsonl',max_calls=100)
            solver=NewtonRefiner(sb,ledger,NewtonConfig(),quiet)
            v,x,diag,_=solver.solve([0,0],[1,0],[.3,-.2])
            np.testing.assert_array_equal(x,[-.2,.3])
            self.assertAlmostEqual(v,-.065)
            self.assertEqual(ledger.counts['backend_calls'],1)
            for ids,vals in [([0,0],[0,0]),([0,2],[0,0]),([0,1],[0,3]),([.1,1],[0,0])]:
                with self.assertRaises(ValueError): solver.solve([0,0],ids,vals)
        for ids,axes in [([0,0],[[0],[0]]),([0,1],[[0,0],[0]]),([0,1],[[3],[0]])]:
            with self.assertRaises(ValueError): validate_grid(bounds,ids,axes)

    def test_curved_ridge_with_reversed_pair_order(self):
        from test_hybrid import Analytic
        bounds=np.array([[-1.,1.],[-1.,1.],[-.5,1.2]])
        f=lambda x: -.5*((x[0]*x[1])**2+(x[2]-x[0]**2)**2)
        g=lambda x: np.array([-x[0]*x[1]**2+2*x[0]*(x[2]-x[0]**2),
                              -x[1]*x[0]**2, -(x[2]-x[0]**2)])
        h=lambda x: np.array([[-x[1]**2+2*x[2]-6*x[0]**2,-2*x[0]*x[1],2*x[0]],
                              [-2*x[0]*x[1],-x[0]**2,0],[2*x[0],0,-1]])
        sb=Analytic(bounds,f,g,h)
        X=np.random.default_rng(23).uniform(bounds[:,0],bounds[:,1],(16,3))
        data=np.column_stack([X,[f(x) for x in X]])
        cfg=ProfileConfig(dimension=2,parameters=['y','x'],workers=2,corrected_tol=1e-6)
        axes=[[-.8,0,.7],[-.9,.1,.8]]
        source=scan_surface_hybrid(None,bounds,data,[1,0],axes,cfg,surrogate=sb,log=quiet)
        y,x=np.meshgrid(*axes,indexing='ij')
        np.testing.assert_allclose(source['logl'],-.5*(x*y)**2,atol=1e-7)
        np.testing.assert_allclose(np.asarray(source['xopt'])[:,:,2],x*x,atol=1e-4)
        with tempfile.TemporaryDirectory() as d:
            result,_=run(d,sb,f,bounds,cfg,source)
            np.testing.assert_allclose(result['logl'],-.5*(x*y)**2,atol=1e-7)
            np.testing.assert_array_equal(np.asarray(result['xopt'])[:,:,0],x)
            np.testing.assert_array_equal(np.asarray(result['xopt'])[:,:,1],y)

    def test_no_nuisance_scan(self):
        bounds=np.array([[-2.,2.]]*2)
        sb=Quadratic(bounds,np.eye(2),np.zeros(2))
        data=np.array([[0,0,0],[.2,.1,-.025]])
        cfg=ProfileConfig(dimension=2,parameters=['x','y'])
        result=scan_surface_hybrid(None,bounds,data,[0,1],[[-1.,0,1.],[-1.,0,1.]],cfg,
                                  surrogate=sb,log=quiet)
        x,y=np.meshgrid(*result['axes'],indexing='ij')
        np.testing.assert_allclose(result['logl'],-.5*(x*x+y*y),atol=1e-12)

    def test_surface_methods_is_gone(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.yaml'
            p.write_text('dimension: 2\nparameters: [x, y]\nsurface_methods: legacy\n')
            with self.assertRaisesRegex(ValueError,'unknown profile settings'):
                ProfileConfig.load(p)
