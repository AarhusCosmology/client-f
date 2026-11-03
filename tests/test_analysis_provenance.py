"""HPC chains cannot be reused across saved-model, domain, region or box changes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from sampling.provenance import target_identity, cached_samples, save_samples
from sampling.targets import resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc/analysis'))


class ChainIdentity(unittest.TestCase):
    def test_cache_requires_exact_target_and_preserves_replaced_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'trained_models').mkdir()
            model = run/'trained_models/model_it_4.keras'
            model.write_bytes(b'old network')
            (run/'metadata.json').write_text('{}')
            md = SimpleNamespace(param_names=['x'], bounds={'x': [-1., 1.]})
            target = target_identity(run, 4, md, None)
            file = run/'posterior_it_4.npz'
            x = np.linspace(-.9, .9, 20)[:, None]
            np.savez(file, x=x, names=['x'])  # previous versions, even a matching iteration
            self.assertIsNone(cached_samples(file, target))
            save_samples(file, x, ['x'], target)
            self.assertEqual(len(list(run.glob('*.previous-*.npz'))), 1)
            np.testing.assert_allclose(cached_samples(file, target), x)
            # The report samples every network again each time it runs: a chain of the same
            # target is replaced, not archived once more.
            save_samples(file, 0.5 * x, ['x'], target)
            self.assertEqual(len(list(run.glob('*.previous-*.npz'))), 1)
            np.testing.assert_allclose(cached_samples(file, target), 0.5 * x)
            x = 0.5 * x
            for key, value in [('model_sha256', 'other'), ('domain', [[-2., 2.]]),
                               ('lower', [-.5]), ('metadata_sha256', 'other'),
                               ('region', {'sha256': 'region', 'edge': 2.}), ('temperature', 3.)]:
                with self.subTest(key=key):
                    changed = dict(target, **{key: value})
                    self.assertIsNone(cached_samples(file, changed))
            model.write_bytes(b'new network')
            self.assertIsNone(cached_samples(file, target_identity(run, 4, md, None)))
            np.savez(file, x=x, names=['wrong'], target_json=json.dumps(target))
            self.assertIsNone(cached_samples(file, target))

    def test_sampling_version_rejects_old_mixtures_and_preserves_diagnostics(self):
        from sampling.provenance import sampler_identity, sampling_correction_problem
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'trained_models').mkdir()
            (root / 'trained_models/model_it_4.keras').write_bytes(b'model')
            (root / 'metadata.json').write_text('{}')
            target = target_identity(root, 4, SimpleNamespace(param_names=['x'], bounds={'x': [-1., 1.]}), None)
            path = root / 'posterior_it_4.npz'
            x = np.linspace(-.9, .9, 20)[:, None]
            save_samples(path, x, ['x'], target)
            single, mixture = sampler_identity(None), sampler_identity([0, 1])
            np.testing.assert_allclose(cached_samples(path, target, sampler=single), x)
            self.assertIsNone(cached_samples(path, target, sampler=mixture))
            info = dict(mixture, cells=[dict(cell=0, efficiency=.001)], unreliable_cells=[0])
            save_samples(path, x, ['x'], target, sampling=info)
            self.assertEqual(len(list(root.glob('*.previous-*.npz'))), 1)
            read = {}
            np.testing.assert_allclose(cached_samples(path, target, sampler=mixture, diagnostics=read), x)
            self.assertEqual(read, info)
            self.assertIn('unreliable region-cell', sampling_correction_problem(read))
            self.assertIsNone(cached_samples(path, target, sampler=dict(mixture, version=2)))
            save_samples(path, x, ['x'], target, sampling=info)
            self.assertEqual(len(list(root.glob('*.previous-*.npz'))), 1)
            np.savez(path, x=x, names=['x'], target_json=json.dumps(target), sampling_json='[]')
            self.assertIsNone(cached_samples(path, target, sampler=mixture))
            # also for a consumer that takes a chain without a record: a record that cannot be
            # read is not the same as none (asked of the mixture alone, this passed either way)
            self.assertIsNone(cached_samples(path, target, sampler=single))
            self.assertIsNone(cached_samples(path, target))

    def test_malformed_reliability_metadata_withholds_corrections(self):
        from sampling.provenance import sampler_identity, sampling_correction_problem
        base = sampler_identity([0, 1])
        for info in (base, dict(base, cells=[{}], unreliable_cells=[]),
                     dict(base, cells=[dict(cell=0, efficiency=.9)], unreliable_cells=[{}]),
                     dict(base, cells=[dict(cell=float('inf'), efficiency=0.)], unreliable_cells=[])):
            self.assertIsNotNone(sampling_correction_problem(info))
        # The efficiency itself also controls reliability, not just the stored flag.
        info = dict(base, cells=[dict(cell=0, efficiency=.001)], unreliable_cells=[])
        self.assertIn('unreliable region-cell', sampling_correction_problem(info))

    def test_report_inherits_earlier_region_without_configuration_hint(self):
        from report import Run
        from test_profile_region import add_region
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            add_region(run, inflate=1.5, iteration=2)
            add_region(run, inflate=3., iteration=8)
            report = Run.__new__(Run)
            report.dir = run
            report._model = (None, None)
            report._metadata_sha256 = "metadata"
            from sampling.targets import load
            entry = resolve(run, 4, "run")
            pinned = SimpleNamespace(model=object(), metadata=object(), region=load(run, entry))
            with mock.patch('sampling.targets.load_snapshot', return_value=(pinned, {"metadata_sha256": "metadata"})), \
                    mock.patch('likelihood.surrogate.SurrogateLikelihood', side_effect=lambda **kw: kw):
                self.assertAlmostEqual(report.surrogate(4, region=True).region.inflate, 1.5)
                self.assertNotIn('region', report.surrogate(4, region=False))
            self.assertEqual(resolve(run, 4, 'run')['file'], 'boost/region_it_2.npz')

    def test_snapshot_and_report_pin_model_metadata_and_region(self):
        from test_evidence import cobaya_run, quadratic_model
        from test_profile_region import add_region
        from sampling.targets import load_snapshot
        from report import Run
        from run.storage import digest

        with tempfile.TemporaryDirectory() as tmp:
            run = cobaya_run(tmp, quadratic_model([0., 0.], [.2, .2], 0.), [-1., -1.], [1., 1.])
            add_region(run, inflate=1.5, iteration=1)
            first, target = load_snapshot(run, 2)
            report = Run.__new__(Run)
            report.dir, report._model = run, (None, None)
            report._metadata_sha256 = digest(run / "metadata.json")
            self.assertEqual(report.target_identity(2), target)
            quadratic_model([0., 0.], [.2, .2], 10.).save(run / "trained_models/model_it_2.keras")
            add_region(run, inflate=3., iteration=2)
            second, changed = load_snapshot(run, 2)
            self.assertNotEqual(target["model_sha256"], changed["model_sha256"])
            self.assertNotEqual(target["region"], changed["region"])
            self.assertEqual(report.target_identity(2), target)
            np.testing.assert_allclose(report.predict(2, [[0., 0.]]), [0.])
            np.testing.assert_allclose(first.network(np.zeros((1, 2), np.float32)), [0.])
            np.testing.assert_allclose(second.network(np.zeros((1, 2), np.float32)), [10.])
            self.assertAlmostEqual(report.surrogate(2, region=True).region.inflate, 1.5)

    def test_snapshot_refuses_changes_during_copy(self):
        from test_evidence import cobaya_run, quadratic_model
        from sampling.targets import load_snapshot
        import shutil

        for asset in ("model", "metadata", "region"):
            with self.subTest(asset=asset), tempfile.TemporaryDirectory() as tmp:
                run = cobaya_run(tmp, quadratic_model([0., 0.], [.2, .2], 0.), [-1., -1.], [1., 1.])
                from test_profile_region import add_region
                add_region(run, inflate=1.5, iteration=1)
                copy = shutil.copyfile
                def changed(src, dst):
                    result = copy(src, dst)
                    if Path(src).name == "metadata.json":
                        if asset == "model":
                            quadratic_model([0., 0.], [.2, .2], 10.).save(run / "trained_models/model_it_2.keras")
                        elif asset == "metadata":
                            (run / "metadata.json").write_text((run / "metadata.json").read_text() + "\n")
                        else:
                            add_region(run, inflate=3., iteration=2)
                    return result
                with mock.patch('shutil.copyfile', side_effect=changed), self.assertRaisesRegex(RuntimeError, "changed while"):
                    load_snapshot(run, 2)

    def test_evidence_uses_loaded_identity_when_source_replaced_during_sampling(self):
        from test_evidence import cobaya_run, quadratic_model, command
        from evidence_summary import collect
        from sampling.targets import load_snapshot
        from run.storage import digest

        with tempfile.TemporaryDirectory() as tmp:
            run = cobaya_run(tmp, quadratic_model([0.], [.2], 0.), [-1.], [1.])
            model = run / "trained_models/model_it_2.keras"
            old_hash = digest(model)
            def nested(log_prob, lo, hi, **kw):
                x = np.linspace(-.5, .5, 21, dtype=np.float32)[:, None]
                lp = np.asarray(log_prob(x), float)
                self.assertEqual(float(lp[10]), 0.)
                quadratic_model([0.], [.2], 10.).save(model)
                return dict(lnZ_box=-1., sigma=.1, H=1., points=x, weights=np.full(21, 1/21), logL=lp,
                            live_fraction=.0001, n_dead=21, iteration_cap_reached=False, seconds=.1, device='toy')
            with mock.patch('evidence.sampler.import_best'), mock.patch('evidence.sampler.run_nested', side_effect=nested):
                command(run, '--n-live', '100', '--quiet')
            e = json.loads((run / 'evidence/evidence_it_2_nlive100.json').read_text())
            self.assertEqual(e['model_sha256'], old_hash)
            self.assertEqual(e['target']['model_sha256'], old_hash)
            self.assertNotEqual(digest(model), old_hash)
            _, newer = load_snapshot(run, 2)
            (run / 'analysis').mkdir()
            x = np.linspace(-.3, .3, 20)[:, None]
            network = 10. - .5 * (x[:, 0] / .2)**2
            np.savez(run / 'analysis/true_check_it_2.npz', x=x, network=network, truth=network+1.,
                     target_json=json.dumps(newer))
            self.assertNotIn('correction', collect(run)[2]['best'][0])

    def test_derived_chain_sidecar_uses_the_sampled_snapshot(self):
        from test_evidence import cobaya_run, quadratic_model
        from derived.chain import chain_with_derived
        from run.storage import digest

        with tempfile.TemporaryDirectory() as tmp:
            run = cobaya_run(tmp, quadratic_model([0.], [.2], 0.), [-1.], [1.])
            model = run / "trained_models/model_it_2.keras"
            old_hash = digest(model)
            sampling = SimpleNamespace(n_steps=2, n_walkers=4, burn_in=0)
            derived = SimpleNamespace(names=["y"], labels=["y"])
            def sample(surrogate, *args, **kwargs):
                chain = np.zeros((2, 4, 1), np.float32)
                lp = np.asarray(surrogate.logpost(chain.reshape(-1, 1))).reshape(2, 4)
                quadratic_model([0.], [.2], 10.).save(model)
                return chain, lp
            with mock.patch('config.config.Config.from_yaml', return_value=SimpleNamespace(sampling=sampling)), \
                    mock.patch('derived.chain.sample_posterior', side_effect=sample), \
                    mock.patch('derived.chain.predict_derived', return_value=np.zeros((8, 1))):
                root, _, lp, _ = chain_with_derived(run, 2, object(), derived)
            np.testing.assert_array_equal(lp, 0.)
            self.assertEqual(json.loads(root.with_suffix('.target.json').read_text())['model_sha256'], old_hash)
            self.assertEqual(json.loads(root.with_suffix('.sampling.json').read_text())['target']['model_sha256'], old_hash)
            self.assertNotEqual(digest(model), old_hash)

    def test_region_start_does_not_require_an_ordinary_chain_summary(self):
        from report import sample_posterior
        region = mock.MagicMock()                                     # one ellipsoid
        region.__len__.return_value = 1
        region.start.return_value = np.zeros((4, 2))
        surrogate = SimpleNamespace(region=region, logpost=lambda x: x)
        run = SimpleNamespace(ndim=2, surrogate=lambda *a, **k: surrogate)
        sampler = mock.Mock()
        sampler.chain.return_value = np.zeros((4, 2, 2))
        sampler.acceptance_fraction.return_value = np.full(4, .5)
        with mock.patch('sampling.base.build_sampler', return_value=sampler):
            _, acceptance = sample_posterior(run, 4, 20, 4, np.random.default_rng(1))
        region.start.assert_called_once()
        self.assertEqual(acceptance, .5)

    def test_a_region_of_several_ellipsoids_is_sampled_cell_by_cell_in_the_report(self):
        # One ensemble started on every ellipsoid kept the shares of its start: the report's
        # posterior and the true check's samples of a network with two modes had them.
        import tensorflow as tf
        from report import sample_posterior
        from sampling.region import Region

        rng = np.random.default_rng(42)
        clouds = [rng.normal([m, 0], .3, (500, 2)) for m in (-3., 3.)]
        region = Region.from_clouds(clouds, [-6, -6], [6, 6], 2, 0)

        def predict(x):
            a = -.5 * tf.reduce_sum(((x - [-3., 0.]) / .3) ** 2, axis=1) + np.log(.9)
            b = -.5 * tf.reduce_sum(((x - [3., 0.]) / .3) ** 2, axis=1) + np.log(.1)
            return tf.reduce_logsumexp(tf.stack([a, b]), axis=0)

        surrogate = SimpleNamespace(region=region, logpost=region.restrict(predict))
        run = SimpleNamespace(ndim=2, lower=np.array([-6., -6.]), upper=np.array([6., 6.]), surrogate=lambda *a, **k: surrogate)
        tf.keras.utils.set_random_seed(3)
        chain, acceptance = sample_posterior(run, 7, 4000, 32, np.random.default_rng(1))
        self.assertEqual(chain.shape[1:], (32, 2))
        self.assertAlmostEqual(np.mean(chain[:, :, 0] < 0), .9, delta=.03)           # equal starts would give one half
        cells = run.mixtures[7]["cells"]
        self.assertAlmostEqual(cells[0]["weight"], .9, delta=.025)
        self.assertEqual(run.mixtures[7]["unreliable_cells"], [])
        self.assertTrue(0 < acceptance < 1)

        # What the sampler itself recorded, as the report saves it, through the cache and
        # every reader of it (the tests of those had records made up for them).
        from chain_sampling import chain_sampling
        from sampling.provenance import sampler_identity, sampling_correction_problem

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'trained_models').mkdir()
            (root / 'boost').mkdir()
            (root / 'trained_models/model_it_7.keras').write_bytes(b'model')
            (root / 'metadata.json').write_text('{}')
            region.save(root / 'boost/region_it_7.npz')
            target = target_identity(root, 7, SimpleNamespace(param_names=['x', 'y'], bounds={'x': [-6., 6.], 'y': [-6., 6.]}),
                                     dict(file='boost/region_it_7.npz', sha256='the region', edge=2.))
            file, flat = root / 'posterior_it_7.npz', chain.reshape(-1, 2)
            sampler = sampler_identity(region)
            save_samples(file, flat, ['x', 'y'], target, sampling=dict(sampler, **run.mixtures[7]), acceptance=acceptance)
            read = {}
            np.testing.assert_allclose(cached_samples(file, target, sampler=sampler, diagnostics=read), flat, rtol=1e-6)
            self.assertEqual(read, json.loads(json.dumps(dict(sampler, **run.mixtures[7]))))
            self.assertIsNone(sampling_correction_problem(read))
            self.assertEqual(chain_sampling(file, root), (None, None))
            # the same chain as the report before 13 October wrote it: no record of the sampler
            old = root / 'old/posterior_it_7.npz'
            old.parent.mkdir()
            save_samples(old, flat, ['x', 'y'], target)
            stop, warning = chain_sampling(old, root)
            self.assertIn('does not record how it was sampled, and the region of network 7 has 2 ellipsoids', stop)
            self.assertIsNone(warning)
            self.assertEqual(chain_sampling(old), (None, None))                              # the run is not at hand
            # and one from before the target was recorded: the region is found by the network's number
            np.savez(old, x=flat, names=['x', 'y'])
            self.assertIn('has 2 ellipsoids', chain_sampling(old, root)[0])
            np.savez(root / 'old/posterior_it_6.npz', x=flat, names=['x', 'y'])              # no region at or before 6
            self.assertEqual(chain_sampling(root / 'old/posterior_it_6.npz', root), (None, None))
            # one cell marked by its efficiency alone: every reader says so
            marked = json.loads(json.dumps(dict(sampler, **run.mixtures[7])))
            marked['cells'][1]['efficiency'] = 0.05
            save_samples(file, flat, ['x', 'y'], target, sampling=marked)
            stop, warning = chain_sampling(file, root)
            self.assertIsNone(stop)
            self.assertIn('marks the weight of cell(s) [1] of its region as not reliable', warning)
            read = {}
            cached_samples(file, target, sampler=sampler, diagnostics=read)
            self.assertIn('cells [1]', sampling_correction_problem(read))

    def test_true_check_predictions_use_the_sampled_wall(self):
        from report import Run
        import tensorflow as tf
        report = Run.__new__(Run)
        surrogate = SimpleNamespace(network=lambda x: tf.fill([len(x)], 100.),
                                    logpost=lambda x: tf.fill([len(x)], -100.))
        report.surrogate = mock.Mock(return_value=surrogate)
        np.testing.assert_array_equal(report.predict(4, [[0, 0]], region=True), [-100.])
        report.surrogate.assert_called_with(4, region=True)
        np.testing.assert_array_equal(report.predict(4, [[0, 0]]), [100.])
