import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.profile_augmentation import make_design, select_nodes, allocate_counts
from benchmarking.augment_profiles import group_validation
from benchmarking.compare_profile_augmentation import check_grids, refined_metrics


class DesignTests(unittest.TestCase):
    def setup_design(self, width=3.):
        rng = np.random.default_rng(10)
        x = rng.uniform(-width, width, (200, 3))
        y = -.5*np.sum(x*x, axis=1)
        g = np.linspace(-width, width, 25)
        profiles = dict(best_logl=0., best_x=[0.,0.,0.], profiles={
            'a':dict(grid=g, logl=-.5*g*g, xopt=np.column_stack([g,g*0,g*0]))})
        bounds = np.array([[-width,width]]*3)
        def optimise(i, v, starts):
            point = np.zeros(3); point[i]=v
            return -.5*v*v, point, dict(stationary=True, projected_gradient_inf=0.)
        return x,y,profiles,['a','b','c'],bounds,optimise

    def test_budget_pinning_distinctness_and_radius(self):
        x,y,p,names,b,opt = self.setup_design()
        points, manifest = make_design(x,y,p,names,b,300,opt,seed=4)
        self.assertEqual(len(points),300)
        self.assertEqual(len({r.tobytes() for r in points}),300)
        self.assertTrue(np.all((points >= b[:,0]) & (points <= b[:,1])))
        self.assertEqual(manifest['clipped_points'],0)
        near=0
        for point, rec in zip(points,manifest['draws']):
            node=manifest['nodes'][rec['node']]
            self.assertEqual(point[node['index']],node['value'])
            if rec['mode']=='near_ridge' and not rec['gibbs_fallback']:
                self.assertAlmostEqual(rec['realised_quadratic_deficit'], rec['requested_deficit'], places=10)
                near+=1
        self.assertGreater(near,100)
        repeated,_=make_design(x,y,p,names,b,300,opt,seed=4)
        np.testing.assert_array_equal(points,repeated)

    def test_truncated_fallback_has_no_clipping_atoms(self):
        x,y,p,names,b,opt = self.setup_design(width=.1)
        points, m = make_design(x,y,p,names,b,40,opt,seed=8,max_rejections=1)
        self.assertGreater(m['gibbs_fallbacks'],0)
        self.assertTrue(np.all((points[:,1:] > -.1) & (points[:,1:] < .1)))

    def test_crossings_receive_extra_allocation(self):
        x,y,p,names,b,opt = self.setup_design()
        nodes=select_nodes(p,names,b)
        counts=allocate_counts(nodes,1000,np.random.default_rng(0))
        self.assertEqual(sum(counts),1000)
        cross=[c for n,c in zip(nodes,counts) if n['weight']==4]
        broad=[c for n,c in zip(nodes,counts) if n['weight']==1]
        self.assertTrue(cross and broad)
        self.assertGreater(min(cross),max(broad))
        with self.assertRaises(ValueError):
            allocate_counts(nodes,-1,np.random.default_rng(0))

    def test_validation_holds_out_entire_nodes(self):
        manifest=dict(nodes=list(range(10)),draws=[dict(node=i) for i in range(10) for _ in range(4)])
        mask,held=group_validation(manifest,np.random.default_rng(2))
        self.assertEqual(len(held),1)
        for i in range(10):
            self.assertTrue(np.all(mask[i*4:(i+1)*4]==(i in held)))

    def test_refined_crossing_score_known_displaced_gaussian(self):
        g=np.linspace(-3,3,25)
        p=dict(names=['a'],best_logl=0.,best_x=[.1,0.],profiles={
            'a':dict(grid=g,logl=-.5*(g-.1)**2,xopt=np.column_stack([g,g*0]))})
        ref=dict(names=['a'],profiles={'a':dict(grid=g,dchi2=g*g)})
        def opt(i,value,starts):
            return -.5*(value-.1)**2,np.array([value,0]),dict(stationary=True,projected_gradient_inf=0.)
        r=refined_metrics(p,ref,opt,['a','b'])
        self.assertAlmostEqual(r['a']['1 sigma']['score'],.1,places=8)
        self.assertAlmostEqual(r['a']['2 sigma']['score'],.05,places=8)


class ReferenceGrid(unittest.TestCase):
    @staticmethod
    def profile(grid):
        return dict(names=['a'],profiles=dict(a=dict(grid=grid)))

    def test_another_grid_is_refused_before_anything_is_computed(self):
        reference=self.profile([0.,1.,2.])
        check_grids(self.profile([0.,1.,2.+1e-13]),reference,'profiles_0.json')
        for grid in ([0.,1.,2.5],[0.,2.]):
            with self.assertRaisesRegex(ValueError,"profiles_0.json: the grid of a is not the reference's"):
                check_grids(self.profile(grid),reference,'profiles_0.json')


class NotFinite(unittest.TestCase):
    def test_a_refused_point_is_null_in_the_json(self):
        import json
        from benchmarking.profile_likelihood import finite_or_none
        found=finite_or_none(dict(a=[1.,-np.inf,(np.nan,)],b=dict(c=np.float64('inf'),d='text',e=None,f=2)))
        self.assertEqual(found,dict(a=[1.,None,[None]],b=dict(c=None,d='text',e=None,f=2)))
        json.dumps(found,allow_nan=False)


class ReferenceChains(unittest.TestCase):
    def test_weights_are_kept_as_they_are(self):
        import tempfile
        from pathlib import Path
        from benchmarking.chains import load_cobaya_chains,load_montepython_chains
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'chain.1.txt').write_text('# weight minuslogpost a\n0.25 1.0 3.0\n1.75 2.0 5.0\n')
            samples,loglikes,weights=load_cobaya_chains(d,['a'])
            np.testing.assert_array_equal(weights[0],[.25,1.75])
            np.testing.assert_array_equal(samples[0].ravel(),[3.,5.])
            np.testing.assert_array_equal(loglikes[0],[-1.,-2.])
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'chain.1.txt').write_text('0.5 1.0 3.0\n2 2.0 5.0\n')
            samples,loglikes,weights=load_montepython_chains(d,['a'],scales=[2.])
            np.testing.assert_array_equal(weights[0],[.5,2.])
            np.testing.assert_array_equal(samples[0].ravel(),[6.,10.])


class ChainKey(unittest.TestCase):
    def test_another_network_region_setting_or_method_is_another_chain(self):
        from benchmarking.benchmark import chain_key
        target=dict(model_sha256='a',region=None)
        settings=dict(sampler='aies',n_walkers=64,burn_in=100)
        sampler=dict(schema=1,version=1,method='single ensemble')
        key=chain_key(target,settings,sampler)
        self.assertEqual(key,chain_key(dict(target,names=['x']),dict(settings,n_steps=7),sampler))
        self.assertNotEqual(key,chain_key(dict(target,model_sha256='b'),settings,sampler))
        self.assertNotEqual(key,chain_key(dict(target,region=dict(file='f')),settings,sampler))
        self.assertNotEqual(key,chain_key(target,dict(settings,n_walkers=128),sampler))
        self.assertNotEqual(key,chain_key(target,dict(settings,burn_in=200),sampler))
        self.assertNotEqual(key,chain_key(target,settings,dict(sampler,method='region-cell bridge mixture')))


class BenchmarkChainRecord(unittest.TestCase):
    """A reused chain keeps what the sampler said of its weights, and a resample is not walkers."""

    def run_chain(self,directory,sample):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest import mock
        from benchmarking.benchmark import load_or_run_chain
        surrogate=SimpleNamespace(region=[0,1])
        with mock.patch('derived.chain.sample_posterior',side_effect=sample) as sampler:
            found=load_or_run_chain(Path(directory),3,dict(sampler='aies',n_walkers=4,burn_in=1),8,1,
                                    dict(x=(-1.,1.)),surrogate,dict(model_sha256='a',region=None))
        return found,sampler.call_count

    def test_the_record_of_unreliable_weights_survives_the_cache(self):
        import tempfile
        from benchmarking.benchmark import sample_inputs,sampling_note
        cells=[dict(cell=0,efficiency=.6),dict(cell=1,efficiency=.01)]
        def sample(surrogate,bounds,settings,n_steps,thin,diagnostics=None):
            diagnostics.update(cells=cells,resampled=True,unreliable_cells=[1])
            return np.arange(8.).reshape(4,2,1),np.arange(8.).reshape(4,2)
        with tempfile.TemporaryDirectory() as d:
            (path,chain,lp,info),calls=self.run_chain(d,sample)
            self.assertEqual(calls,1)
            (again,_,_,kept),calls=self.run_chain(d,sample)
            self.assertEqual((calls,again),(0,path))
            self.assertEqual(kept,info)
            self.assertIn('unreliable region-cell mixture weights',sampling_note(kept))
            # An older file without the record is sampled again.
            np.savez(path,chain=chain,log_prob=lp)
            self.assertEqual(self.run_chain(d,sample)[1],1)
        samples,loglikes,convergence=sample_inputs(chain,lp,'aies',info)
        self.assertEqual((samples.shape,loglikes.shape,convergence),((8,1),(8,),False))

    def test_one_ensemble_is_still_walkers(self):
        from benchmarking.benchmark import sample_inputs,sampling_note
        info=dict(schema=1,version=1,method='single ensemble')
        samples,loglikes,convergence=sample_inputs(np.arange(8.).reshape(4,2,1),np.arange(8.).reshape(4,2),'aies',info)
        self.assertEqual((len(samples),convergence),(2,True))
        self.assertEqual(sampling_note(info),'Surrogate chain: single ensemble')


class PeakBias(unittest.TestCase):
    def test_a_refused_true_anchor_has_no_bias(self):
        from benchmarking.compare_profile_augmentation import peak_bias
        self.assertEqual(peak_bias(dict(true_at_best=None,best_logl=-1.)),None)
        self.assertEqual(peak_bias(dict(true_at_best=-1.5,best_logl=-1.)),-1.)


if __name__=='__main__':
    unittest.main()
