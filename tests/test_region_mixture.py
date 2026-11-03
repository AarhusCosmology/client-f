"""Regressions for supported clouds and current-network mixture masses."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampling.region import Region, boost_clouds, boost_region
from derived.chain import sample_posterior

class CloudRetention(unittest.TestCase):
    def write(self, root, iteration, groups):
        frame=pd.DataFrame(np.concatenate(groups),columns=['x','y','z'])
        frame['mode']=np.concatenate([np.full(len(g),i) for i,g in enumerate(groups)])
        frame['loglkl']=-1.
        frame.to_csv(Path(root)/f'boost_it_{iteration}.csv',index=False)

    def test_sparse_mode_is_retained_and_singleton_is_explicit(self):
        rng=np.random.default_rng(42)
        a=rng.normal(.2,.01,(50,3)); b=rng.normal(.8,.01,(4,3))
        with tempfile.TemporaryDirectory() as d:
            self.write(d,0,[a,b])
            r,info=boost_region(d,['x','y','z'],[0]*3,[1]*3,2,1,0,20)
            self.assertEqual(r.rows,[50,4]);self.assertEqual(info['small_clouds'],1)
            self.assertTrue(r.inside(np.concatenate([a,b])).all())
            np.testing.assert_array_equal(r.labels_tf(tf.constant(b,tf.float32)).numpy(),r.labels(b))
            # One verified point gives no shape. It raised here for a day, from the saved boost
            # files and inside the loop: at every restart. It is left out, counted, and the run goes on.
            self.write(d,0,[a,b[:1]])
            r,info=boost_region(d,['x','y','z'],[0]*3,[1]*3,2,1,0,20)
            self.assertEqual((r.rows,info['rows'],info['left_out'],info['small_clouds']),([50],[50],[1],0))
            with self.assertRaisesRegex(ValueError,'fewer than two distinct verified points'):
                Region.from_clouds([a,b[:1]],[0]*3,[1]*3,2,0)

    def test_new_group_merges_old_clouds_without_duplicating_rows(self):
        rng=np.random.default_rng(31)
        a=rng.normal(.3,.02,(40,3));b=rng.normal(.6,.02,(40,3))
        merged=np.concatenate([a,b])+rng.normal(0,.002,(80,3))
        with tempfile.TemporaryDirectory() as d:
            self.write(d,0,[a,b]);self.write(d,1,[merged])
            clouds,info=boost_clouds(d,['x','y','z'],2,0,20,[0]*3,[1]*3)
            self.assertEqual(info['merged_clouds'],1)
            self.assertEqual([len(c) for c in clouds],[160])

class SparseClouds(unittest.TestCase):
    """A mode known from a handful of verified rows, in a posterior whose widths differ by a
    factor of a hundred from one direction to another, as real ones do."""
    NDIM = 16
    WIDTHS = np.geomspace(0.001, 0.1, 16)

    def mode(self, rng, centre, n):
        return centre + rng.normal(size=(n, self.NDIM)) * self.WIDTHS

    def write(self, root, iteration, groups):
        names = [f"p{k}" for k in range(self.NDIM)]
        frame = pd.DataFrame(np.concatenate(groups), columns=names)
        frame["mode"] = np.concatenate([np.full(len(g), k + 1) for k, g in enumerate(groups)])
        frame["loglkl"] = -1.0
        frame.to_csv(Path(root) / f"boost_it_{iteration}.csv", index=False)
        return names

    def test_a_sparse_cloud_has_the_shape_of_the_well_sampled_one(self):
        # Shrunk towards a sphere in box units, its ellipsoid reached tens of times too far along
        # the narrow directions: the network was admitted where no boost point had verified it.
        rng = np.random.default_rng(3)
        lower, upper = np.zeros(self.NDIM), np.ones(self.NDIM)
        big, small = self.mode(rng, 0.3, 4000), self.mode(rng, 0.7, 4000)
        full = Region.from_clouds([small], lower, upper, 2.0, 0.0)
        for n in (2, 5, 32):
            region = Region.from_clouds([big, small[:n]], lower, upper, 2.0, 0.0)
            np.testing.assert_array_equal(region.whiten[1], region.whiten[0])          # the shape is the donor's
            self.assertLessEqual(region.reach[1], region.reach[0])                     # and it reaches no farther
            sparse = Region([region.means[1]], [region.whiten[1]], [region.reach[1]], lower, upper, 2.0, region.floor)
            inside = sparse.start(4000, rng)
            self.assertGreater(np.mean(full.inside(inside)), 0.99, n)                  # where the mode's own rows put it
            self.assertTrue(sparse.inside(small[:n]).all())
        # with more than twice ndim distinct rows it has its own shape, as before
        own = Region.from_clouds([big, small[:33]], lower, upper, 2.0, 0.0)
        self.assertFalse(np.array_equal(own.whiten[1], own.whiten[0]))
        alone = Region.from_clouds([small[:33]], lower, upper, 2.0, 0.0)
        np.testing.assert_array_equal(own.whiten[1], alone.whiten[0])
        np.testing.assert_array_equal(own.reach[1], alone.reach[0])

    def test_without_a_well_sampled_cloud_the_sparse_ones_wait_and_then_share_a_shape(self):
        rng = np.random.default_rng(4)
        lower, upper = np.zeros(self.NDIM), np.ones(self.NDIM)
        with tempfile.TemporaryDirectory() as d:
            names = self.write(d, 0, [self.mode(rng, 0.4, 20)])
            region, info = boost_region(d, names, lower, upper, 2.0, 1, 0.0, 20.0)
            self.assertIsNone(region)                                      # 20 rows in 16 dimensions: no shape yet
            self.assertEqual((info["rows"], info["left_out"]), ([], [20]))
            self.write(d, 1, [self.mode(rng, 0.4, 20)])
            region, info = boost_region(d, names, lower, upper, 2.0, 2, 0.0, 20.0)
            self.assertEqual((region.rows, info["left_out"], info["small_clouds"]), ([40], [], 0))    # one mode, joined
        with self.assertRaisesRegex(ValueError, "no well-sampled cloud to take a shape from"):
            Region.from_clouds([self.mode(rng, 0.4, 20)], lower, upper, 2.0, 0.0)

    def test_a_stray_row_does_not_join_two_modes_and_a_small_mode_joins_itself(self):
        rng = np.random.default_rng(5)
        lower, upper = np.zeros(self.NDIM), np.ones(self.NDIM)
        with tempfile.TemporaryDirectory() as d:
            # the second mode is seen in the first boost only, with three rows, one of them a
            # stray inside the first mode's cloud (as a heavy tail gives): the next boost has
            # the first mode alone. One row of overlap merged the two, and the cloud's reach
            # went from 10.6 to 29 on a real boost history.
            first = self.mode(rng, 0.3, 500)
            second = np.vstack([self.mode(rng, 0.7, 2), first[:1] + 1e-4])
            names = self.write(d, 0, [first, second])
            self.write(d, 1, [self.mode(rng, 0.3, 500)])
            clouds, info = boost_clouds(d, names, 2, 0.0, 20.0, lower, upper)
            self.assertEqual(([len(c) for c in clouds], info["merged_clouds"]), ([1000, 3], 0))
            region, info = boost_region(d, names, lower, upper, 2.0, 2, 0.0, 20.0)
            self.assertLess(region.reach[0], 8.0)
        with tempfile.TemporaryDirectory() as d:
            # two verified rows of the second mode in every boost: they are one mode, and were
            # one more ellipsoid per boost
            for it in range(5):
                names = self.write(d, it, [self.mode(rng, 0.3, 300), self.mode(rng, 0.7, 2)])
            clouds, info = boost_clouds(d, names, 5, 0.0, 20.0, lower, upper)
            self.assertEqual([len(c) for c in clouds], [1500, 10])

    def test_a_cell_inside_another_ellipsoid_gets_its_starting_points(self):
        # A small ellipsoid inside a large one: its cell is a small part of it, and uniform
        # draws in the ellipsoid found none in 200 tries (regions that runs had saved).
        rng = np.random.default_rng(6)
        lower, upper = np.zeros(self.NDIM), np.ones(self.NDIM)
        wide = self.mode(rng, 0.5, 2000) * 1.0
        nested = 0.5 + 0.05 * (self.mode(rng, 0.5, 400) - 0.5)
        region = Region.from_clouds([wide, 0.5 + 3.0 * (wide - 0.5), nested], lower, upper, 2.0, 0.0)
        for k in range(3):
            x = region.start(64, rng, component=k)
            self.assertEqual(len(np.unique(x, axis=0)), 64)
            np.testing.assert_array_equal(region.labels(x), k)
            self.assertTrue(np.all((x >= lower) & (x <= upper)))
        # without a component the draw is what it was
        a, b = region.start(100, np.random.default_rng(1)), region.start(100, np.random.default_rng(1))
        np.testing.assert_array_equal(a, b)


class MixtureMass(unittest.TestCase):
    def test_unequal_modes_get_current_target_masses(self):
        tf.keras.utils.set_random_seed(45)
        rng=np.random.default_rng(42)
        clouds=[rng.normal([m,0],.3,(500,2)) for m in [-3.,3.]]
        region=Region.from_clouds(clouds,[-6,-6],[6,6],2,0)
        def predict(x):
            a=-.5*tf.reduce_sum(((x-[-3.,0.])/.3)**2,axis=1)+np.log(.9)
            b=-.5*tf.reduce_sum(((x-[3.,0.])/.3)**2,axis=1)+np.log(.1)
            return tf.reduce_logsumexp(tf.stack([a,b]),axis=0)
        target=SimpleNamespace(region=region,logpost=region.restrict(predict))
        settings=SimpleNamespace(sampler='aies',n_walkers=32,burn_in=200,chunk_size=500)
        info={}
        chain,lp=sample_posterior(target,[[-6,6]]*2,settings,1000,5,seed=8,diagnostics=info)
        self.assertEqual(chain.shape,(200,32,2));self.assertEqual(lp.shape,(200,32))
        self.assertAlmostEqual(info['cells'][0]['weight'],.9,delta=.025)
        self.assertAlmostEqual(np.mean(chain[:,:,0]<0),.9,delta=.03)
        self.assertAlmostEqual(chain[:,:,0].mean(),-2.4,delta=.2)
        np.testing.assert_allclose(lp.ravel(),target.logpost(tf.constant(chain.reshape(-1,2))).numpy(),atol=1e-5)
        self.assertTrue(info['resampled'])
        self.assertEqual(info['unreliable_cells'],[])
        # drawn without replacement: no row of a cell twice (the two cells' rows are distinct sets)
        self.assertGreater(len(np.unique(chain.reshape(-1,2),axis=0)),0.85*chain.shape[0]*chain.shape[1])

    def test_overlapping_cells_do_not_double_count_target(self):
        tf.keras.utils.set_random_seed(51)
        rng=np.random.default_rng(7)
        clouds=[rng.normal([m,0],.5,(500,2)) for m in [-.4,1.2]]
        region=Region.from_clouds(clouds,[-6,-6],[6,6],4,0)
        predict=lambda x:-.5*tf.reduce_sum(x*x,axis=1)
        target=SimpleNamespace(region=region,logpost=region.restrict(predict))
        settings=SimpleNamespace(sampler='aies',n_walkers=32,burn_in=200,chunk_size=500)
        info={}
        chain,_=sample_posterior(target,[[-6,6]]*2,settings,1500,5,seed=9,diagnostics=info)
        x=chain.reshape(-1,2)
        np.testing.assert_allclose(x.mean(axis=0),0,atol=.08)
        np.testing.assert_allclose(x.var(axis=0),1,atol=.12)
        reference=rng.normal(size=(200000,2))
        mass=np.mean(region.labels(reference)==0)
        self.assertAlmostEqual(info['cells'][0]['weight'],mass,delta=.035)

if __name__=='__main__':unittest.main()
