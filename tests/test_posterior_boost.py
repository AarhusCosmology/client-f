import unittest
from types import SimpleNamespace
import numpy as np
from dataset.dataset import TrainingDataset
from dataset.posterior_boost import select_posterior_boost

class PosteriorBoostTests(unittest.TestCase):
    def data(self,x):
        return TrainingDataset(x,-.5*np.sum(x*x,axis=1),SimpleNamespace(param_names=['x','y']),10,7.)

    def test_duplicate_states_preserve_mass_but_are_not_selected_twice(self):
        rng=np.random.default_rng(11); x=rng.normal(size=(1200,2))*3
        data=self.data(x); modes=[np.repeat(rng.normal(size=(4000,2)),3,axis=0)]
        points,info=select_posterior_boost(data,modes,150,seed=2,pool_factor=5)
        self.assertEqual(len(points),len(np.unique(points,axis=0)))
        self.assertEqual(info['max_multiplicity'],3)
        self.assertFalse(info['evidence_correction_valid'])
        self.assertEqual(data.target_temperature,7.)
        np.testing.assert_array_equal(data.inputs,x.astype(np.float32))

    def test_existing_coordinates_not_added_again_and_seed_is_local(self):
        rng=np.random.default_rng(12); x=rng.normal(size=(1200,2)); data=self.data(x)
        candidates=np.concatenate([x,rng.normal(size=(8000,2))]).astype(np.float32)
        np.random.seed(123); before=np.random.get_state()
        p,_=select_posterior_boost(data,[candidates],100,seed=3,pool_factor=10)
        after=np.random.get_state(); np.testing.assert_array_equal(before[1],after[1])
        stored={r.tobytes() for r in data.inputs}
        self.assertTrue(all(r.tobytes() not in stored for r in p))
        q,_=select_posterior_boost(data,[candidates],100,seed=3,pool_factor=10)
        np.testing.assert_array_equal(p,q)

    def test_shares_of_one_are_the_equal_target(self):
        rng=np.random.default_rng(21); data=self.data(rng.normal(size=(600,2))*4)
        clouds=[rng.normal(size=(3000,2))*.3+[-2,0],rng.normal(size=(2500,2))*.3+[2,0]]
        p,info=select_posterior_boost(data,clouds,120,seed=5,pool_factor=5)
        q,info_q=select_posterior_boost(data,clouds,120,seed=5,pool_factor=5,shares=[1.,1.])
        np.testing.assert_array_equal(p,q)
        self.assertEqual(info['target'],'equal-mode T=1 posterior')
        self.assertEqual(info_q['states_per_mode'],[2500,2500])

    def test_a_small_share_gets_a_small_part_of_the_points(self):
        rng=np.random.default_rng(22); data=self.data(rng.normal(size=(600,2))*4)
        clouds=[rng.normal(size=(4000,2))*.3+[-2,0],rng.normal(size=(4000,2))*.3+[2,0]]
        p,_=select_posterior_boost(data,clouds,200,seed=6,pool_factor=5)
        q,info=select_posterior_boost(data,clouds,200,seed=6,pool_factor=5,shares=[1.,.1])
        self.assertEqual(info['states_per_mode'],[4000,400])
        self.assertGreater(np.mean(p[:,0]>0),.35)
        self.assertLess(np.mean(q[:,0]>0),.2)
        for wrong in ([1.],[1.,0.],[.5,.5],[1.,np.nan]):
            with self.assertRaises(ValueError):
                select_posterior_boost(data,clouds,200,seed=6,pool_factor=5,shares=wrong)

    def test_oversupplied_mode_is_not_forced_to_accept_equal_quota(self):
        rng=np.random.default_rng(13)
        left=rng.normal(size=(1900,2))*.2+[-3,0]; right=rng.normal(size=(100,2))*.2+[3,0]
        data=self.data(np.r_[left,right])
        clouds=[rng.normal(size=(6000,2))*.2+[-3,0],rng.normal(size=(6000,2))*.2+[3,0]]
        p,_=select_posterior_boost(data,clouds,200,seed=4,pool_factor=5)
        self.assertGreater(np.mean(p[:,0]>0),.9)

    def test_too_few_states_for_the_pool_shrink_the_pool_only_then(self):
        # One mode of 3000 distinct states, n 150, pool_factor 20: a pool of all 3000 left no
        # reference sample and select_points raised (the SIDR + DRMD World Cup run wc_low,
        # 25 October 2026). The pool now leaves min(training points, half the states).
        from dataset.acquisition import select_points
        rng=np.random.default_rng(15); data=self.data(rng.normal(size=(1200,2)))
        cloud=rng.normal(size=(3000,2)).astype(np.float32)
        with self.assertRaises(ValueError):
            select_points(data,np.sort(cloud,axis=0)[:,None,:],np.zeros((3000,1)),150,1.,20,batch_size=1,
                          rng=np.random.default_rng(0),exclude_existing=True)
        p,info=select_posterior_boost(data,[cloud],150,seed=5,pool_factor=20)
        self.assertEqual((info['pool_factor_requested'],info['pool_factor_used'],info['distinct_states']),(20,12,3000))
        self.assertGreater(len(p),0)
        self.assertEqual(len(p),len(np.unique(p,axis=0)))
        # the same cloud with a pool that fits is selected as before: same points, no new keys
        q,info=select_posterior_boost(data,[cloud],150,seed=5,pool_factor=12)
        np.testing.assert_array_equal(p,q)
        self.assertNotIn('pool_factor_used',info)
        # and fewer distinct states than the request still raises
        with self.assertRaises(ValueError):
            select_posterior_boost(data,[cloud[:100]],150,seed=5,pool_factor=20)
        # through the boost: the smaller pool is logged and recorded
        from unittest.mock import patch
        from config.config import BoostConfig
        from dataset.boost import draw_boost
        mode=dict(x=cloud,f=-.5*np.sum(cloud*cloud,axis=1),weight=1.,log_evidence=0.,log_evidence_spread=0.,
                  efficiency=1.,walkers=32,peak=np.zeros(2),peak_value=0.,acceptance=.4)
        cfg=BoostConfig.from_dict(dict(n=150,start_iteration=15,density_filter=True,pool_factor=20))
        lines=[]
        with patch('dataset.boost.posterior_modes',return_value=([mode],{})):
            points,draw=draw_boost(None,[-10,-10],[10,10],cloud,7.,cfg,seed=5,dataset=data,log=lines.append)
        np.testing.assert_array_equal(np.sort(points,axis=0),np.sort(p,axis=0))
        self.assertEqual(draw['selection']['pool_factor_used'],12)
        self.assertTrue(any('3000 distinct states' in line and 'pool factor 12' in line for line in lines))

    def test_rejects_invalid_cloud(self):
        data=self.data(np.random.default_rng(14).normal(size=(100,2)))
        with self.assertRaises(ValueError): select_posterior_boost(data,[np.full((20,2),np.nan)],10)

    def test_native_boost_filters_and_never_reports_biased_weight_correction(self):
        from unittest.mock import patch
        from config.config import BoostConfig
        from dataset.boost import draw_boost, finish_boost
        from pathlib import Path
        import tempfile
        rng=np.random.default_rng(22)
        data=self.data(rng.normal(size=(1200,2))*3)
        x=rng.normal(size=(8000,2)).astype(np.float32)
        mode=dict(x=x,f=-.5*np.sum(x*x,axis=1),weight=1.,log_evidence=0.,
                  log_evidence_spread=0.,efficiency=1.,walkers=32,peak=np.zeros(2),
                  peak_value=0.,acceptance=.4)
        cfg=BoostConfig.from_dict(dict(n=100,start_iteration=15,density_filter=True,pool_factor=5))
        with patch('dataset.boost.posterior_modes',return_value=([mode],{})):
            points,draw=draw_boost(None,[-10,-10],[10,10],x,7.,cfg,seed=2,
                                   dataset=data,log=lambda *_:None)
        self.assertEqual(len(points),100)
        self.assertEqual(len(np.unique(points,axis=0)),100)
        np.testing.assert_allclose(draw['predicted'],-.5*np.sum(points*points,axis=1))
        with tempfile.TemporaryDirectory() as tmp:
            report=finish_boost(Path(tmp),15,['x','y'],draw,points,draw['predicted'],log=lambda *_:None)
            # The error of the network at the boost points is printed and kept whatever the
            # selection, boost by boost, so that the user can decide when to stop; the weight
            # correction stays withheld for density-selected rows.
            lines=[]
            truth=draw['predicted']+np.where(np.arange(100)%2,.3,-.3)
            later=finish_boost(Path(tmp),16,['x','y'],draw,points,truth,log=lines.append)
            from dataset.boost import boost_errors
            self.assertEqual([i for i,_ in boost_errors(Path(tmp))],[15,16])
        self.assertFalse(report['evidence_correction_valid'])
        self.assertNotIn('corrected_weight',report['modes'][0])
        self.assertNotIn('effective_points',report['modes'][0])
        self.assertEqual(report['selection']['requested'],100)
        self.assertEqual((report['error']['n'],report['error']['rms']),(100,0.0))
        self.assertAlmostEqual(later['error']['sd'],.3,places=5)
        self.assertAlmostEqual(later['error']['rms'],.3,places=5)
        self.assertNotIn('effective_points',later['modes'][0])
        text='\n'.join(lines)
        self.assertIn('posterior weight correction withheld',text)
        self.assertIn('network 16 at its boost points, which it had not seen: network minus truth rms 0.3',text)
        self.assertIn('scatter at the boost points, boost by boost: 0, 0.3',text)

    def test_all_candidates_already_stored_accepts_nothing(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(3000, 2)).astype(np.float32)
        points, info = select_posterior_boost(self.data(x), [x], 20,
                                             seed=7, pool_factor=5)
        self.assertEqual(points.shape, (0, 2))
        self.assertEqual(info['selected'], 0)
        self.assertGreater(info['excluded_existing'], 0)

    def test_native_boost_stops_when_filter_accepts_nothing(self):
        from unittest.mock import patch
        from config.config import BoostConfig
        from dataset.boost import draw_boost
        rng = np.random.default_rng(32)
        x = rng.normal(size=(3000, 2)).astype(np.float32)
        cfg = BoostConfig(n=20, start_iteration=0, density_filter=True,
                          pool_factor=5)
        with patch('dataset.boost.posterior_modes',
                   return_value=([dict(x=x, f=np.zeros(len(x)))], {})):
            with self.assertRaisesRegex(RuntimeError, 'no unfiltered top-up'):
                draw_boost(None, [-10, -10], [10, 10], x, 7., cfg,
                           seed=8, dataset=self.data(x), log=lambda *_: None)

    def test_filter_config_requires_boolean(self):
        from config.config import BoostConfig
        for value in ('false',1,None):
            with self.assertRaises(ValueError):
                BoostConfig.from_dict(dict(n=10,start_iteration=0,density_filter=value))

if __name__=='__main__': unittest.main()
