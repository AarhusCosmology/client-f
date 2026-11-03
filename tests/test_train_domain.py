"""acquisition.boost.train_domain and region_quantile (training/domain.py, sampling/region.py).

The loop's use of them, under MPI and through the launcher and branch tool in hpc/, is
tests/mpi_train_domain_smoke.py.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config.config import BoostConfig
from sampling.region import FLOOR, Region, boost_region
from training.domain import TrainDomain, region_coverage_message, region_outside, saved_domain

NDIM = 4


def training_set(seed=0):
    """Points over a wide box, log L a Gaussian in units of 0.1 around 0.5, so that the points
    within a depth of the best are a cloud far smaller than the box."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(0, 1, size=(3000, NDIM)), 0.5 + 0.1 * rng.normal(size=(2000, NDIM))])
    y = -0.5 * (((x - 0.5) / 0.1) ** 2).sum(axis=1)
    return x, y


class Domain(unittest.TestCase):
    def test_the_points_near_the_best_reach_exactly_its_edge(self):
        x, y = training_set()
        domain = TrainDomain.build(x, y, depth=8.0)
        near = (y.max() - y) < 8.0
        self.assertEqual(domain.rows, int(near.sum()))
        self.assertAlmostEqual(float(domain.rho(x[near]).max()), 1.0, places=12)
        self.assertTrue(domain.inside(x[near], 1.0).all())
        far = (y.max() - y) > 60.0
        self.assertFalse(domain.inside(x[far], 1.0).any())            # the far points are not trained on
        self.assertGreater(domain.inside(x, 1.5).sum(), domain.inside(x, 1.0).sum())

    def test_saved_and_read_again_it_is_the_same(self):
        x, y = training_set()
        domain = TrainDomain.build(x, y, depth=8.0)
        with tempfile.TemporaryDirectory() as d:
            domain.save(Path(d) / "train_domain_it_5.npz", iteration=5)
            again = TrainDomain.load(Path(d) / "train_domain_it_5.npz")
        np.testing.assert_array_equal(again.rho(x), domain.rho(x))
        self.assertEqual((again.depth, again.top, again.rows), (domain.depth, domain.top, domain.rows))

    def test_too_few_points_near_the_best_is_an_error(self):
        x, y = training_set()
        with self.assertRaisesRegex(ValueError, "rows within"):
            TrainDomain.build(x, y, depth=1e-3)

    def test_a_training_reads_the_earliest_domain_up_to_its_network(self):
        x, y = training_set()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(saved_domain(d, 9), (None, None))
            TrainDomain.build(x, y, depth=8.0).save(Path(d) / "train_domain_it_5.npz")
            TrainDomain.build(x, y, depth=20.0).save(Path(d) / "train_domain_it_7.npz")
            path, domain = saved_domain(d, 9)
            self.assertEqual((path.name, domain.depth), ("train_domain_it_5.npz", 8.0))
            self.assertEqual(saved_domain(d, 4), (None, None))         # built for a later network

    def test_the_share_of_the_region_outside_the_domain(self):
        x, y = training_set()
        domain = TrainDomain.build(x, y, depth=8.0)
        cloud = x[(y.max() - y) < 2.0]                                  # the posterior's points
        lower, upper = np.zeros(NDIM), np.ones(NDIM)
        rng = np.random.default_rng(1)
        self.assertEqual(region_outside(domain, 1.0, Region.from_clouds([cloud], lower, upper, 1.2, best=0.0), rng), 0.0)
        self.assertGreater(region_outside(domain, 1.0, Region.from_clouds([cloud], lower, upper, 4.0, best=0.0), rng), 0.5)

    def test_volume_weights_unequal_ellipsoids_not_equal_cell_counts(self):
        region = Region([[.2, .5], [.7, .5]], [np.eye(2), np.eye(2)], [.01, .2],
                        [0., 0.], [1., 1.], 1., -100.)
        for centre, radius, expected in (([.2, .5], .02, 400/401),
                                          ([.7, .5], .22, 1/401)):
            domain = TrainDomain(centre, np.ones(2), np.eye(2), radius, 0., 0., 1)
            measured = region_outside(domain, 1., region, np.random.default_rng(21), n=40000,
                                      diagnostics=True)
            self.assertAlmostEqual(measured["fraction"], expected, delta=.001)
            self.assertEqual(measured["proposal"], "ellipsoids")
            self.assertAlmostEqual(measured["effective_samples"], 40000)

    def test_overlap_is_not_counted_twice(self):
        # In 1D intervals [0.1,0.5] and [0.3,0.9] have union length .8.
        # The domain [0.3,0.5] occupies .2: the outside fraction is .75.
        region = Region([[.3], [.6]], [[[1.]], [[1.]]], [.2, .3],
                        [0.], [1.], 1., -100.)
        domain = TrainDomain([.4], [1.], [[1.]], .1, 0., 0., 1)
        actual = region_outside(domain, 1., region, np.random.default_rng(4), n=50000)
        self.assertAlmostEqual(actual, .75, delta=.01)
        # Duplicate cells must leave the volume measure unchanged too.
        duplicate = Region([[.3], [.3]], [[[1.]], [[1.]]], [.2, .2],
                           [0.], [1.], 1., -100.)
        actual = region_outside(domain, 1., duplicate, np.random.default_rng(4), n=50000)
        self.assertAlmostEqual(actual, .5, delta=.01)

    def test_box_clipping_and_inflation(self):
        # Inflated interval [-.2,.2] intersects the box in [0,.2];
        # domain [-.1,.1] covers half of that intersection.
        region = Region([[0.]], [[[1.]]], [.1], [0.], [1.], 2., -100.)
        domain = TrainDomain([0.], [1.], [[1.]], .1, 0., 0., 1)
        result = region_outside(domain, 1., region, np.random.default_rng(5), n=30000, diagnostics=True)
        self.assertAlmostEqual(result["fraction"], .5, delta=.01)
        self.assertLess(result["hits"], result["proposals"])
        # A huge region uses box proposals, without slow rejection or shrinking.
        region.reach[:] = 100.
        result = region_outside(domain, 1., region, np.random.default_rng(5), n=30000, diagnostics=True)
        self.assertEqual(result["proposal"], "box")
        self.assertAlmostEqual(result["fraction"], .9, delta=.01)

    def test_non_diagonal_ellipsoids_in_physical_coordinates(self):
        whiten = np.array([[1., .7], [0., 2.]])
        region = Region([[.5, .5]], [whiten], [.1], [-10., 20.], [30., 22.], 2., -100.)
        domain = TrainDomain([10., 21.], [40., 2.], whiten, .1, 0., 0., 1)
        # Same shape and centre; doubling the radius quadruples its 2D volume.
        result = region_outside(domain, 1., region, np.random.default_rng(8), n=20000)
        self.assertAlmostEqual(result, .75, delta=.01)

    def test_no_hits_is_unknown_and_low_ess_is_not_a_coverage_claim(self):
        region = Region([[10.]], [[[1.]]], [.1], [0.], [1.], 1., -100.)
        domain = TrainDomain([0.], [1.], [[1.]], .1, 0., 0., 1)
        result = region_outside(domain, 1., region, np.random.default_rng(1), diagnostics=True)
        self.assertTrue(np.isnan(result["fraction"]))
        self.assertEqual(result["effective_samples"], 0)
        self.assertIn("unknown", region_coverage_message(result))
        self.assertIn("poorly resolved", region_coverage_message(dict(
            fraction=.5, effective_samples=10., proposals=4000)))


class TooFewInside(unittest.TestCase):
    def test_a_domain_that_does_not_belong_to_the_training_set_stops_with_its_name(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3))
        y = -0.5 * np.sum(x ** 2, axis=1)
        domain = TrainDomain.build(x, y, depth=8.0)
        np.testing.assert_array_equal(domain.kept(x, 1.0, "train_domain_it_3.npz"), domain.inside(x, 1.0))
        with self.assertRaisesRegex(ValueError, r"train_domain_it_3.npz: 0 of 200 points.*at least 6 needed"):
            domain.kept(x + 1000.0, 1.0, "train_domain_it_3.npz")

    def test_the_rows_that_built_it_are_enough_even_if_rounding_takes_the_farthest(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(7, 3))
        domain = TrainDomain.build(x, np.zeros(7), depth=1.0)
        domain.far *= 1 - 1e-12                                    # the farthest row is now just outside
        self.assertEqual(int(domain.kept(x, 1.0, "d").sum()), 6)


class Settings(unittest.TestCase):
    def test_absent_by_default_and_needing_the_region(self):
        base = dict(n=10, start_iteration=2)
        boost = BoostConfig.from_dict(base)
        self.assertIsNone(boost.train_domain)
        self.assertIsNone(boost.region_quantile)
        boost = BoostConfig.from_dict(dict(base, region=1.5, train_domain=1, region_quantile=0.999))
        self.assertEqual((boost.train_domain, boost.region_quantile), (1.0, 0.999))
        for wrong, message in ((dict(train_domain=1.25), "need acquisition.boost.region"),
                               (dict(region_quantile=0.9), "need acquisition.boost.region"),
                               (dict(region=1.5, train_domain=0.8), "train_domain"),
                               (dict(region=1.5, region_quantile=0.0), "region_quantile"),
                               (dict(region=1.5, region_quantile=1.5), "region_quantile"),
                               (dict(region=1.5, train_domain=True), "train_domain")):
            with self.assertRaisesRegex(ValueError, message):
                BoostConfig.from_dict(dict(base, **wrong))


class QuantileReach(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.cloud = 0.4 + 0.05 * rng.normal(size=(2000, NDIM))
        self.cloud[0] = 0.4 + 0.05 * 12.0                              # one far point
        self.lower, self.upper = np.zeros(NDIM), np.ones(NDIM)

    def test_without_a_quantile_the_reach_is_the_farthest_point_as_before(self):
        plain = Region.from_clouds([self.cloud], self.lower, self.upper, 2.0, best=0.0)
        same = Region.from_clouds([self.cloud], self.lower, self.upper, 2.0, best=0.0, quantile=None)
        for a, b in ((plain.means, same.means), (plain.whiten, same.whiten), (plain.reach, same.reach)):
            np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(float(plain.rho(self.cloud).max()), 1.0, places=6)

    def test_a_quantile_is_not_moved_by_one_far_point(self):
        q = Region.from_clouds([self.cloud], self.lower, self.upper, 2.0, best=0.0, quantile=0.999)
        rho = q.rho(self.cloud)
        self.assertAlmostEqual(float(np.quantile(rho, 0.999)), 1.0, places=5)
        self.assertGreater(rho[0], 2.0)                                 # the far point is beyond the wall
        plain = Region.from_clouds([self.cloud], self.lower, self.upper, 2.0, best=0.0)
        self.assertLess(q.reach[0], 0.5 * plain.reach[0])

    def test_boost_region_passes_it_on_and_records_it(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as d:
            frame = pd.DataFrame(self.cloud.astype(np.float32), columns=[f"p{k}" for k in range(NDIM)])
            frame["loglkl"], frame["predicted"], frame["mode"] = -1.0, -1.0, 0
            frame.to_csv(Path(d) / "boost_it_5.csv", index=False)
            names = [f"p{k}" for k in range(NDIM)]
            plain, info = boost_region(d, names, self.lower, self.upper, 2.0, before=6, best=0.0, depth=20.0)
            self.assertNotIn("quantile", info)
            q, info = boost_region(d, names, self.lower, self.upper, 2.0, before=6, best=0.0, depth=20.0, quantile=0.999)
            self.assertEqual(info["quantile"], 0.999)
            self.assertLess(q.reach[0], plain.reach[0])
            self.assertEqual(q.floor, plain.floor)
            self.assertEqual(plain.floor, -FLOOR)


if __name__ == "__main__":
    unittest.main()
