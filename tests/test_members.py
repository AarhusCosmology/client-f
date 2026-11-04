"""training.members (training/members.py): the pieces rank 0 and the member ranks use, and one
member trained on a small analytic training set. The loop itself under MPI, with members on
other ranks, is tests/mpi_members_smoke.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import TrainingConfig  # noqa: E402
from run.storage import phase_outputs  # noqa: E402
from training import members  # noqa: E402

TRAINING = dict(learning_rate=1e-3, loss="msre", sigma_level=3, n_epochs=4, batch_size=32, validation_split=0.2,
                patience=10)


class Ranks(unittest.TestCase):
    def test_spread_and_never_rank_zero(self):
        self.assertEqual(members.member_ranks(96, 4), [24, 48, 72])
        self.assertEqual(members.member_ranks(4, 3), [1, 3])
        self.assertEqual(members.member_ranks(3, 3), [1, 2])
        for size in range(1, 20):
            for k in range(1, 8):
                ranks = members.member_ranks(size, k)
                self.assertEqual(len(ranks), min(k - 1, size - 1))
                self.assertEqual(len(set(ranks)), len(ranks))
                self.assertTrue(all(0 < r < size for r in ranks))

    def test_warning_when_the_members_cannot_train_at_once(self):
        # 12 members at 8 threads fit on two 48-core nodes; on one node they share the cores
        self.assertIsNone(members.oversubscription(96, 12, 8, 8, 48))
        self.assertIsNone(members.oversubscription(48, 6, 8, 8, 48))
        warning = members.oversubscription(48, 12, 8, 8, 48)
        self.assertIn("96 TensorFlow threads", warning)
        self.assertIn("2.0 times as long", warning)
        self.assertIn("At most 6 members train at once", warning)
        # one process: the members after rank 0's own network, one after another
        warning = members.oversubscription(1, 3, 4, 4, 8)
        self.assertIn("2 member(s) are trained on rank 0", warning)
        self.assertIn("about 3 times as long", warning)
        self.assertNotIn("TensorFlow threads (", warning)
        # fewer ranks than the node has cores: the node's cores count
        self.assertIsNone(members.oversubscription(3, 3, 2, 2, 8))
        self.assertIn("1.5 times as long", members.oversubscription(3, 3, 4, 4, 8))

    def test_no_member_ranks_on_one_process(self):
        self.assertEqual(members.member_ranks(1, 4), [])
        self.assertEqual(members.member_ranks(48, 1), [])

    def test_seeds_differ_and_are_fixed(self):
        seeds = {members.member_seed(42, it, k) for it in range(5) for k in range(4)}
        self.assertEqual(len(seeds), 20)
        self.assertEqual(members.member_seed(42, 3, 1), members.member_seed(42, 3, 1))
        self.assertIsInstance(members.member_seed(None, 0, 0), int)


class Selection(unittest.TestCase):
    def test_newest_rows(self):
        self.assertEqual(members.newest_rows(10, 7).tolist(), [False] * 7 + [True] * 3)
        self.assertTrue(members.newest_rows(5, 0).all())
        self.assertFalse(members.newest_rows(5, 9).any())

    def test_selection_is_a_share_of_the_newest_and_reproducible(self):
        newest = members.newest_rows(1000, 400)
        a = members.selection_rows(newest, 0.1, np.random.default_rng(1))
        b = members.selection_rows(newest, 0.1, np.random.default_rng(1))
        self.assertEqual(a.sum(), 60)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(a[:400].any())
        self.assertEqual(members.selection_rows(np.zeros(5, bool), 0.1, np.random.default_rng(0)).sum(), 0)
        self.assertEqual(members.selection_rows(members.newest_rows(5, 4), 0.1, np.random.default_rng(0)).sum(), 1)

    def test_previous_rows_counts_the_previous_set_and_its_audit(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "training_data").mkdir()
            (run / "audit").mkdir()
            (run / "training_data" / "data_it_2.csv").write_text("a,loglkl\n" + "1,2\n" * 30)
            (run / "audit" / "audit_it_2.csv").write_text("a,loglkl\n" + "1,2\n" * 4)
            self.assertEqual(members.previous_rows(run, 3), 34)
            self.assertEqual(members.previous_rows(run, 0), 0)
            self.assertEqual(members.previous_rows(run, 5), 0)


class Choice(unittest.TestCase):
    def result(self, member, loss, error=None):
        return dict(member=member, selection_loss=loss, error=error)

    def test_warm_kept_within_the_margin(self):
        kept, reason = members.choose([self.result(0, 1.0), self.result(1, 0.85), self.result(2, 0.9)], True, 0.2)
        self.assertEqual(kept, 0)
        self.assertIn("warm", reason)

    def test_cold_member_kept_when_clearly_better(self):
        kept, _ = members.choose([self.result(0, 1.0), self.result(1, 0.2), self.result(2, 0.7)], True, 0.2)
        self.assertEqual(kept, 1)

    def test_without_a_warm_network_the_lowest(self):
        kept, _ = members.choose([self.result(0, 1.0), self.result(1, 0.95), self.result(2, 0.9)], False, 0.2)
        self.assertEqual(kept, 2)

    def test_failed_members_never_kept(self):
        kept, _ = members.choose([self.result(0, 1.0), self.result(1, float("nan")),
                                  self.result(2, 0.1, error="MemoryError: x")], True, 0.2)
        self.assertEqual(kept, 0)
        kept, _ = members.choose([self.result(0, float("nan"), error="x"), self.result(1, 0.5)], True, 0.2)
        self.assertEqual(kept, 1)
        with self.assertRaises(RuntimeError):
            members.choose([self.result(0, float("nan"), error="x")], True, 0.2)


class Pilot(unittest.TestCase):
    """The choice by the members' own posterior points (training.member_pilot)."""

    def test_depth_is_the_msre_scale(self):
        from training.losses import _chi2_quantile_wilson_hilferty

        for sigma, ndim in ((3, 16), (3, 24), (2, 5)):
            self.assertAlmostEqual(members.msre_scale(sigma, ndim),
                                   0.5 * float(_chi2_quantile_wilson_hilferty(sigma, ndim)), places=4)
        self.assertAlmostEqual(members.msre_scale(3, 16), 19.2, places=1)

    def test_a_false_plateau_and_a_missed_peak_both_count(self):
        truth = np.array([-100.0, -100.0, -1.0, -2.0, np.nan])
        best, depth = 0.0, 10.0
        honest = np.array([-90.0, -110.0, -1.5, -2.5, 0.0])   # wrong far below, right near the best
        plateau = np.array([-3.0, -100.0, -1.0, -2.0, 0.0])    # puts a point 97 too high near the best
        missed = np.array([-100.0, -100.0, -30.0, -2.0, 0.0])  # the truth is near the best, it is not
        scores = members.pilot_scores(dict(a=honest, b=plateau, c=missed), truth, best, depth)
        self.assertAlmostEqual(scores["a"], 0.5)
        self.assertAlmostEqual(scores["b"], np.sqrt(97.0 ** 2 / 3))
        self.assertAlmostEqual(scores["c"], np.sqrt(29.0 ** 2 / 2))
        self.assertEqual(members.pilot_scores(dict(a=np.full(2, -40.0)), np.array([-50.0, -60.0]), 0.0, 10.0),
                         dict(a=float("inf")))
        refused = np.array([-1.0, -30000.0])                   # a launcher's fixed value: not a true value
        self.assertAlmostEqual(members.pilot_scores(dict(a=np.array([-2.0, -1.0])), refused, 0.0, 10.0,
                                                    ignore=-30000.0)["a"], 1.0)

    def test_choice_keeps_warm_within_the_margin(self):
        kept, reason = members.choose_by_pilot({0: 2.0, 1: 1.7, 2: 3.0}, True, 0.2)
        self.assertEqual(kept, 0)
        self.assertIn("warm", reason)
        kept, _ = members.choose_by_pilot({0: 12.5, 1: 2.1, 2: 1.8}, True, 0.2)
        self.assertEqual(kept, 2)
        kept, _ = members.choose_by_pilot({0: 2.0, 1: 1.9, 2: float("inf")}, False, 0.2)
        self.assertEqual(kept, 1)
        kept, _ = members.choose_by_pilot({0: float("inf"), 1: 4.0}, True, 0.2)
        self.assertEqual(kept, 1)
        kept, reason = members.choose_by_pilot({0: 4.0}, True, 0.2)
        self.assertEqual(kept, 0)
        with self.assertRaises(RuntimeError):
            members.choose_by_pilot({0: float("inf"), 1: float("nan")}, True, 0.2)

    def test_starts_without_a_region_are_the_best_distinct_points(self):
        x = np.array([[0, 0], [1, 1], [1, 1], [2, 2], [3, 3]], np.float32)
        y = np.array([1.0, 5.0, 5.0, 3.0, -1.0])
        starts = members.pilot_starts(None, x, y, 3, np.random.default_rng(0))
        self.assertEqual(starts.tolist(), [[1, 1], [2, 2], [0, 0]])

    def test_points_are_read_back_for_a_continued_run(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "trained_models/members").mkdir(parents=True)
            frame = pd.DataFrame(dict(a=[0.1, 0.2, 0.3], b=[1.0, 2.0, 3.0], member=[0, 1, 1],
                                      loglkl=[-3.0, -4.0, np.nan]))
            frame["derived:H0"] = [67.0, 68.0, 69.0]
            frame["pred_member0"] = [-3.1, -4.2, -5.0]
            frame.to_csv(run / "trained_models/members/pilot_it_4.csv", index=False)
            inputs, targets, derived = members.load_pilot(run, 4, ["a", "b"], ["H0", "sigma8"])
            self.assertEqual(inputs.tolist(), [[0.1, 1.0], [0.2, 2.0]])
            self.assertEqual(targets.tolist(), [-3.0, -4.0])
            self.assertEqual(derived[:, 0].tolist(), [67.0, 68.0])
            self.assertTrue(np.isnan(derived[:, 1]).all())
            with self.assertRaises(ValueError):
                members.load_pilot(run, 4, ["a", "c"])
            (run / "training_data").mkdir()
            (run / "training_data" / "data_it_4.csv").write_text("a,loglkl\n" + "1,2\n" * 10)
            self.assertEqual(members.previous_rows(run, 5), 12)


class BoostPhaseOnly(unittest.TestCase):
    def test_members_train_only_networks_trained_on_boost_points(self):
        # boosts from network 5: network 5 is the tempered phase's last, 6 the first trained on boost points
        self.assertEqual([n for n in range(10) if members.trains_members(12, 5, n)], [6, 7, 8, 9])
        self.assertFalse(any(members.trains_members(12, None, n) for n in range(10)))   # a run without boosts
        self.assertFalse(any(members.trains_members(1, 5, n) for n in range(10)))      # the default

    def test_a_run_without_boosts_is_refused(self):
        import yaml

        from config.config import Config

        raw = yaml.safe_load((ROOT / "docs/site/examples/gaussian_client.yaml").read_text())
        raw["training"]["members"] = 3
        with self.assertRaises(ValueError):
            Config.from_dict(raw)
        raw["acquisition"]["boost"] = dict(n=24, start_iteration=1)
        self.assertEqual(Config.from_dict(raw).training.members, 3)


class Settings(unittest.TestCase):
    def test_defaults_are_one_training(self):
        cfg = TrainingConfig.from_dict(dict(TRAINING))
        self.assertEqual((cfg.members, cfg.selection_share, cfg.member_margin, cfg.member_threads), (1, 0.1, 0.2, None))
        self.assertEqual((cfg.member_pilot, cfg.member_pilot_steps, cfg.member_pilot_walkers, cfg.member_pilot_depth),
                         (0, 20000, 256, None))

    def test_pilot_values_and_errors(self):
        cfg = TrainingConfig.from_dict(dict(TRAINING, members=6, member_pilot=100, member_pilot_steps=500,
                                            member_pilot_walkers=64, member_pilot_depth=25))
        self.assertEqual((cfg.member_pilot, cfg.member_pilot_steps, cfg.member_pilot_walkers, cfg.member_pilot_depth),
                         (100, 500, 64, 25.0))
        for bad in (dict(members=1, member_pilot=10), dict(members=3, member_pilot=-1),
                    dict(members=3, member_pilot=10, member_pilot_steps=2),
                    dict(members=3, member_pilot=10, member_pilot_walkers=1),
                    dict(members=3, member_pilot=10, member_pilot_depth=0)):
            with self.assertRaises(ValueError):
                TrainingConfig.from_dict(dict(TRAINING, **bad))

    def test_values_and_errors(self):
        cfg = TrainingConfig.from_dict(dict(TRAINING, members=4, selection_share=0.2, member_margin=0.0,
                                            member_threads=8))
        self.assertEqual((cfg.members, cfg.selection_share, cfg.member_margin, cfg.member_threads), (4, 0.2, 0.0, 8))
        for bad in (dict(members=0), dict(selection_share=0), dict(selection_share=1), dict(member_margin=1),
                    dict(member_threads=0)):
            with self.assertRaises(ValueError):
                TrainingConfig.from_dict(dict(TRAINING, **bad))


class PhaseFiles(unittest.TestCase):
    def test_training_phase_lists_the_members(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            for name in ("trained_models/members/model_it_3_member1.keras", "trained_models/members/members_it_3.json",
                         "training_history/history_it_3_member1.csv", "trained_models/members/model_it_13_member1.keras",
                         "training_history/history_it_13_member0.csv", "trained_models/members/pilot_it_3.csv",
                         "trained_models/members/pilot_it_13.csv"):
                (run / name).parent.mkdir(parents=True, exist_ok=True)
                (run / name).write_text("x")
            found = sorted(str(p.relative_to(run)) for p in phase_outputs(run, "training_it_3"))
            self.assertEqual(found, ["trained_models/members/members_it_3.json",
                                     "trained_models/members/model_it_3_member1.keras",
                                     "trained_models/members/pilot_it_3.csv",
                                     "training_history/history_it_3_member1.csv"])


class OneMember(unittest.TestCase):
    """A member trained from random weights on a small quadratic, as a member rank does."""

    def test_trains_saves_and_scores(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, size=(400, 3)).astype(np.float32)
        y = (-0.5 * 20 * (x ** 2).sum(1)).astype(np.float32).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as d:
            task = dict(member=1, seed=7, threads=1, inputs=x[:360], targets=y[:360], selection_inputs=x[360:],
                        selection_targets=y[360:], background=None,
                        model=dict(n_layers=2, n_neurons=16, activation="alsing"),
                        loss=dict(name="msre", sigma_level=3, chi2_dof=3, ess_floor=0.1),
                        schedule=dict(learning_rate=1e-3, n_epochs=3, batch_size=32, validation_split=0.2,
                                      patience=5, lr_schedule="none", lr_factor=0.3, lr_patience=75, lr_min=1e-6,
                                      lr_grace=100),
                        model_path=str(Path(d) / "m.keras"), history_path=str(Path(d) / "h.csv"))
            result = members.train_on_master(task)
            self.assertIsNone(result.get("error"), result.get("error"))
            self.assertTrue(np.isfinite(result["selection_loss"]))
            self.assertTrue(Path(task["model_path"]).exists() and Path(task["history_path"]).exists())
            self.assertEqual(result["epochs"], 3)
            json.dumps(result)                      # the summary record must be writable


class MemberChain(unittest.TestCase):
    """A member's chain for the pilot (sample_member), without a region and in a region of two
    ellipsoids (the mixture sampler of the boosts)."""

    @classmethod
    def setUpClass(cls):
        from likelihood.surrogate import ParameterInfo, SurrogateMetadata

        cls.directory = tempfile.TemporaryDirectory()
        d = Path(cls.directory.name)
        rng = np.random.default_rng(0)
        x = rng.uniform(-6, 6, size=(600, 2)).astype(np.float32)
        y = np.logaddexp(-0.5 * (((x - [-3, 0]) / 0.6) ** 2).sum(1), -0.5 * (((x - [3, 0]) / 0.6) ** 2).sum(1))
        cls.x, cls.y = x, y.astype(np.float32).reshape(-1, 1)
        task = dict(member=1, seed=3, threads=1, inputs=x[:540], targets=cls.y[:540], selection_inputs=x[540:],
                    selection_targets=cls.y[540:], background=None,
                    model=dict(n_layers=2, n_neurons=16, activation="alsing"),
                    loss=dict(name="msre", sigma_level=3, chi2_dof=2, ess_floor=0.1),
                    schedule=dict(learning_rate=3e-3, n_epochs=5, batch_size=32, validation_split=0.2, patience=5,
                                  lr_schedule="none", lr_factor=0.3, lr_patience=75, lr_min=1e-6, lr_grace=100),
                    model_path=str(d / "m.keras"), history_path=str(d / "h.csv"))
        result = members.train_on_master(task)
        assert result.get("error") is None, result
        SurrogateMetadata(tuple(ParameterInfo(name=n, label=n, scale=1.0, lower=-6.0, upper=6.0, center=0.0,
                                              sigma=1.0) for n in ("a", "b"))).save(d / "metadata.json")
        cls.task = dict(member=1, model_path=task["model_path"], metadata_path=str(d / "metadata.json"), n=40,
                        steps=200, walkers=16, seed=11)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_without_a_region(self):
        starts = members.pilot_starts(None, self.x, self.y, 16, np.random.default_rng(0))
        out = members.sample_on_master(dict(self.task, starts=starts, region_path=None))
        self.assertIsNone(out.get("error"), out.get("error"))
        self.assertEqual(out["points"].shape, (40, 2))
        self.assertTrue(np.all(np.abs(out["points"]) <= 6))

    def test_in_a_region_of_two_ellipsoids(self):
        from sampling.region import Region

        rng = np.random.default_rng(1)
        region = Region.from_clouds([rng.normal(m, 0.5, (300, 2)) for m in ([-3, 0], [3, 0])], [-6, -6], [6, 6],
                                    1.5, best=0.0)
        path = Path(self.directory.name) / "region.npz"
        region.save(path)
        state = np.random.get_state()
        out = members.sample_on_master(dict(self.task, starts=None, region_path=str(path)))
        self.assertIsNone(out.get("error"), out.get("error"))
        self.assertEqual(out["points"].shape, (40, 2))
        self.assertTrue(region.inside(out["points"]).all())
        self.assertTrue(np.array_equal(np.random.get_state()[1], state[1]))   # the run's own draws are untouched


if __name__ == "__main__":
    unittest.main()
