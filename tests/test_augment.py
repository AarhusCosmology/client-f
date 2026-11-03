"""client.py augment: the density-deficit filter for points with known true values, the
profile-study reader, and the command with the file a continued run picks up."""

import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.config import TrainingConfig
from dataset.augment import (DensityLaw, load_profile_calls, main, mark_trained, pending_augmentations,
                             read_augmentation, select_known_points, training_log_density)
from dataset.dataset import TrainingDataset

NDIM, T, K = 4, 2.0, 20
NAMES = ["a", "b", "c", "d"]


def loglike(x):
    return -0.5 * np.sum(np.asarray(x, float) ** 2, axis=1)


def tempered(rng, n):
    """Samples of L^(1/T) for the unit Gaussian: N(0, T)."""
    return rng.normal(scale=np.sqrt(T), size=(n, NDIM))


def holed_dataset(rng, n=6000, edge=1.0):
    """A training set that follows L^(1/T) except that everything at a > edge is missing."""
    x = tempered(rng, n)
    x = x[x[:, 0] <= edge]
    return TrainingDataset(x, loglike(x), SimpleNamespace(param_names=NAMES, derived_names=[]),
                           n_neighbors=K, target_temperature=T)


class FilterTests(unittest.TestCase):
    def test_density_law_rises_with_the_likelihood(self):
        rng = np.random.default_rng(1)
        dataset = holed_dataset(rng, edge=np.inf)
        law = DensityLaw(dataset.targets[:, 0], training_log_density(dataset), T, NDIM)
        self.assertTrue(np.all(np.diff(law.values) >= 0))
        # In a set distributed as L^(1/T), log density = log L / T + const: the k-NN estimate
        # flattens the slope but keeps it well above zero.
        slope = (law.values[-1] - law.values[0]) / (law.centres[-1] - law.centres[0])
        self.assertGreater(slope, 0.25 / T)
        self.assertLess(slope, 1.5 / T)

    def test_points_go_where_the_training_set_is_short_and_not_elsewhere(self):
        rng = np.random.default_rng(2)
        dataset = holed_dataset(rng)
        x = tempered(rng, 4000)
        hole = x[:, 0] > 1.2
        covered = x[:, 0] < 0.5
        x = np.vstack([x[hole][:300], x[covered][:300]])
        in_hole = np.arange(len(x)) < 300
        chosen, info, _ = select_known_points(dataset, x, loglike(x), 200, np.random.default_rng(3))
        self.assertGreater(len(chosen), 100)
        self.assertGreater(np.mean(in_hole[chosen]), 0.9)
        self.assertEqual(info["selected"], len(chosen))
        self.assertEqual(len(set(chosen.tolist())), len(chosen))

    def test_nothing_is_added_to_a_set_already_at_its_target(self):
        rng = np.random.default_rng(4)
        dataset = holed_dataset(rng, edge=np.inf)
        x = tempered(rng, 300)
        chosen, info, _ = select_known_points(dataset, x, loglike(x), 300, np.random.default_rng(5))
        # k-NN noise leaves about half the candidates nominally short; few of them are kept.
        self.assertGreater(info["short_at_start"], 60)
        self.assertLess(len(chosen), 45)

    def test_a_tight_cluster_contributes_about_k_points(self):
        rng = np.random.default_rng(6)
        dataset = holed_dataset(rng)
        x = np.array([2.0, 0.0, 0.0, 0.0]) + 1e-3 * rng.normal(size=(400, NDIM))
        chosen, info, _ = select_known_points(dataset, x, loglike(x), 400, np.random.default_rng(7))
        self.assertGreaterEqual(len(chosen), K // 2)
        self.assertLessEqual(len(chosen), 2 * K)
        self.assertEqual(info["short_at_end"], 0)  # the cluster is saturated

    def test_a_sparse_training_set_reads_the_law_from_its_best_points(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(-30, 30, size=(400, NDIM))  # nearly nothing within the window of the best
        dataset = TrainingDataset(x, loglike(x), SimpleNamespace(param_names=NAMES, derived_names=[]),
                                  n_neighbors=K, target_temperature=T)
        law = DensityLaw(dataset.targets[:, 0], training_log_density(dataset), T, NDIM)
        self.assertTrue(law.extended)
        self.assertEqual(int(law.counts.sum()), 120)
        # A uniform set has the same density at every likelihood value, so its law is flat and
        # it is nowhere short of it: the fallback runs, and admits next to nothing.
        candidates = rng.normal(scale=0.5, size=(50, NDIM))
        chosen, info, _ = select_known_points(dataset, candidates, loglike(candidates), 30, np.random.default_rng(12))
        self.assertLess(len(chosen), 10)
        self.assertTrue(info["density_law"]["extended"])
        few = TrainingDataset(x[:100], loglike(x[:100]), SimpleNamespace(param_names=NAMES, derived_names=[]),
                              n_neighbors=K, target_temperature=T)
        with self.assertRaises(ValueError):
            DensityLaw(few.targets[:, 0], training_log_density(few), T, NDIM)

    def test_limit_copies_low_and_non_finite_candidates(self):
        rng = np.random.default_rng(8)
        dataset = holed_dataset(rng)
        x = tempered(rng, 3000)
        x = x[x[:, 0] > 1.2][:100]
        y = loglike(x)
        # A training point as a double that differs from the stored single below its precision.
        x = np.vstack([x, dataset.inputs[:1].astype(float) * (1 + 1e-9), [[2.0, 0.1, 0.1, 0.1]], [[2.0, 0.2, 0.2, 0.2]]])
        y = np.concatenate([y, [float(dataset.targets[0, 0])], [-1e4], [np.nan]])
        chosen, info, _ = select_known_points(dataset, x, y, 7, np.random.default_rng(9))
        self.assertEqual(len(chosen), 7)
        self.assertTrue(np.all(chosen < 100))
        self.assertEqual((info["copies_of_training_points"], info["not_finite"]), (1, 1))
        self.assertGreaterEqual(info["below_law"], 1)  # the -1e4 point, and any tail draws
        everything, _, _ = select_known_points(dataset, x, y, 500, np.random.default_rng(9))
        self.assertTrue(np.all(everything < 100))


    def test_points_injected_earlier_are_recognised_and_not_selected_again(self):
        rng = np.random.default_rng(13)
        dataset = holed_dataset(rng)
        x = tempered(rng, 4000)
        x = x[x[:, 0] > 1.2][:200]
        first, info, _ = select_known_points(dataset, x, loglike(x), 40, np.random.default_rng(14))
        self.assertEqual(info["copies_of_training_points"], 0)
        dataset.add_data(inputs=x[first], targets=loglike(x[first]))  # stored in single precision
        again, info, _ = select_known_points(dataset, x, loglike(x), 200, np.random.default_rng(15))
        self.assertEqual(info["copies_of_training_points"], 40)
        self.assertFalse(set(first.tolist()) & set(again.tolist()))
        twice = np.vstack([x, x])  # a candidate listed twice is taken at most once
        chosen, _, _ = select_known_points(holed_dataset(np.random.default_rng(13)), twice, loglike(twice), 400,
                                           np.random.default_rng(16))
        self.assertEqual(len(set((chosen % len(x)).tolist())), len(chosen))


class ScheduleTests(unittest.TestCase):
    def test_marker_is_committed_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            path = run / "augmentation/augment_it_3.csv"
            path.parent.mkdir()
            path.write_text("x,loglkl\n1,-1\n")
            with mock.patch("run.storage.os.replace", side_effect=OSError("interrupted commit")):
                with self.assertRaisesRegex(OSError, "interrupted commit"):
                    mark_trained([path], iteration=3, epochs=10, schedule_factor=3)
            self.assertEqual(pending_augmentations(run, np.array([[1.]])), [path])
            self.assertEqual(list(path.parent.iterdir()), [path])
            mark_trained([path], iteration=3, epochs=10, schedule_factor=3)
            self.assertEqual(pending_augmentations(run, np.array([[1.]])), [])

    def test_factor_defaults_to_three_and_must_be_at_least_one(self):
        root = Path(__file__).resolve().parents[1]
        training = yaml.safe_load((root / "input" / "_smoke.yaml").read_text())["training"]
        self.assertEqual(TrainingConfig.from_dict(training).augment_schedule_factor, 3.0)
        self.assertEqual(TrainingConfig.from_dict(dict(training, augment_schedule_factor=1)).augment_schedule_factor, 1.0)
        with self.assertRaises(ValueError):
            TrainingConfig.from_dict(dict(training, augment_schedule_factor=0.5))

    def test_an_injection_is_pending_until_its_first_training(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "augmentation").mkdir()
            inputs = rng.normal(size=(500, NDIM)).astype(np.float32)
            added = rng.normal(size=(20, NDIM)).astype(np.float32)
            path = run / "augmentation" / "augment_it_4.csv"
            frame = pd.DataFrame(added, columns=NAMES)
            frame["loglkl"] = loglike(added).astype(np.float32)
            frame.to_csv(path, index=False)
            self.assertEqual(pending_augmentations(run, inputs), [])  # written, but not in the training set
            both = np.vstack([inputs, pd.read_csv(path)[NAMES].to_numpy(np.float32)])
            self.assertEqual(pending_augmentations(run, both), [path])
            mark_trained([path], iteration=5, epochs=1234, schedule_factor=3.0)
            record = json.loads((run / "augmentation" / "augment_it_4.trained.json").read_text())
            self.assertEqual((record["iteration"], record["epochs"], record["schedule_factor"]), (5, 1234, 3.0))
            self.assertEqual(pending_augmentations(run, both), [])
            self.assertEqual(pending_augmentations(run / "nowhere", both), [])


def write_study(study, order, calls):
    (study / "calls").mkdir(parents=True)
    (study / "manifest.json").write_text(json.dumps(dict(
        objective="native backend objective, matching ordinary client training",
        bounds={n: [-10.0, 10.0] for n in order})))
    lines = []
    for number, (x, value) in enumerate(calls, start=1):
        lines.append(dict(event="start", call=number, stage="corrected/node/0", x=list(map(float, x))))
        lines.append(dict(event="return", call=number, value=value))
    lines.append(dict(event="cache_hit", x=list(map(float, calls[0][0])), value=calls[0][1]))
    lines.append(dict(event="start", call=len(calls) + 1, stage="corrected/node/0", x=[0.0] * len(order)))
    (study / "calls" / "attempt_0000.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")


class ReaderTests(unittest.TestCase):
    def test_profile_calls_come_back_in_the_runs_parameter_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            write_study(study, ["d", "c", "b", "a"], [([4, 3, 2, 1], -1.5), ([8, 7, 6, 5], None), ([9, 9, 9, 9], -2.5)])
            x, y, labels = load_profile_calls(study, NAMES)
            np.testing.assert_array_equal(x, [[1, 2, 3, 4], [9, 9, 9, 9]])
            np.testing.assert_array_equal(y, [-1.5, -2.5])
            self.assertIn("attempt_0000#1", labels[0])
            (study / "manifest.json").write_text(json.dumps(dict(objective="shifted", bounds={n: [0, 1] for n in NAMES})))
            with self.assertRaises(ValueError):
                load_profile_calls(study, NAMES)

    def test_augmentation_file_must_match_the_runs_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "augment_it_3.csv"
            pd.DataFrame([[1.0, 2.0, 3.0, 4.0, -7.0]], columns=NAMES + ["loglkl"]).to_csv(path, index=False)
            x, y = read_augmentation(path, NAMES)
            self.assertEqual((x.tolist(), y.tolist()), ([[1.0, 2.0, 3.0, 4.0]], [-7.0]))
            with self.assertRaises(ValueError):
                read_augmentation(path, NAMES[::-1])


class CommandTests(unittest.TestCase):
    def test_command_writes_the_file_a_continued_run_loads_and_changes_nothing_else(self):
        rng = np.random.default_rng(10)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "training_data").mkdir(parents=True)
            (run / "trained_models").mkdir()
            cfg = yaml.safe_load((root / "input" / "_smoke.yaml").read_text())
            cfg["acquisition"].update(n_neighbors=K, target_temperature=T)
            cfg["seed"] = 5
            (run / "run.yaml").write_text(yaml.safe_dump(cfg))
            (run / "metadata.json").write_text(json.dumps(dict(parameters=[
                dict(name=n, label=n, scale=1.0, lower=-10.0, upper=10.0, center=0.0, sigma=1.0) for n in NAMES])))
            dataset = holed_dataset(rng)
            for it in (0, 1):  # the later one is what a continued run loads
                dataset.save(run / "training_data" / f"data_it_{it}.csv")
            before = (run / "training_data" / "data_it_1.csv").read_bytes()
            x = tempered(rng, 4000)
            x = x[x[:, 0] > 1.2][:250]
            study = Path(tmp) / "study"
            write_study(study, NAMES, list(zip(x, loglike(x))) + [([20.0, 0, 0, 0], -200.0)])

            with redirect_stdout(io.StringIO()) as out:
                main([str(run), "--profiles", str(study), "-n", "50"])
            self.assertIn("selected 50", out.getvalue())
            self.assertIn("(1 outside)", out.getvalue())
            self.assertEqual((run / "training_data" / "data_it_1.csv").read_bytes(), before)
            self.assertEqual(sorted(p.name for p in (run / "training_data").iterdir()), ["data_it_0.csv", "data_it_1.csv"])
            added, values = read_augmentation(run / "augmentation" / "augment_it_1.csv", NAMES)
            self.assertEqual(added.shape, (50, NDIM))
            np.testing.assert_allclose(values, loglike(added), rtol=1e-5)
            record = json.loads((run / "augmentation" / "augment_it_1.json").read_text())
            self.assertEqual((record["iteration"], record["selected"], record["seed"], len(record["points"])), (1, 50, 6, 50))
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                main([str(run), "--profiles", str(study), "-n", "50"])
            with redirect_stdout(io.StringIO()):
                main([str(run), "--profiles", str(study), "-n", "50", "--force"])
            again, _ = read_augmentation(run / "augmentation" / "augment_it_1.csv", NAMES)
            np.testing.assert_array_equal(added, again)  # same seed, same selection
            with redirect_stdout(io.StringIO()):
                main([str(run), "--profiles", str(study), "-n", "10", "-i", "0", "--output", str(Path(tmp) / "trial")])
            self.assertTrue((Path(tmp) / "trial" / "augment_it_0.csv").exists())
            self.assertFalse((run / "augmentation" / "augment_it_0.csv").exists())


if __name__ == "__main__":
    unittest.main()
