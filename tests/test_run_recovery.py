"""A run continued after an interruption, and what its objective identity holds.

run/storage.py: a phase that did not reach its commit is set aside and done again.
run/identity.py: the record is of the objective, not of CLiENT's wrapper files or the launcher,
and a reviewed change can be accepted. No external likelihood; a Cobaya model of two lines.
"""

import importlib
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run.run import _start_iteration  # noqa: E402
from run.storage import PhaseStore, phase_outputs  # noqa: E402


def touch(root, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return [root / name for name in names]


def files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


class InterruptedPhases(unittest.TestCase):
    def test_an_interrupted_training_is_set_aside_and_done_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            kept = ["training_data/data_it_3.csv", "trained_models/model_it_2.keras", "audit/audit_it_13.csv",
                    "audit/audit_it_2.csv", "boost/region_it_2.npz", "boost/boost_it_2.csv",
                    "derived_data/derived_it_3.csv", "augmentation/augment_it_1.csv"]
            left = ["trained_models/model_it_3.keras", "training_history/history_it_3.csv", "boost/region_it_3.npz",
                    "audit/audit_it_3.csv", "audit/audit_it_3.json", "audit/audit_evaluations_it_3.jsonl",
                    "audit/audit_derived_it_3.csv", "derived_data/derived_model_it_3.keras",
                    "derived_data/derived_report_it_3.csv"]
            touch(run, *kept, *left)
            # The marker of an augmentation file names the training that used its points first.
            (run / "augmentation/augment_it_1.trained.json").write_text(json.dumps(dict(iteration=2)))
            (run / "augmentation/augment_it_3.csv").write_text("x")
            (run / "augmentation/augment_it_3.trained.json").write_text(json.dumps(dict(iteration=3)))
            phases = PhaseStore(run)
            phases.complete("acquisition_it_2", [run / "training_data/data_it_3.csv", run / "derived_data/derived_it_3.csv"])
            phases.begin("training_it_3")
            with self.assertRaisesRegex(RuntimeError, "Incomplete phase training_it_3"):
                phases.check()                                   # never taken as done
            self.assertEqual(_start_iteration(run), 3)           # and the saved network would be loaded as it is

            said = []
            self.assertEqual(phases.recover(log=said.append), ["training_it_3"])
            aside = next((run / "interrupted").iterdir())
            self.assertTrue(aside.name.startswith("training_it_3-"))
            self.assertEqual(files(aside), sorted(left + ["augmentation/augment_it_3.trained.json"]))
            self.assertEqual((aside / "trained_models/model_it_3.keras").read_text(), "trained_models/model_it_3.keras")
            for name in kept + ["augmentation/augment_it_1.trained.json", "augmentation/augment_it_3.csv"]:
                self.assertTrue((run / name).exists(), name)
            self.assertFalse((run / "phases/training_it_3.json").exists())
            record = json.loads(next((run / "phases/history").glob("training_it_3-*.json")).read_text())
            self.assertEqual((record["status"], record["phase"]), ("interrupted", "training_it_3"))
            self.assertEqual(sorted(record["set_aside"]), sorted(left + ["augmentation/augment_it_3.trained.json"]))
            self.assertIn("Phase training_it_3 did not finish", said[0])
            self.assertIn("10 file(s) it left are kept in interrupted/", said[0])
            phases.check()
            self.assertEqual(phases.recover(log=said.append), [])  # nothing more to do
            # Network 3 has no file now: the run starts at 3 and trains it.
            self.assertEqual(_start_iteration(run), 3)
            self.assertFalse((run / "trained_models/model_it_3.keras").exists())

    def test_an_interrupted_acquisition_is_set_aside_and_done_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            model, = touch(run, "trained_models/model_it_3.keras")
            touch(run, "training_data/data_it_3.csv", "boost/boost_it_2.csv", "boost/region_it_3.npz")
            left = ["training_data/data_it_4.csv", "derived_data/derived_it_4.csv", "boost/boost_it_3.csv",
                    "boost/boost_it_3.json", "boost/fallback_it_3.json"]
            touch(run, *left)
            phases = PhaseStore(run)
            phases.complete("training_it_3", [model, run / "boost/region_it_3.npz"])
            phases.begin("acquisition_it_3")
            self.assertEqual(_start_iteration(run), 4)           # the training set of 4 was written, but not committed
            self.assertEqual(phases.recover(log=lambda _: None), ["acquisition_it_3"])
            self.assertEqual(files(next((run / "interrupted").iterdir())), sorted(left))
            self.assertEqual(_start_iteration(run), 3)           # network 3 is kept; its acquisition is done again
            for name in ("trained_models/model_it_3.keras", "boost/region_it_3.npz", "boost/boost_it_2.csv"):
                self.assertTrue((run / name).exists(), name)
            phases.check()

    def test_a_phase_that_left_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            touch(run, "training_data/data_it_0.csv")
            phases = PhaseStore(run)
            phases.begin("training_it_0")                        # stopped before the network was saved
            said = []
            self.assertEqual(phases.recover(log=said.append), ["training_it_0"])
            self.assertFalse((run / "interrupted").exists())
            self.assertIn("It left no file", said[0])
            phases.check()

    def test_what_cannot_be_done_again_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            phases = PhaseStore(tmp)
            phases.begin("initial")
            with self.assertRaisesRegex(RuntimeError, "initial design of this run did not finish.*start a new run"):
                phases.recover()
        with tempfile.TemporaryDirectory() as tmp:
            phases = PhaseStore(tmp)
            phases.begin("audit0")                               # a kind this code does not know
            with self.assertRaisesRegex(RuntimeError, "Incomplete phase audit0"):
                phases.recover()
            self.assertTrue((Path(tmp) / "phases/audit0.json").exists())

    def test_the_outputs_of_a_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            touch(run, "audit/audit_it_1.csv", "audit/audit_it_11.csv", "trained_models/model_it_1.keras",
                  "trained_models/model_it_1_pa2000.keras", "training_data/data_it_1.csv", "training_data/data_it_2.csv")
            names = lambda phase: sorted(str(p.relative_to(run)) for p in phase_outputs(run, phase))   # noqa: E731
            self.assertEqual(names("training_it_1"), ["audit/audit_it_1.csv", "trained_models/model_it_1.keras"])
            self.assertEqual(names("acquisition_it_1"), ["training_data/data_it_2.csv"])
            self.assertEqual(names("initial"), [])
            self.assertEqual(names("training_it_x"), [])

    def test_torn_augmentation_marker_is_archived_and_retried(self):
        import numpy as np
        from dataset.augment import pending_augmentations

        for content in ('{"iteration": 3, "epochs":', '[]', '{}', '{"iteration": "3"}'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                (run / "augmentation").mkdir()
                points = run / "augmentation/augment_it_3.csv"
                points.write_text("x,loglkl\n1,-1\n")
                marker = run / "augmentation/augment_it_3.trained.json"
                marker.write_text(content)
                # A malformed marker is never proof that the first training finished.
                self.assertEqual(pending_augmentations(run, np.array([[1.]])), [points])
                phases = PhaseStore(run)
                phases.begin("training_it_3")
                phases.recover(log=lambda _: None)
                phases.check()
                self.assertFalse(marker.exists())
                saved = next((run / "interrupted").glob("*/augmentation/*.trained.json"))
                self.assertEqual(saved.read_text(), content)
                self.assertEqual(pending_augmentations(run, np.array([[1.]])), [points])


LIKELIHOOD = """likelihood:
  toy:
    external: "{external}"
params:
  x:
    prior: {{min: -5, max: 5}}
    ref: {{dist: norm, loc: 0, scale: 1}}
    proposal: 0.2
  y:
    prior: {{min: -5, max: 5}}
    ref: {{dist: norm, loc: 0, scale: 1}}
    proposal: 0.2
"""


class Rewind(unittest.TestCase):
    """client.py RUN -s N in a run that has gone further: refused, or with --rewind what
    belongs to the later iterations is set aside first (run.storage.set_later_aside)."""

    def run_at_4(self, root):
        """A run that has trained networks 0 to 4, with boosts from 3, analysed at 2 and 4."""
        run = Path(root) / "run"
        phases = PhaseStore.__new__(PhaseStore)
        for k in range(5):
            touch(run, f"training_data/data_it_{k}.csv", f"trained_models/model_it_{k}.keras",
                  f"training_history/history_it_{k}.csv", f"audit/audit_it_{k}.csv", f"audit/audit_it_{k}.json",
                  f"convergence_stats/chain_summary_it_{k}.npz")
        touch(run, "boost/boost_it_3.csv", "boost/boost_it_3.json", "boost/region_it_4.npz", "boost/fallback_it_1.json",
              "metadata.json", "client.yaml", "metrics.json", "metrics.md", "likelihood_input/model_it_9.yaml",
              "objective_calls/rank0-a.jsonl", "interrupted/training_it_4-x/trained_models/model_it_4.keras",
              "analysis/posterior_it_2.npz", "analysis/posterior_it_4.npz", "analysis/true_check_it_4.npz",
              "analysis/report.json", "analysis/true_check.json", "evidence/evidence_it_2_nlive2048.json",
              "evidence/evidence_it_4_nlive2048.json", "evidence/evidence_it_4_nlive2048.txt", "modes/modes_it_3.json",
              "posterior_chains/surrogate_it_4.txt", "augmentation/augment_it_1.csv", "augmentation/augment_it_2.csv",
              "augmentation/augment_it_4.csv")
        (run / "augmentation/augment_it_1.trained.json").write_text(json.dumps(dict(iteration=2)))
        (run / "augmentation/augment_it_2.trained.json").write_text(json.dumps(dict(iteration=3)))
        (run / "boost/start.json").write_text(json.dumps(dict(start_iteration=3)))
        phases = PhaseStore(run)
        phases.complete("initial", [run / "training_data/data_it_0.csv"])
        for k in range(5):
            phases.complete(f"training_it_{k}", [run / f"trained_models/model_it_{k}.keras", run / f"audit/audit_it_{k}.csv"]
                            + ([run / "boost/region_it_4.npz"] if k == 4 else []))
            if k < 4:
                phases.complete(f"acquisition_it_{k}", [run / f"training_data/data_it_{k + 1}.csv"]
                                + ([run / "boost/boost_it_3.json"] if k == 3 else []))
        phases.begin("training_it_2")                 # a retraining once: its earlier record is in phases/history/
        phases.complete("training_it_2", [run / "trained_models/model_it_2.keras", run / "audit/audit_it_2.csv"])
        return run, phases

    def test_what_is_later_than_a_network_is_set_aside_and_nothing_is_deleted(self):
        from run.run import latest_iteration
        from run.storage import later_files, set_later_aside

        with tempfile.TemporaryDirectory() as tmp:
            run, phases = self.run_at_4(tmp)
            before = {name: (run / name).read_bytes() for name in files(run)}
            self.assertEqual(latest_iteration(run), 4)
            said = []
            aside = set_later_aside(run, 2, log=said.append)
            self.assertEqual(aside.parent, run / "interrupted")
            self.assertTrue(aside.name.startswith("rewind_to_it_2-"))
            # what a run stopped at network 2 holds, and the things a rewind never touches
            kept = sorted(
                ["metadata.json", "client.yaml", "metrics.json", "metrics.md", "likelihood_input/model_it_9.yaml",
                 "objective_calls/rank0-a.jsonl", "interrupted/training_it_4-x/trained_models/model_it_4.keras",
                 "analysis/posterior_it_2.npz", "evidence/evidence_it_2_nlive2048.json", "boost/fallback_it_1.json",
                 "augmentation/augment_it_1.csv", "augmentation/augment_it_1.trained.json", "augmentation/augment_it_2.csv",
                 "phases/initial.json", "phases/acquisition_it_0.json", "phases/acquisition_it_1.json"]
                + [f"phases/training_it_{k}.json" for k in range(3)]
                + [f"{folder}_it_{k}.{suffix}" for k in range(3) for folder, suffix in (
                    ("training_data/data", "csv"), ("trained_models/model", "keras"), ("training_history/history", "csv"),
                    ("audit/audit", "csv"), ("audit/audit", "json"), ("convergence_stats/chain_summary", "npz"))])
            here = [name for name in files(run) if not name.startswith(str(aside.relative_to(run))) and not name.startswith("phases/history/")]
            self.assertEqual(sorted(here), kept)
            record = json.loads((aside / "REWIND.json").read_text())
            self.assertEqual((record["to_iteration"], record["networks_set_aside"]), (2, [3, 4]))
            # every file is where it was or under the same path in the folder; the metrics are in both
            for name, content in before.items():
                places = [path for path in (run / name, aside / name) if path.exists()]
                self.assertTrue(places, name)
                self.assertTrue(all(path.read_bytes() == content for path in places), name)
                self.assertEqual(len(places), 2 if name in ("metrics.json", "metrics.md") else 1, name)
            for gone in ("phases/acquisition_it_2.json", "phases/training_it_3.json", "training_data/data_it_3.csv",
                         "boost/start.json", "boost/boost_it_3.json", "augmentation/augment_it_2.trained.json",
                         "analysis/report.json", "analysis/true_check.json", "modes/modes_it_3.json"):
                self.assertTrue((aside / gone).exists(), gone)
            phases.check()                                          # what is left is committed and whole
            self.assertEqual(latest_iteration(run), 2)
            self.assertIn("networks 3 to 4", said[0])
            self.assertIn("nothing is deleted", said[0])
            self.assertIsNone(set_later_aside(run, 2))              # a second time there is nothing later
            self.assertEqual([p for p, copy in later_files(run, 2) if not copy], [])

    def test_a_hand_over_at_or_before_the_network_and_a_boost_before_it_stay(self):
        from run.storage import set_later_aside

        with tempfile.TemporaryDirectory() as tmp:
            run, phases = self.run_at_4(tmp)
            aside = set_later_aside(run, 3, log=lambda text: None)
            self.assertTrue((run / "boost/start.json").exists())           # the boosts began at 3
            self.assertTrue((run / "augmentation/augment_it_2.trained.json").exists())    # network 3 was trained on them
            self.assertTrue((aside / "boost/boost_it_3.json").exists())    # the acquisition of 3 is done again
            self.assertTrue((aside / "phases/acquisition_it_3.json").exists())
            self.assertTrue((run / "phases/training_it_3.json").exists())
            phases.check()

    def test_every_interrupted_rewind_move_is_recovered_without_losing_files(self):
        import shutil
        from run.storage import later_files, recover_rewind, set_later_aside

        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.run_at_4(tmp)
            moves = sum(not copy for _, copy in later_files(run, 2))
        # Failure before every move, and after the last move but before the completion
        # record. Covers lost phase records with old models/data still in the run.
        for stop_after in range(moves + 1):
            with self.subTest(stop_after=stop_after), tempfile.TemporaryDirectory() as tmp:
                run, phases = self.run_at_4(tmp)
                before = {name: (run / name).read_bytes() for name in files(run)}
                original_move, count = shutil.move, 0

                def interrupted(source, target):
                    nonlocal count
                    if count == stop_after:
                        raise OSError('interrupted rewind')
                    original_move(source, target)
                    count += 1
                    if count == moves == stop_after:
                        raise OSError('interrupted rewind')

                with mock.patch('run.storage.shutil.move', side_effect=interrupted):
                    with self.assertRaisesRegex(OSError, 'interrupted rewind'):
                        set_later_aside(run, 2, log=lambda text: None)
                (record_path,) = run.glob('interrupted/rewind_to_it_*/REWIND.json')
                self.assertEqual(json.loads(record_path.read_text())['status'], 'pending')
                self.assertEqual(recover_rewind(run, log=lambda text: None), [2])
                self.assertEqual(json.loads(record_path.read_text())['status'], 'complete')
                self.assertEqual(_start_iteration(run), 2)
                phases.check()
                for name, content in before.items():
                    places = [p for p in (run / name, record_path.parent / name) if p.exists()]
                    self.assertTrue(places, name)
                    self.assertTrue(all(p.read_bytes() == content for p in places), name)
                self.assertEqual(recover_rewind(run), [])

    def test_rewind_refuses_a_missing_checkpoint_before_moving_anything(self):
        from run.storage import set_later_aside

        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.run_at_4(tmp)
            before = files(run)
            for iteration in (-1, 7):
                with self.assertRaisesRegex(ValueError, 'Cannot rewind'):
                    set_later_aside(run, iteration)
                self.assertEqual(files(run), before)

    def test_interrupted_rewind_refuses_changed_artifacts(self):
        from run.storage import recover_rewind, set_later_aside

        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.run_at_4(tmp)
            with mock.patch('run.storage.shutil.move', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    set_later_aside(run, 2)
            (run / 'trained_models/model_it_4.keras').write_text('changed')
            before = files(run)
            with self.assertRaisesRegex(RuntimeError, r'trained_models/model_it_4\.keras has changed.*first move every remaining copy'):
                recover_rewind(run)
            self.assertEqual(files(run), before)
            # Omitting a changed checkpoint must not leave it active after the rewind.
            (record_path,) = run.glob('interrupted/rewind_to_it_*/REWIND.json')
            record = json.loads(record_path.read_text())
            record['files'] = [e for e in record['files'] if e['file'] != 'trained_models/model_it_4.keras']
            record_path.write_text(json.dumps(record))
            before = files(run)
            with self.assertRaisesRegex(RuntimeError, 'model_it_4.keras is not in the pending plan'):
                recover_rewind(run)
            self.assertEqual(files(run), before)
            quarantine = run / 'interrupted/manual/model_it_4.keras'
            quarantine.parent.mkdir(parents=True)
            (run / 'trained_models/model_it_4.keras').rename(quarantine)
            self.assertEqual(recover_rewind(run, log=lambda text: None), [2])
            self.assertEqual(quarantine.read_text(), 'changed')
            self.assertEqual(_start_iteration(run), 2)

    def interrupted_after(self, tmp, moves):
        """A run whose rewind to network 2 was cut after `moves` of its moves."""
        import shutil
        from run.storage import set_later_aside

        run, phases = self.run_at_4(tmp)
        original, count = shutil.move, [0]

        def interrupted(source, target):
            if count[0] == moves:
                raise OSError('interrupted rewind')
            original(source, target)
            count[0] += 1

        with mock.patch('run.storage.shutil.move', side_effect=interrupted), self.assertRaises(OSError):
            set_later_aside(run, 2, log=lambda text: None)
        (record,) = run.glob('interrupted/rewind_to_it_*/REWIND.json')
        return run, phases, record

    def test_a_pending_rewind_gets_past_what_the_analysis_commands_rewrote_or_removed(self):
        # The report is made again, or a stale summary removed, between the kill and the next
        # start: neither file can be put back as the plan has it, and refusing them was a
        # start that could never be got past.
        from run.storage import recover_rewind

        with tempfile.TemporaryDirectory() as tmp:
            run, phases, record = self.interrupted_after(tmp, 1)              # analysis/posterior_it_4.npz is set aside
            aside = record.parent
            self.assertTrue((aside / 'analysis/posterior_it_4.npz').exists() and (run / 'analysis/report.json').exists())
            (run / 'analysis/report.json').write_text('the report, made again')             # planned, not yet moved
            (run / 'analysis/true_check.json').unlink()                                      # planned, removed by hand
            touch(run, 'analysis/posterior_it_4.npz')                                        # moved already, written again
            (run / 'analysis/posterior_it_4.npz').write_text('sampled again')
            said = []
            self.assertEqual(recover_rewind(run, log=said.append), [2])
            self.assertEqual((aside / 'analysis/report.json').read_text(), 'the report, made again')
            self.assertFalse((run / 'analysis/report.json').exists() or (run / 'analysis/posterior_it_4.npz').exists())
            self.assertEqual((aside / 'analysis/posterior_it_4.npz').read_text(), 'analysis/posterior_it_4.npz')   # the first one
            (later,) = aside.glob('analysis/posterior_it_4.npz.written-after-*')
            self.assertEqual(later.read_text(), 'sampled again')
            done = json.loads(record.read_text())
            self.assertEqual(done['status'], 'complete')
            self.assertEqual(len(done['not_as_planned']), 3)
            for words in ('report.json was written again since the plan', 'true_check.json was removed since the plan',
                          'posterior_it_4.npz was written again after it had been set aside'):
                self.assertIn(words, said[0])
            self.assertEqual(_start_iteration(run), 2)
            phases.check()
            self.assertEqual(recover_rewind(run), [])

    def test_regenerated_analysis_survives_a_second_interruption(self):
        import shutil
        from run.storage import recover_rewind, digest

        # Both sides of the regenerated report's move, including a copy without removal.
        for boundary in ('before', 'after', 'copied'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                run, phases, record = self.interrupted_after(tmp, 1)
                source = run / 'analysis/report.json'
                source.write_text('regenerated report')
                original_move = shutil.move

                def interrupted(src, dst):
                    if Path(src) != source:
                        return original_move(src, dst)
                    if boundary == 'after':
                        original_move(src, dst)
                    elif boundary == 'copied':
                        shutil.copy2(src, dst)
                    raise OSError('second interruption')

                with mock.patch('run.storage.shutil.move', side_effect=interrupted):
                    with self.assertRaisesRegex(OSError, 'second interruption'):
                        recover_rewind(run, log=lambda _: None)
                pending = json.loads(record.read_text())
                entry = next(e for e in pending['files'] if e['file'] == 'analysis/report.json')
                self.assertNotEqual(entry['sha256'], entry['archived_sha256'])
                self.assertTrue(pending['not_as_planned'])
                self.assertEqual(recover_rewind(run, log=lambda _: None), [2])
                self.assertEqual((record.parent / 'analysis/report.json').read_text(), 'regenerated report')
                self.assertEqual(digest(record.parent / 'analysis/report.json'), entry['archived_sha256'])
                self.assertFalse(source.exists())                 # also after a copy without its removal
                self.assertEqual(len(list(record.parent.glob('analysis/report.json*'))), 1)
                self.assertEqual(_start_iteration(run), 2)
                phases.check()
                self.assertEqual(recover_rewind(run), [])

    def test_new_analysis_joins_pending_plan_but_new_checkpoint_is_refused(self):
        from run.storage import recover_rewind
        with tempfile.TemporaryDirectory() as tmp:
            run, phases, record = self.interrupted_after(tmp, 1)
            touch(run, 'analysis/new_report.json', 'trained_models/model_it_5.keras')
            before = files(run)
            with self.assertRaisesRegex(RuntimeError, 'model_it_5.keras is not in the pending plan'):
                recover_rewind(run)
            self.assertEqual(files(run), before)
            quarantine = run / 'interrupted/manual'
            quarantine.mkdir()
            (run / 'trained_models/model_it_5.keras').rename(quarantine / 'model_it_5.keras')
            self.assertEqual(recover_rewind(run, log=lambda _: None), [2])
            self.assertTrue((record.parent / 'analysis/new_report.json').exists())
            self.assertEqual(_start_iteration(run), 2)
            phases.check()

    def test_what_a_command_without_the_lock_wrote_since_the_plan_is_set_aside_with_the_rest(self):
        # `client.py augment` holds no run lock either: an augmentation of a later training
        # set, made between the kill and the next start, stopped that start as a checkpoint
        # would. The marker of a first training is the loop's, and a derived emulator is
        # committed by a training phase: those stay refused.
        from run.storage import recover_rewind

        with tempfile.TemporaryDirectory() as tmp:
            run, phases, record = self.interrupted_after(tmp, 1)
            touch(run, 'augmentation/augment_it_3.csv', 'augmentation/augment_it_3.json')
            self.assertEqual(recover_rewind(run, log=lambda _: None), [2])
            for name in ('augmentation/augment_it_3.csv', 'augmentation/augment_it_3.json'):
                self.assertEqual((record.parent / name).read_text(), name)
                self.assertFalse((run / name).exists())
            done = json.loads(record.read_text())
            self.assertIn('augmentation/augment_it_3.csv', [e['file'] for e in done['files']])
            self.assertNotIn('not_as_planned', done)              # nothing of the plan was found otherwise
            self.assertEqual(_start_iteration(run), 2)
            phases.check()
        for name, content in (('augmentation/augment_it_3.trained.json', json.dumps(dict(iteration=4))),
                              ('derived_data/derived_model_it_4.keras', 'an emulator')):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                run, phases, record = self.interrupted_after(tmp, 1)
                (run / name).parent.mkdir(parents=True, exist_ok=True)
                (run / name).write_text(content)
                before = files(run)
                with self.assertRaisesRegex(RuntimeError, 'is not in the pending plan.*a file of a later iteration must not'):
                    recover_rewind(run)
                self.assertEqual(files(run), before)

    def test_a_report_recorded_as_written_again_and_then_removed_is_not_said_to_be_set_aside(self):
        # The second start records the report made again, and is cut before its move; the
        # report is then removed. The record said both "set aside as it is now" and "removed",
        # with the hash of a file that is in neither place.
        import shutil
        from run.storage import recover_rewind

        with tempfile.TemporaryDirectory() as tmp:
            run, phases, record = self.interrupted_after(tmp, 1)
            source = run / 'analysis/report.json'
            source.write_text('the report, made again')
            original = shutil.move

            def cut(src, dst):
                if Path(src) == source:
                    raise OSError('second interruption')
                return original(src, dst)

            with mock.patch('run.storage.shutil.move', side_effect=cut), self.assertRaises(OSError):
                recover_rewind(run, log=lambda _: None)
            entry = next(e for e in json.loads(record.read_text())['files'] if e['file'] == 'analysis/report.json')
            self.assertIn('archived_sha256', entry)
            source.unlink()
            said = []
            self.assertEqual(recover_rewind(run, log=said.append), [2])
            done = json.loads(record.read_text())
            entry = next(e for e in done['files'] if e['file'] == 'analysis/report.json')
            self.assertNotIn('archived_sha256', entry)
            self.assertEqual(done['not_as_planned'], ['analysis/report.json was removed since the plan; nothing to set aside'])
            self.assertNotIn('set aside as it is now', said[0])
            self.assertFalse((record.parent / 'analysis/report.json').exists())
            phases.check()

    def test_the_message_names_the_hash_the_copy_is_compared_with(self):
        # A report made again is expected in the folder as it was recorded before its move, not
        # as the plan first had it: the message named the plan's hash, which no copy could match.
        from run.storage import digest, recover_rewind

        with tempfile.TemporaryDirectory() as tmp:
            run, phases, record = self.interrupted_after(tmp, 1)
            (run / 'analysis/report.json').write_text('the report, made again')
            again = digest(run / 'analysis/report.json')
            self.assertEqual(recover_rewind(run, log=lambda _: None), [2])
            pending = json.loads(record.read_text())
            first = next(e for e in pending['files'] if e['file'] == 'analysis/report.json')['sha256']
            record.write_text(json.dumps(dict(pending, status='pending')))           # as a start cut after that move
            (record.parent / 'analysis/report.json').write_text('torn')
            with self.assertRaises(RuntimeError) as stopped:
                recover_rewind(run)
            self.assertIn(f'(sha256 {again[:12]}...)', str(stopped.exception))
            self.assertNotIn(first[:12], str(stopped.exception))

    def test_a_move_cut_between_its_copy_and_its_removal_is_finished(self):
        # What a move between two file systems leaves when it is killed: the file in both
        # places, the same. The copy in the folder is whole; the one in the run goes.
        import shutil
        from run.storage import recover_rewind

        with tempfile.TemporaryDirectory() as tmp:
            run, phases, record = self.interrupted_after(tmp, 3)
            aside = record.parent
            moved = next(e['file'] for e in json.loads(record.read_text())['files'] if (aside / e['file']).exists() and not e['kept_in_place'])
            shutil.copy2(aside / moved, run / moved)
            self.assertEqual(recover_rewind(run, log=lambda text: None), [2])
            self.assertFalse((run / moved).exists())
            self.assertTrue((aside / moved).exists())
            phases.check()
            # the same for a committed file that differs from its copy: that one stops the start
            from run.storage import later_files

            probe, _ = self.run_at_4(tmp + '/probe')
            planned = [str(path.relative_to(probe)) for path, copy in later_files(probe, 2) if not copy]
            run, phases, record = self.interrupted_after(tmp + '/again', planned.index('phases/training_it_3.json') + 1)
            aside = record.parent
            self.assertTrue((aside / 'phases/training_it_3.json').exists())
            touch(run, 'phases/training_it_3.json')
            with self.assertRaisesRegex(RuntimeError, r'phases/training_it_3\.json is in the run again'):
                recover_rewind(run)

    def test_the_command_refuses_without_the_option_and_names_it(self):
        import client

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self.run_at_4(tmp)
            before = files(run_dir)
            run = SimpleNamespace(run_dir=run_dir, is_new=False)
            with self.assertRaisesRegex(RuntimeError, r"gone on to iteration 4.*--rewind.*nothing is\s+deleted"):
                client.rewind_to_start(run, SimpleNamespace(start=2, rewind=False))
            self.assertEqual(files(run_dir), before)
            client.rewind_to_start(run, SimpleNamespace(start=4, rewind=False))        # the latest: as before
            client.rewind_to_start(run, SimpleNamespace(start=None, rewind=False))     # no -s: as before
            self.assertEqual(files(run_dir), before)
            with self.assertRaisesRegex(RuntimeError, "--rewind goes with -s"):
                client.rewind_to_start(run, SimpleNamespace(start=None, rewind=True))
            with self.assertRaisesRegex(RuntimeError, "--rewind goes with -s"):
                client.rewind_to_start(SimpleNamespace(run_dir=run_dir, is_new=True), SimpleNamespace(start=2, rewind=True))
            with mock.patch.object(client, "print_master") as said:
                client.rewind_to_start(run, SimpleNamespace(start=4, rewind=True))      # nothing later
                self.assertEqual(files(run_dir), before)
                client.rewind_to_start(run, SimpleNamespace(start=2, rewind=True))
            self.assertIn("nothing is set aside", said.call_args_list[0].args[0])
            self.assertTrue(any(name.startswith("interrupted/rewind_to_it_2-") for name in files(run_dir)))


class Identity(unittest.TestCase):
    def identity(self, directory, external="lambda x, y: -0.5 * (x*x + y*y)", launcher=None, extra=None):
        from likelihood.cobaya import CobayaLikelihood
        from run import identity as module

        path = Path(directory) / "toy.yaml"
        path.write_text(LIKELIHOOD.format(external=external))
        truth = CobayaLikelihood(str(path))
        config = SimpleNamespace(likelihood=SimpleNamespace(input=str(path), wrapper="cobaya", identity_files=extra))
        try:
            with mock.patch.object(module, "_launcher_directory", lambda: launcher):
                return module.identity(truth, config)
        finally:
            truth.cobaya_model.close()

    def test_what_is_recorded_and_what_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.identity(tmp)
            keys = list(first["files"])
            self.assertIn("input", keys)
            self.assertFalse([k for k in keys if k.startswith("wrapper:")])          # no text of CLiENT's wrappers
            self.assertFalse([k for k in keys if "__main__" in k or "__mp_main__" in k])
            self.assertEqual((first["schema"], first["wrapper"], first["convention"]), (2, "cobaya", 1))
            self.assertEqual([p["name"] for p in first["parameters"]], ["x", "y"])
            # the Cobaya component's own class file is there
            self.assertIn("likelihood:toy", keys)
            self.assertEqual(self.identity(tmp), first)                              # the same again

    def test_a_function_of_the_input_is_recorded_wherever_the_launcher_lies(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher_dir = Path(tmp).resolve() / "launcher"                     # as _launcher_directory gives it
            launcher_dir.mkdir()
            (launcher_dir / "review_toy_objective.py").write_text("def loglike(x, y):\n    return -0.5 * (x * x + y * y)\n")
            (launcher_dir / "review_toy_helper.py").write_text("BOX = 1\n")
            sys.path.insert(0, str(launcher_dir))
            try:
                importlib.import_module("review_toy_helper")
                external = "import_module('review_toy_objective').loglike"
                beside = self.identity(tmp, external=external, launcher=launcher_dir)        # the launcher is in the same folder
                elsewhere = self.identity(tmp, external=external, launcher=None)
                # a launcher's helper modules are no part of the objective ...
                self.assertNotIn("module:review_toy_helper", beside["files"])
                self.assertIn("module:review_toy_objective", beside["files"])
                # ... the function the input names is, by its own file
                self.assertIn("external:toy", beside["files"])
                self.assertEqual(beside["files"]["external:toy"], elsewhere["files"]["external:toy"])
                # without a launcher beside them, local modules are recorded as before
                self.assertIn("module:review_toy_helper", elsewhere["files"])
                self.assertIn("module:review_toy_objective", elsewhere["files"])
                (launcher_dir / "review_toy_objective.py").write_text("def loglike(x, y):\n    return -0.5 * (x * x + 2 * y * y)\n")
                changed = self.identity(tmp, external=external, launcher=launcher_dir)
                self.assertNotEqual(changed["files"]["external:toy"], beside["files"]["external:toy"])
                (launcher_dir / "review_toy_helper.py").write_text("BOX = 1  # a comment\n")
                self.assertEqual(self.identity(tmp, external=external, launcher=launcher_dir)["files"], changed["files"])
            finally:
                sys.path.remove(str(launcher_dir))
                for name in ("review_toy_helper", "review_toy_objective"):
                    sys.modules.pop(name, None)

    def test_the_wrapper_s_convention_is_part_of_it(self):
        from likelihood.base import BaseLikelihood

        with tempfile.TemporaryDirectory() as tmp:
            first = self.identity(tmp)
            with mock.patch.object(BaseLikelihood, "objective_convention", 2):
                self.assertEqual(self.identity(tmp)["convention"], 2)
            self.assertEqual(first["convention"], 1)

    def record(self, **files):
        return dict(schema=2, wrapper="cobaya", convention=1, objective="native",
                    parameters=[dict(name="x", lower=0.0, upper=1.0, scale=1.0)], files=dict(input="aaa", **files))

    def test_a_difference_is_named_and_stops_the_run(self):
        from run.identity import verify

        with tempfile.TemporaryDirectory() as tmp:
            first = self.record(**{"module:camb": "111", "module:gone": "222"})
            verify(tmp, first, new=True)
            verify(tmp, first)
            later = self.record(**{"module:camb": "999", "library:camb": "333"})
            with self.assertRaises(ValueError) as stopped:
                verify(tmp, later)
            message = str(stopped.exception)
            for part in ("changed: module:camb", "no longer loaded: module:gone", "new: library:camb", "--accept-identity-change"):
                self.assertIn(part, message)
            with self.assertRaisesRegex(ValueError, "parameters: bounds, scale or prior of x"):
                verify(tmp, dict(first, parameters=[dict(name="x", lower=0.0, upper=2.0, scale=1.0)]))
            with self.assertRaisesRegex(ValueError, "convention: 1 -> 2"):
                verify(tmp, dict(first, convention=2))
            self.assertEqual(json.loads((Path(tmp) / "objective_identity.json").read_text())["identity"], first)

    def test_a_reviewed_change_can_be_accepted_once(self):
        from run.identity import verify

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            first, later = self.record(**{"module:camb": "111"}), self.record(**{"module:camb": "999"})
            verify(run, first, new=True)
            with self.assertWarns(RuntimeWarning):
                verify(run, later, accept_change=True)
            saved = json.loads((run / "objective_identity.json").read_text())
            self.assertEqual(saved["identity"], later)
            self.assertEqual(saved["accepted_changes"][0]["differences"], ["changed: module:camb"])
            old = json.loads(next((run / "objective_identity_history").glob("*.json")).read_text())
            self.assertEqual((old["identity"], old["differences"]), (first, ["changed: module:camb"]))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                verify(run, later)                                # the next start needs no flag
                verify(run, later, accept_change=True)            # and the flag alone changes nothing
            self.assertEqual(len(list((run / "objective_identity_history").glob("*.json"))), 1)
            # Accepting does not reach the domain the networks were trained in.
            (run / "metadata.json").write_text(json.dumps(dict(parameters=first["parameters"])))
            moved = dict(later, parameters=[dict(name="x", lower=0.0, upper=2.0, scale=1.0)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaisesRegex(ValueError, "Saved model metadata differs"):
                    verify(run, moved, accept_change=True)

    def test_a_record_of_the_first_schema_requires_review(self):
        from run.identity import verify

        current = self.record(**{"module:camb": "111", "library:camb": "333"})
        old = dict(wrapper="cobaya", objective="native", parameters=current["parameters"],
                   files={"input": "aaa", "module:camb": "111", "wrapper:likelihood.cobaya.CobayaLikelihood": "w1",
                          "wrapper:abc.ABC": "w2", "module:refused": "hpc", "module:__mp_main__": "launcher"})
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "objective_identity.json").write_text(json.dumps(dict(identity=old, historical_values_verified=True)))
            before = (run / "objective_identity.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "migration requires review"):
                verify(run, current)
            self.assertEqual((run / "objective_identity.json").read_bytes(), before)
            with self.assertWarns(RuntimeWarning):
                verify(run, current, accept_change=True)
            saved = json.loads((run / "objective_identity.json").read_text())
            self.assertEqual((saved["identity"], saved["historical_values_verified"]), (current, True))
            self.assertTrue(saved["accepted_changes"])
            self.assertEqual(json.loads(next((run / "objective_identity_history").glob("*.json")).read_text())["identity"], old)
            verify(run, current)  # acceptance is needed only once
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "objective_identity.json").write_text(json.dumps(dict(identity=dict(old, files=dict(old["files"], input="bbb")))))
            with self.assertRaisesRegex(ValueError, "changed: input"):
                verify(run, current)

    def test_schema_migration_does_not_trust_a_renamed_changed_source(self):
        from run.identity import verify

        current = self.record(**{"external:toy": "new objective"})
        old = dict(current, schema=1, files={"input": "aaa", "module:objective": "old objective"})
        old.pop("convention")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "objective_identity.json"
            path.write_text(json.dumps(dict(identity=old, historical_values_verified=True)))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "no longer loaded: module:objective"):
                verify(tmp, current)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((Path(tmp) / "objective_identity_history").exists())

    def test_objective_imports_beside_launcher_are_tracked(self):
        from run.identity import verify

        # Module import, imported scalar, and transitive relative import all affect the
        # objective. An unrelated helper beside the launcher should remain excluded.
        for statement in ("import review_helper as h\nfactor = h.factor",
                          "from review_helper import factor"):
            with self.subTest(statement=statement), tempfile.TemporaryDirectory() as tmp:
                launcher_dir = Path(tmp).resolve()
                package = launcher_dir / "review_dep"
                package.mkdir()
                (package / "__init__.py").write_text("")
                (package / "helper.py").write_text("from .nested import factor\n")
                (package / "nested.py").write_text("factor = 1\n")
                (launcher_dir / "review_helper.py").write_text("from review_dep.helper import factor\n")
                (launcher_dir / "review_objective.py").write_text(statement + "\ndef loglike(x, y):\n    return -factor*(x*x+y*y)\n")
                (launcher_dir / "review_unused.py").write_text("factor = 1\n")
                sys.path.insert(0, str(launcher_dir))
                try:
                    importlib.import_module("review_unused")
                    first = self.identity(tmp, external="import_module('review_objective').loglike", launcher=launcher_dir)
                    self.assertNotIn("module:review_unused", first["files"])
                    for name in ("review_objective", "review_helper", "review_dep", "review_dep.helper", "review_dep.nested"):
                        self.assertIn("module:" + name, first["files"])
                    verify(launcher_dir, first, new=True)
                    (launcher_dir / "review_helper.py").write_text("factor = 4\n")
                    changed = self.identity(tmp, external="import_module('review_objective').loglike", launcher=launcher_dir)
                    with self.assertRaisesRegex(ValueError, "changed: module:review_helper"):
                        verify(launcher_dir, changed)
                finally:
                    sys.path.remove(str(launcher_dir))
                    for name in list(sys.modules):
                        if name in ("review_objective", "review_helper", "review_unused") or name.startswith("review_dep"):
                            sys.modules.pop(name, None)

    def test_two_accepted_changes_in_one_second_preserve_both_records(self):
        from run.identity import verify

        with tempfile.TemporaryDirectory() as tmp, mock.patch("run.identity.time.strftime", return_value="same-second"):
            first = self.record()
            second, third = dict(first, convention=2), dict(first, convention=3)
            verify(tmp, first, new=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                verify(tmp, second, accept_change=True)
                verify(tmp, third, accept_change=True)
            records = [json.loads(p.read_text()) for p in (Path(tmp) / "objective_identity_history").glob("*.json")]
            self.assertEqual(sorted(r["identity"]["convention"] for r in records), [1, 2])


if __name__ == "__main__":
    unittest.main()
