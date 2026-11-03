"""The branch of a run at an earlier network (hpc/analysis/branch_run.py)."""

import contextlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc" / "analysis"))
import branch_run  # noqa: E402

SETTINGS = """acquisition:
  n_append: 30
  audit:
    start_iteration: 1     # not the boost's
    fraction: 0.1
  boost:
    n: 30                  # posterior points per boost
    start_iteration: 4     # networks 0 to 4 are the ordinary loop
convergence:
  max_iterations: 7
"""


def saved_run(parent, refused=None):
    run = parent / "20261008_143804_some_run"
    for folder, pattern, last in (("trained_models", "model_it_{}.keras", 6), ("training_data", "data_it_{}.csv", 6),
                                  ("convergence_stats", "chain_summary_it_{}.npz", 5), ("audit", "audit_it_{}.csv", 6),
                                  ("boost", "boost_it_{}.json", 5), ("analysis", "posterior_it_{}.npz", 6)):
        (run / folder).mkdir(parents=True)
        for k in range(4 if folder == "boost" else 0, last + 1):
            (run / folder / pattern.format(k)).write_text(
                f"x,loglkl\n{k},-{k + 1}\n" if folder in ("training_data", "audit") else f"{folder} {k}")
    (run / "client.yaml").write_text(SETTINGS)
    (run / "metadata.json").write_text(json.dumps(dict(parameters=[dict(name="x")])))
    if refused is not None:
        (run / "refused_value.json").write_text(json.dumps(dict(value=float(refused), setting=str(refused))))
    return run


def branch(run, *arguments):
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["branch_run.py", str(run), *arguments]), contextlib.redirect_stdout(out):
        branch_run.main()
    new = [p for p in run.parent.iterdir() if p != run]
    return new[0] if new else None, out.getvalue()


class BranchRun(unittest.TestCase):
    def test_the_branch_holds_the_run_up_to_the_network_and_boosts_from_there(self):
        with tempfile.TemporaryDirectory() as d:
            run = saved_run(Path(d), refused=-30000)
            before = {p: p.read_text() for p in run.rglob("*") if p.is_file()}
            new, said = branch(run, "--at", "2")
            self.assertTrue(new.name.endswith("_some_run") and new.name != run.name)
            names = sorted(str(p.relative_to(new)) for p in new.rglob("*") if p.is_file())
            self.assertEqual(names, sorted(
                ["BRANCHED_FROM.txt", "client.yaml", "metadata.json", "refused_value.json", "branch_provenance/imports.json"]
                + [f"trained_models/model_it_{k}.keras" for k in range(3)] + [f"training_data/data_it_{k}.csv" for k in range(3)]
                + [f"audit/audit_it_{k}.csv" for k in range(3)]
                + [f"convergence_stats/chain_summary_it_{k}.npz" for k in range(2)]))     # network 2 is sampled again
            settings, source = yaml.safe_load((new / "client.yaml").read_text()), yaml.safe_load(SETTINGS)
            self.assertEqual(settings["acquisition"]["boost"]["start_iteration"], 2)
            self.assertEqual(settings["acquisition"]["audit"]["start_iteration"], 1)
            settings["acquisition"]["boost"]["start_iteration"] = 4
            self.assertEqual(settings, source)
            self.assertIn("# posterior points per boost", (new / "client.yaml").read_text())      # the comments stay
            self.assertIn("CLIENT_REFUSED_VALUE=-30000 sbatch submit_client.sh", said)
            self.assertEqual(before, {p: p.read_text() for p in run.rglob("*") if p.is_file()})     # the source is not touched

    def test_source_journals_are_provenance_not_new_branch_calls(self):
        from run.objective_calls import ObjectiveCalls, summarize
        from types import SimpleNamespace
        from run.storage import digest

        with tempfile.TemporaryDirectory() as tmp:
            source = saved_run(Path(tmp))
            ledger = ObjectiveCalls(source, 0)
            toy = SimpleNamespace(evaluate_with_derived=lambda x: [-x[0] ** 2])
            for it in range(6):
                ledger.stage = f"acquisition/it{it}"
                ledger.evaluate(toy, [it])
            summarize(source)
            (source / "augmentation").mkdir()
            (source / "augmentation/augment_it_2.csv").write_text("x,loglkl\n2,-3\n3,-4\n4,nan\n")
            before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            new, _ = branch(source, "--at", "2")
            result = summarize(new)
            self.assertEqual(result["stages"], {})
            self.assertEqual(result["imports"], 2)  # data + duplicate audit/augmentation row + one new finite value
            self.assertEqual(result["cache_hits"], 0)
            record = result["inherited"]
            self.assertEqual(record["files"]["training_data/data_it_2.csv"]["rows"], 1)
            archived = new / "branch_provenance/source/objective_calls" / ledger.path.name
            self.assertEqual(archived.read_bytes(), ledger.path.read_bytes())
            self.assertEqual(record["source_history"][str(archived.relative_to(new / "branch_provenance"))], digest(archived))
            local = ObjectiveCalls(new, 0)
            local.stage = "boost/it2"
            local.evaluate(toy, [2.5])
            self.assertEqual(summarize(new)["stages"], {"boost/it2": dict(attempts=1, finite=1, unfinished=0)})
            self.assertEqual(before, {p: p.read_bytes() for p in source.rglob("*") if p.is_file()})
            # Branch again before its boosts: neither parent's work is a new call.
            parent = Path(tmp) / "second"
            parent.mkdir()
            with mock.patch.object(sys, "argv", ["branch_run.py", str(new), "--at", "1", "--out", str(parent)]):
                branch_run.main()
            child = next(parent.iterdir())
            self.assertEqual(summarize(child)["stages"], {})
            self.assertEqual(len(list((child / "branch_provenance/source").rglob("rank*.jsonl"))), 2)

    def test_the_marker_of_a_training_the_branch_does_not_hold_stays_behind(self):
        # `client.py augment` at network 2 writes augment_it_2.csv, and the training of network
        # 3 on those points leaves augment_it_2.trained.json: named after 2, about 3. Copied
        # into a branch at 2, the branch's own first training on the points ran the ordinary
        # schedule (training.augment_schedule_factor is for the first one).
        import numpy as np

        from dataset.augment import mark_trained, pending_augmentations

        with tempfile.TemporaryDirectory() as d:
            source = saved_run(Path(d))
            (source / "augmentation").mkdir()
            for k, trained in ((1, 2), (2, 3)):
                points = source / f"augmentation/augment_it_{k}.csv"
                points.write_text(f"x,loglkl\n{k}.5,-3\n{k}.25,-4\n")
                mark_trained([points], trained, 13, 3.0)                     # as the loop writes it
            (source / "augmentation/augment_it_0.csv").write_text("x,loglkl\n0.5,-3\n")
            (source / "augmentation/augment_it_0.trained.json").write_text('{"iteration": ')   # torn
            new, said = branch(source, "--at", "2")
            self.assertEqual(sorted(p.name for p in (new / "augmentation").iterdir()),
                             ["augment_it_0.csv", "augment_it_1.csv", "augment_it_1.trained.json", "augment_it_2.csv"])
            # what CLiENT makes of it: the points of network 2's augmentation await their first training
            inputs = np.array([[0.5], [1.5], [1.25], [2.5], [2.25]], np.float32)
            self.assertEqual([p.name for p in pending_augmentations(new, inputs)], ["augment_it_0.csv", "augment_it_2.csv"])
            self.assertEqual([p.name for p in pending_augmentations(source, inputs)], ["augment_it_0.csv"])

    def test_a_branch_is_there_whole_or_not_at_all(self):
        # A copy that stops half-way (a quota, a time limit) left a directory that read as a run,
        # which the threshold suite then took for the branch; and the record of what was
        # inherited is read by CLiENT after every acquisition of the branch.
        with tempfile.TemporaryDirectory() as d:
            source = saved_run(Path(d))
            (source / "objective_calls").mkdir()
            (source / "objective_calls/rank0-a.jsonl").write_text("")
            before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            with mock.patch.object(branch_run.shutil, "copytree", side_effect=OSError("Disk quota exceeded")), \
                    self.assertRaisesRegex(OSError, "quota"):
                branch(source, "--at", "2")
            self.assertEqual([p.name for p in Path(d).iterdir()], [source.name])
            self.assertEqual(before, {p: p.read_bytes() for p in source.rglob("*") if p.is_file()})
            new, _ = branch(source, "--at", "2")
            self.assertFalse(new.name.endswith(".partial"))
            self.assertEqual(json.loads((new / "branch_provenance/imports.json").read_text())["imports"], 1)

    def test_the_call_summary_of_a_branch_it_cannot_account_for_says_so(self):
        from run.objective_calls import ObjectiveCalls, summarize
        from types import SimpleNamespace

        toy = SimpleNamespace(evaluate_with_derived=lambda x: [-x[0] ** 2])
        with tempfile.TemporaryDirectory() as d:
            source = saved_run(Path(d))
            new, _ = branch(source, "--at", "2")
            record = new / "branch_provenance/imports.json"
            whole = record.read_text()
            for damaged in (whole[:len(whole) // 2], "[]", json.dumps(dict(json.loads(whole), imports="many"))):
                record.write_text(damaged)
                result = summarize(new)                    # after every acquisition: it must not stop the run
                self.assertIsNone(result["imports"])
                self.assertIsNone(result["inherited"])
                self.assertIn("imports.json cannot be read", result["branch_note"])
            record.write_text(whole)
            self.assertNotIn("branch_note", summarize(new))
            # A branch made by the tool as it was before 11 October 2026: the source's journals
            # are in its own objective_calls/, and its stages are the source's too.
            shutil.rmtree(new / "branch_provenance")
            ledger = ObjectiveCalls(new, 0)
            ledger.stage = "acquisition/it5"
            ledger.evaluate(toy, [5.0])
            result = summarize(new)
            self.assertEqual(result["stages"], {"acquisition/it5": dict(attempts=1, finite=1, unfinished=0)})
            self.assertIsNone(result["imports"])
            self.assertIn("include the source's calls", result["branch_note"])
            self.assertEqual(json.loads((new / "objective_calls/summary.json").read_text())["branch_note"], result["branch_note"])
            # a run that is no branch is as before
            (new / "BRANCHED_FROM.txt").unlink()
            result = summarize(new)
            self.assertEqual((result["imports"], result["inherited"]), (0, None))
            self.assertNotIn("branch_note", result)

    def test_a_branch_of_a_run_with_phase_records_can_be_continued(self):
        # CLiENT commits every phase's files by hash (run/storage.py) and a continued run
        # checks them all. The acquisition of network N committed the training set of N + 1,
        # which a branch at N leaves behind: its record must stay behind too.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from run.storage import PhaseStore

        with tempfile.TemporaryDirectory() as d:
            run = saved_run(Path(d))
            phases = PhaseStore(run)
            phases.complete("initial", [run / "training_data/data_it_0.csv", run / "metadata.json"])
            for k in range(6):
                phases.complete(f"training_it_{k}", [run / f"trained_models/model_it_{k}.keras", run / f"audit/audit_it_{k}.csv"])
                phases.complete(f"acquisition_it_{k}", [run / f"training_data/data_it_{k + 1}.csv"])
            phases.begin("training_it_6")                          # the source is still running, or was stopped there
            (run / "interrupted/training_it_5-earlier/trained_models").mkdir(parents=True)
            (run / "interrupted/training_it_5-earlier/trained_models/model_it_2.keras").write_text("set aside")
            new, _ = branch(run, "--at", "2")
            self.assertEqual(sorted(p.name for p in (new / "phases").glob("*.json")),
                             ["acquisition_it_0.json", "acquisition_it_1.json", "initial.json",
                              "training_it_0.json", "training_it_1.json", "training_it_2.json"])
            self.assertFalse((new / "interrupted").exists())
            store = PhaseStore(new)
            self.assertEqual(store.recover(), [])
            store.check()                                          # what a continued run does first
            # The same check on a copy that kept the record of the acquisition of network 2:
            (new / "phases/acquisition_it_2.json").write_text((run / "phases/acquisition_it_2.json").read_text())
            with self.assertRaisesRegex(RuntimeError, "artifact changed or missing"):
                store.check()

    def test_what_cannot_be_branched(self):
        with tempfile.TemporaryDirectory() as d:
            run = saved_run(Path(d))
            for arguments in (["--at", "9"], ["--at", "5"]):          # no such network; after the boosts began
                with self.assertRaises(SystemExit):
                    branch(run, *arguments)
            self.assertEqual([p.name for p in Path(d).iterdir()], [run.name])
            (run / "client.yaml").write_text(SETTINGS.replace("  boost:\n    n: 30                  # posterior points per boost\n"
                                                              "    start_iteration: 4     # networks 0 to 4 are the ordinary loop\n", ""))
            with self.assertRaises(SystemExit):
                branch(run, "--at", "2")
            self.assertEqual([p.name for p in Path(d).iterdir()], [run.name])

    def test_branch_rejects_uncommitted_or_broken_retained_checkpoint(self):
        from run.storage import PhaseStore

        for problem in ("incomplete", "malformed", "missing audit", "changed model", "missing phase"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                run = saved_run(Path(tmp))
                store = PhaseStore(run)
                model = run / "trained_models/model_it_2.keras"
                audit = run / "audit/audit_it_2.csv"
                store.complete("initial", [run / "training_data/data_it_0.csv"])
                store.complete("training_it_2", [model, audit])
                marker = run / "phases/training_it_2.json"
                if problem == "incomplete":
                    store.begin("training_it_2")
                elif problem == "malformed":
                    marker.write_text("{")
                elif problem == "missing audit":
                    audit.unlink()
                elif problem == "changed model":
                    model.write_text("uncommitted replacement")
                else:
                    marker.unlink()
                before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(SystemExit, "Cannot branch at network 2"):
                    branch(run, "--at", "2")
                self.assertEqual(list(Path(tmp).iterdir()), [run])  # reject before writing a branch
                self.assertEqual(before, {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()})

    def test_a_run_begun_before_the_phase_records_is_branched_before_they_start(self):
        # Every run made before 11 October 2026 has phase records from its first continuation
        # on only. Its earlier networks are branched as those of a run without any record.
        from run.storage import PhaseStore

        with tempfile.TemporaryDirectory() as d:
            run = saved_run(Path(d))
            phases = PhaseStore(run)
            phases.complete("acquisition_it_3", [run / "training_data/data_it_4.csv"])    # continued at network 3
            for k in (4, 5):
                phases.complete(f"training_it_{k}", [run / f"trained_models/model_it_{k}.keras"])
                phases.complete(f"acquisition_it_{k}", [run / f"training_data/data_it_{k + 1}.csv"])
            # the acquisition of network 3 committed the training set of network 4
            for at, kept in ((2, []), (3, []), (4, ["acquisition_it_3.json", "training_it_4.json"])):
                new, _ = branch(run, "--at", str(at))
                self.assertEqual(sorted(p.name for p in (new / "phases").glob("*.json")), kept)
                PhaseStore(new).check()
                shutil.rmtree(new)
            (run / "phases/training_it_4.json").unlink()            # within the records a missing one is refused
            with self.assertRaisesRegex(SystemExit, "no committed training phase"):
                branch(run, "--at", "4")

    def test_a_run_stopped_in_the_acquisition_of_the_network_is_branched_there(self):
        # The branch does that acquisition again, as its first boost: neither the unfinished
        # record nor what the source's own acquisition of that network wrote belongs to it.
        from run.storage import PhaseStore

        with tempfile.TemporaryDirectory() as d:
            run = saved_run(Path(d))
            phases = PhaseStore(run)
            phases.complete("initial", [run / "training_data/data_it_0.csv"])
            for k in range(5):
                phases.complete(f"training_it_{k}", [run / f"trained_models/model_it_{k}.keras"])
                if k < 4:
                    phases.complete(f"acquisition_it_{k}", [run / f"training_data/data_it_{k + 1}.csv"])
            phases.begin("acquisition_it_4")
            new, _ = branch(run, "--at", "4")
            self.assertEqual(sorted(p.name for p in (new / "phases").glob("*.json")),
                             [f"acquisition_it_{k}.json" for k in range(4)] + ["initial.json"]
                             + [f"training_it_{k}.json" for k in range(5)])
            self.assertFalse((new / "boost").exists())             # boost_it_4.json is the source's first boost
            store = PhaseStore(new)
            self.assertEqual(store.recover(), [])
            store.check()
            shutil.rmtree(new)
            phases.begin("acquisition_it_3")                       # an earlier unfinished phase is still refused
            with self.assertRaisesRegex(SystemExit, "acquisition_it_3.json: phase is incomplete"):
                branch(run, "--at", "4")

    def test_the_source_s_evidences_mode_searches_and_phase_history_stay_behind(self):
        # evidence_it_6_nlive2000.json has no dot after its iteration, which the filter asked for:
        # a branch at 2 received the source's evidences of networks 3 to 6, and the summary of
        # the branch listed them as the branch's.
        with tempfile.TemporaryDirectory() as d:
            run = saved_run(Path(d))
            for k in range(7):
                for name in (f"evidence/evidence_it_{k}_nlive2000.json", f"evidence/evidence_it_{k}_nlive2000.txt",
                             f"evidence/evidence_it_{k}_nlive2000_seed43.json", f"modes/modes_it_{k}.json",
                             f"phases/history/training_it_{k}-20261010_101500.json", f"derived_data/derived_it_{k}.csv",
                             f"audit/audit_evaluations_it_{k}.jsonl"):
                    (run / name).parent.mkdir(parents=True, exist_ok=True)
                    (run / name).write_text(name)
            new, _ = branch(run, "--at", "2")
            copied = sorted(str(p.relative_to(new)) for p in new.rglob("*") if p.is_file())
            for folder in ("evidence", "modes", "phases"):
                self.assertEqual([name for name in copied if name.startswith(folder + "/")], [])
            self.assertEqual([name for name in copied if name.startswith("derived_data/")],
                             [f"derived_data/derived_it_{k}.csv" for k in range(3)])
            self.assertEqual([name for name in copied if "audit_evaluations" in name],
                             [f"audit/audit_evaluations_it_{k}.jsonl" for k in range(3)])
            self.assertEqual([name for name in copied if re.search(r"_it_[3-6](?!\d)", name)], [])

    def test_a_source_whose_loop_handed_over_early_began_its_boosts_there(self):
        # With convergence.metric: tempered_batch_scatter the boosts begin when the ordinary loop
        # has converged, and boost/start.json records it; the settings hold the latest start.
        with tempfile.TemporaryDirectory() as d:
            run = saved_run(Path(d))
            (run / "boost" / "start.json").write_text(json.dumps(dict(start_iteration=2, configured=4)))
            with self.assertRaisesRegex(SystemExit, "the source's boosts start at 2, before network 3 .*its settings say 4"):
                branch(run, "--at", "3")                    # network 3 is trained on the source's first boost
            self.assertEqual([p.name for p in Path(d).iterdir()], [run.name])
            new, said = branch(run, "--at", "2")
            self.assertFalse((new / "boost" / "start.json").exists())       # the branch's boosts begin by its settings
            self.assertEqual(yaml.safe_load((new / "client.yaml").read_text())["acquisition"]["boost"]["start_iteration"], 2)
            self.assertIn("(the source: 2)", said)
            self.assertIn("the source's boosts began at 2 (boost/start.json)", (new / "BRANCHED_FROM.txt").read_text())
            shutil.rmtree(new)
            (run / "boost" / "start.json").write_text("{")              # torn: where the boosts began is not known
            with self.assertRaisesRegex(SystemExit, "cannot be read"):
                branch(run, "--at", "2")

    def test_without_a_recorded_refused_value_the_command_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            _, said = branch(saved_run(Path(d)), "--at", "3")
            self.assertNotIn("CLIENT_REFUSED_VALUE=", said)
            self.assertIn("if it had one", said)


class Settings(unittest.TestCase):
    """--set: a branch with a setting other than its source's, from its start."""

    def test_a_setting_is_replaced_or_added_and_the_rest_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = saved_run(Path(tmp))
            (run / "client.yaml").write_text(SETTINGS + "training:\n  warm_start: true     # inherit the weights\n")
            new, said = branch(run, "--at", "4", "--set", "training.lr_schedule=plateau",
                               "--set", "convergence.max_iterations=9")
            text = (new / "client.yaml").read_text()
            settings, source = yaml.safe_load(text), yaml.safe_load((run / "client.yaml").read_text())
            self.assertEqual(settings["training"], dict(warm_start=True, lr_schedule="plateau"))
            self.assertEqual(settings["convergence"]["max_iterations"], 9)
            for section in ("training", "convergence"):
                source.pop(section), settings.pop(section)
            self.assertEqual(settings, source)                                  # nothing else changed
            self.assertIn("# inherit the weights", text)
            self.assertIn("# posterior points per boost", text)
            noted = (new / "BRANCHED_FROM.txt").read_text()
            self.assertIn('training.lr_schedule: "plateau" (--set', noted)
            self.assertIn("convergence.max_iterations: 9 (--set", noted)

    def test_a_key_of_the_boosts_is_replaced_or_added_and_the_rest_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = saved_run(Path(tmp))
            new, _ = branch(run, "--at", "4", "--set", "acquisition.boost.n=50", "--set", "acquisition.boost.train_domain=1.25",
                            "--set", 'acquisition.audit={"chain": true}')     # a block replaced by its new value
            text = (new / "client.yaml").read_text()
            settings, source = yaml.safe_load(text), yaml.safe_load((run / "client.yaml").read_text())
            self.assertEqual(settings["acquisition"]["boost"], dict(n=50, start_iteration=4, train_domain=1.25))
            self.assertEqual(settings["acquisition"]["audit"], dict(chain=True))
            self.assertEqual(settings["acquisition"]["n_append"], 30)
            settings["acquisition"].pop("boost"), source["acquisition"].pop("boost")
            settings["acquisition"].pop("audit"), source["acquisition"].pop("audit")
            self.assertEqual(settings, source)                                  # nothing else changed
            self.assertIn("# networks 0 to 4 are the ordinary loop", text)
            noted = (new / "BRANCHED_FROM.txt").read_text()
            self.assertIn("acquisition.boost.n: 50 (--set", noted)
            self.assertIn("acquisition.boost.train_domain: 1.25 (--set", noted)

    def test_a_value_reads_back_as_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = saved_run(Path(tmp))
            new, _ = branch(run, "--at", "4", "--set", "acquisition.audit.posterior_tolerance=1.0e-6",
                            "--set", "acquisition.boost.train_domain=1.25", "--set", "convergence.label=last",
                            "--set", "acquisition.boost.flags=[1, 2]")
            settings = yaml.safe_load((new / "client.yaml").read_text())
            self.assertEqual(settings["acquisition"]["audit"]["posterior_tolerance"], 1.0e-6)
            self.assertEqual(settings["acquisition"]["boost"]["train_domain"], 1.25)
            self.assertEqual((settings["convergence"]["label"], settings["acquisition"]["boost"]["flags"]), ("last", [1, 2]))

    def test_a_key_of_a_subsection_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = saved_run(Path(tmp))
            new, _ = branch(run, "--at", "4", "--set", "acquisition.audit.chain=true")
            settings = yaml.safe_load((new / "client.yaml").read_text())
            self.assertEqual(settings["acquisition"]["audit"], dict(chain=True, start_iteration=1, fraction=0.1))

    def test_the_training_domain_stays_behind(self):
        # Built at the first training after the boosts' start (network 5 here), from the
        # training set there: a branch at the start builds its own.
        with tempfile.TemporaryDirectory() as tmp:
            run = saved_run(Path(tmp))
            (run / "boost" / "train_domain_it_5.npz").write_text("the domain of network 5's training")
            new, _ = branch(run, "--at", "4")
            self.assertFalse((new / "boost" / "train_domain_it_5.npz").exists())

    def test_what_cannot_be_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = saved_run(Path(tmp))
            for words, message in ((["--set", "training.lr_schedule=plateau"], "no top-level section training"),
                                   (["--set", "lr_schedule=plateau"], "SECTION.KEY=VALUE"),
                                   (["--set", "acquisition.boost=null"], "--at"),
                                   (["--set", "acquisition.boost.start_iteration=2"], "--at"),
                                   (["--set", "acquisition.boost.region.edge=1"], "SECTION.KEY=VALUE"),
                                   (["--set", "acquisition.inject.k=1"], "no section acquisition.inject")):
                with self.assertRaisesRegex(SystemExit, message):
                    branch(run, "--at", "4", *words)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], [run.name])   # no branch was made


if __name__ == "__main__":
    unittest.main()
