"""The summary of a run's evidences (hpc/analysis/evidence_summary.py)."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc" / "analysis"))
import evidence_summary  # noqa: E402


def target(region=None):
    return dict(schema=1, model_sha256="model", metadata_sha256="metadata", names=["x", "y"],
                domain=[[-1, 1], [-1, 1]], lower=[-1, -1], upper=[1, 1], region=region,
                temperature=1., objective="native saved-network output")


def evidence(lnZ, sigma, region=None, faces=(), seed=42):
    return dict(lnZ=lnZ, sigma_lnZ=sigma, lnZ_box=lnZ - 10.0, information_H=60.0, n_dead=120000, seconds=3600.0,
                region=region, target=target(region), seed=seed, iteration_cap_reached=False,
                box_faces_reached=[dict(parameter=p, side=s, mass=m) for p, s, m in faces])


class Summary(unittest.TestCase):
    def run_folder(self, tmp):
        run = Path(tmp) / "20261008_143804_run"
        for folder in ("evidence", "boost", "analysis", "training_data"):
            (run / folder).mkdir(parents=True)
        files = {"evidence_it_10_nlive2000.json": evidence(-15658.9, 0.18),
                 "evidence_it_15_nlive2000.json": evidence(-15657.2, 0.18, faces=[("Aksz", "upper", 2e-3)]),
                 "evidence_it_25_nlive2000.json": evidence(-15656.6, 0.17, region="region_it_25.npz"),
                 "evidence_it_25_nlive2000_seed43.json": evidence(-15656.4, 0.17, region="region_it_25.npz", seed=43)}
        for name, content in files.items():
            (run / "evidence" / name).write_text(json.dumps(content))
        (run / "boost" / "boost_it_15.json").write_text(json.dumps(dict(modes=[dict(log_evidence=-15657.1, log_evidence_spread=0.04)])))
        rng = np.random.default_rng(0)
        for iteration, scatter in ((15, 1.0), (25, 0.4)):
            network = -15520.0 - rng.random(500) * 20
            d = scatter * rng.standard_normal(500)
            np.savez(run / "analysis" / f"true_check_it_{iteration}.npz", network=network, truth=network + d, x=np.zeros((500, 2)),
                     target_json=json.dumps(target("region_it_25.npz" if iteration == 25 else None)))
            self.d = getattr(self, "d", {})
            self.d[iteration] = d
        (run / "training_data" / "data_it_15.csv").write_text("a,b,loglkl\n" + "1,2,3\n" * 7)
        return run

    def test_table_and_corrections(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_folder(tmp)
            with mock.patch.object(sys, "argv", ["evidence_summary.py", str(run)]), contextlib.redirect_stdout(io.StringIO()):
                evidence_summary.main()
            text = (run / "analysis" / "evidence.md").read_text()
            saved = json.loads((run / "analysis" / "evidence.json").read_text())["rows"]
            self.assertGreater((run / "analysis" / "evidence.png").stat().st_size, 5000)
        rows = {(r["network"], r["tag"]): r for r in saved}
        self.assertEqual(sorted(rows), [(10, ""), (15, ""), (25, ""), (25, "seed43")])
        # the correction is ln <exp(truth - network)>, and the corrected evidence adds it
        for iteration in (15, 25):
            d = self.d[iteration]
            expected = float(np.log(np.mean(np.exp(d))))
            row = rows[(iteration, "")]
            self.assertAlmostEqual(row["correction"]["value"], expected, places=10)
            self.assertAlmostEqual(row["lnZ_corrected"], row["lnZ"] + expected, places=10)
            self.assertGreater(row["lnZ_corrected_error"], row["sigma"])
            self.assertAlmostEqual(row["correction"]["scatter"], d.std(), places=10)
            self.assertEqual((row["correction"]["points"], row["correction"]["refused"], row["correction"]["failed"]), (500, 0, 0))
            self.assertIsNone(row["correction_unavailable"])
        self.assertGreater(rows[(15, "")]["correction"]["value"], 0.3)             # about s^2 / 2 for s = 1
        self.assertIsNone(rows[(10, "")]["correction"])
        self.assertIsNone(rows[(10, "")]["lnZ_corrected"])
        self.assertEqual(rows[(15, "")]["boost"]["value"], -15657.1)
        self.assertEqual(rows[(15, "")]["training_points"], 7)
        # against network 25, first seed: the networks are compared whatever each was integrated over
        self.assertIn("| 10 | - | -15658.90 | 0.18 | -2.30 (-9.3 sigma) |", text)
        self.assertIn("| 25 | - | -15656.60 | 0.17 | - |", text)
        self.assertIn("| 25 (seed43) |", text)
        self.assertIn("-15657.10 ± 0.04", text)
        self.assertIn("region |", text)
        self.assertIn("network 15: 2.0e-03 of the posterior within 2% of the box's upper face in Aksz", text)
        self.assertIn("network 25: 2 BEST runs span 0.20 in ln Z", text)
        self.assertIn("corrected ln Z of networks 15, 25: spread", text)
        self.assertIn("the networks were not integrated over the same part of the box (the whole box: 10, 15; "
                      "region_it_25.npz: 25, 25 (seed43))", text)
        self.assertNotIn("before they recorded their target", text)      # every file here records one
        self.assertNotIn("refused", text)
        self.assertNotIn("no correction", text)

    def test_refused_and_failed_true_values(self):
        # Network 15: a true check from before it marked refused points, 50 of its 500 points refused, 30 of
        # them carrying the run's refused value and 20 at -inf. Network 25: two evaluations that failed.
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_folder(tmp)
            (run / "refused_value.json").write_text(json.dumps(dict(value=-30700.0, setting="design", job="1")))
            saved = dict(np.load(run / "analysis" / "true_check_it_15.npz"))
            saved["truth"][:30], saved["truth"][30:50] = -30700.0, -np.inf
            np.savez(run / "analysis" / "true_check_it_15.npz", **saved)
            saved = dict(np.load(run / "analysis" / "true_check_it_25.npz"))
            saved["truth"][[7, 8]] = np.nan
            np.savez(run / "analysis" / "true_check_it_25.npz", **saved)
            with mock.patch.object(sys, "argv", ["evidence_summary.py", str(run)]), contextlib.redirect_stdout(io.StringIO()):
                evidence_summary.main()
            text = (run / "analysis" / "evidence.md").read_text()
            rows = {(r["network"], r["tag"]): r for r in json.loads((run / "analysis" / "evidence.json").read_text())["rows"]}
        d = self.d[15][50:]
        row = rows[(15, "")]
        expected = float(np.log(np.sum(np.exp(d)) / 500))                         # the mean is over all 500 drawn
        self.assertAlmostEqual(row["correction"]["value"], expected, places=10)
        self.assertAlmostEqual(row["correction"]["value"], float(np.log(np.mean(np.exp(d)))) + np.log(1 - 50 / 500), places=10)
        self.assertAlmostEqual(row["lnZ_corrected"], row["lnZ"] + expected, places=10)
        self.assertEqual((row["correction"]["points"], row["correction"]["refused"], row["correction"]["failed"]), (500, 50, 0))
        self.assertAlmostEqual(row["correction"]["scatter"], d.std(), places=10)   # of the points with a true value
        self.assertAlmostEqual(row["correction"]["mean"], d.mean(), places=10)
        self.assertIn("network 15: the likelihood refused 50 of the 500 posterior samples of its true check", text)
        self.assertIn(f"puts the correction {-np.log(0.9):.3f} below a mean over the others alone", text)
        for tag in ("", "seed43"):                                                 # network 25: no correction, and it says why
            self.assertIsNone(rows[(25, tag)]["correction"])
            self.assertIsNone(rows[(25, tag)]["lnZ_corrected"])
            self.assertEqual((rows[(25, tag)]["correction_unavailable"]["failed"], rows[(25, tag)]["correction_unavailable"]["refused"]), (2, 0))
        self.assertIn("network 25: no correction to the true likelihood: 2 of the 500 true evaluations failed", text)
        self.assertEqual(text.count("network 25: no correction"), 1)               # once for the network, not per BEST run
        self.assertIn("| 25 | - | -15656.60 | 0.17 | - | 60.0 | 120000 | 60 | region | - | - | - | - | - | - |", text)
        # the table has the columns it had
        self.assertIn("ln Z (boost) | scatter of truth - network | effective samples | correction | ln Z (corrected) | error |\n"
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n", text)

    def test_recorded_targets_must_match_and_earlier_results_are_joined_by_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_folder(tmp)
            base = evidence(-15656.3, .17, region="region_it_25.npz", seed=44)
            for key, value in (("region", None), ("model_sha256", "changed"),
                               ("lower", [-.5, -1]), ("metadata_sha256", "changed")):
                changed = json.loads(json.dumps(base))
                changed["target"][key] = value
                (run / "evidence" / f"evidence_it_25_nlive2000_{key}.json").write_text(json.dumps(changed))
            base.pop("target")                      # an evidence from before targets were recorded
            (run / "evidence" / "evidence_it_25_nlive2000_legacy.json").write_text(json.dumps(base))
            rows = {row["tag"]: row for row in evidence_summary.collect(run)[25]["best"]}
            # both record a target: the same one, or no correction
            for tag in ("", "seed43"):
                self.assertTrue(rows[tag]["target_recorded"])
                self.assertIsNotNone(rows[tag]["correction"])
            for tag in ("region", "model_sha256", "lower", "metadata_sha256"):
                self.assertNotIn("correction", rows[tag])
                self.assertIn("different targets", rows[tag]["correction_unavailable"]["unavailable"])
            # the earlier evidence: the true check of its network, as before the rule, and marked
            self.assertEqual(rows["legacy"]["correction"], rows[""]["correction"])
            self.assertFalse(rows["legacy"]["target_recorded"])
            with mock.patch.object(sys, "argv", ["evidence_summary.py", str(run)]), contextlib.redirect_stdout(io.StringIO()):
                evidence_summary.main()
            text = (run / "analysis" / "evidence.md").read_text()
            self.assertIn("- networks 25 (legacy): the evidence, the true check or both were made before they recorded "
                          "their target (11 October 2026). The correction is joined by the network's number", text)
            self.assertIn("network 25 (region): no correction: true check and evidence sampled different targets", text)
            # a true check from before the rule corrects every evidence of its network, each marked
            saved = dict(np.load(run / "analysis" / "true_check_it_25.npz"))
            saved.pop("target_json")
            np.savez(run / "analysis" / "true_check_it_25.npz", **saved)
            rows = evidence_summary.collect(run)[25]["best"]
            self.assertTrue(all(row["correction"] is not None and not row["target_recorded"] for row in rows))

    def test_the_table_of_a_run_made_before_targets_were_recorded_keeps_its_numbers(self):
        # Every existing run: no file records a target. Withholding the corrections emptied their
        # tables, and their evidences take hours each.
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_folder(tmp)
            for file in (run / "evidence").glob("*.json"):
                saved = json.loads(file.read_text())
                saved.pop("target")
                file.write_text(json.dumps(saved))
            for file in (run / "analysis").glob("true_check_it_*.npz"):
                saved = dict(np.load(file))
                saved.pop("target_json")
                np.savez(file, **saved)
            # one evidence of network 25 made since, with the stop at a fraction of the evidence
            (run / "evidence" / "evidence_it_25_nlive2000_ref.json").write_text(json.dumps(dict(
                evidence(-15656.5, 0.17, region=dict(file="boost/region_it_25.npz", sha256="hash", edge=2.0), seed=42),
                target=None, live_fraction=1e-3)))
            with mock.patch.object(sys, "argv", ["evidence_summary.py", str(run)]), contextlib.redirect_stdout(io.StringIO()):
                evidence_summary.main()
            text = (run / "analysis" / "evidence.md").read_text()
            rows = {(r["network"], r["tag"]): r for r in json.loads((run / "analysis" / "evidence.json").read_text())["rows"]}
        for key in ((15, ""), (25, ""), (25, "seed43"), (25, "ref")):
            self.assertIsNotNone(rows[key]["lnZ_corrected"])
        self.assertIn("| 10 | - | -15658.90 | 0.18 | -2.30 (-9.3 sigma) |", text)
        self.assertIn("corrected ln Z of networks 15, 25: spread", text)
        self.assertIn("- networks 15, 25, 25 (ref), 25 (seed43): the evidence, the true check or both were made before", text)
        self.assertIn("region_it_25.npz: 25, 25 (ref), 25 (seed43)", text)       # the file's name, however it was recorded
        # the two runs at the old stop are a test of the sampler; the one at the new stop is another method
        self.assertIn("network 25: 2 BEST runs span 0.20 in ln Z", text)
        self.assertIn("network 25: its BEST runs differ in target, prior convention or stopping rule", text)

    def test_a_true_check_of_a_region_with_several_ellipsoids_that_records_no_sampler_is_named(self):
        # Before 13 October 2026 a true check did not record how its chain was sampled, and
        # until the 12th a chain of several ellipsoids kept the shares of its start. Such a
        # true check still corrects (the earlier results keep their numbers), with a note.
        note = "network 25: its true check does not record how it was sampled, and the region of network 25 has 2 ellipsoids"
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_folder(tmp)
            self.assertNotIn("does not record how it was sampled", self.summary(run))        # the region is not at hand
            np.savez(run / "boost" / "region_it_25.npz", means=np.zeros((1, 2)))
            self.assertNotIn("does not record how it was sampled", self.summary(run))        # one ellipsoid
            np.savez(run / "boost" / "region_it_25.npz", means=np.zeros((2, 2)))
            text = self.summary(run)
            self.assertIn(note, text)
            self.assertIn("make the true check again", text)
            table = json.loads((run / "analysis" / "evidence.json").read_text())["rows"]
            rows = {(row["network"], row["tag"]): row for row in table}
            self.assertIsNotNone(rows[(25, "")]["lnZ_corrected"])                           # corrected all the same
            self.assertIn("2 ellipsoids", rows[(25, "")]["unrecorded_sampler"])
            self.assertIsNone(rows[(15, "")]["unrecorded_sampler"])
            # the same true check as it is written since: the sampler is recorded, nothing to name
            check = run / "analysis" / "true_check_it_25.npz"
            saved = dict(np.load(check))
            np.savez(check, **saved, sampling_json=json.dumps(dict(schema=1, version=1, method="region-cell bridge mixture",
                                                                   cells=[dict(cell=0, efficiency=0.6), dict(cell=1, efficiency=0.6)],
                                                                   unreliable_cells=[])),
                     sampling_correction_unavailable="")
            self.assertNotIn("does not record how it was sampled", self.summary(run))

    def realistic(self, run):
        """The evidence files as `client.py evidence` writes them: the prior convention, and a
        priors table whose rows also hold the integration box and the mass measured at its faces."""
        for file in (run / "evidence").glob("*.json"):
            saved = json.loads(file.read_text())
            saved.update(convention="Cobaya's normalised priors", ln_prior_correction=0.0, tolerance=1e-3, live_fraction=9e-4)
            saved["priors"] = [dict(name=n, lower=-1.0, upper=1.0, source="Cobaya prior", box_lower=-1.0, box_upper=1.0,
                                    edge_mass_lower=0.0, edge_mass_upper=0.0) for n in ("x", "y")]
            file.write_text(json.dumps(saved))

    def change(self, run, name, **changes):
        file = run / "evidence" / name
        saved = json.loads(file.read_text())
        for key, value in changes.items():
            if key == "edge_mass_lower":
                saved["priors"][0][key] = value
            else:
                saved[key] = value
        file.write_text(json.dumps(saved))

    def summary(self, run):
        with mock.patch.object(sys, "argv", ["evidence_summary.py", str(run)]), contextlib.redirect_stdout(io.StringIO()):
            evidence_summary.main()
        return (run / "analysis" / "evidence.md").read_text()

    def test_the_mass_measured_at_a_face_is_not_the_prior_convention(self):
        # The priors table of an evidence carries the box and the posterior mass at its faces,
        # a sampled number that differs in any two runs. Compared with the priors, every real
        # evidence was "another prior convention" than every other: no difference to the last
        # network, no spread of the corrected values, no seed scatter.
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_folder(tmp)
            self.realistic(run)
            for k, name in enumerate(sorted(p.name for p in (run / "evidence").glob("*.json"))):
                self.change(run, name, edge_mass_lower=1e-4 * (k + 1))
            text = self.summary(run)
            self.assertNotIn("another prior convention", text)
            self.assertIn("| 10 | - | -15658.90 | 0.18 | -2.30 (-9.3 sigma) |", text)
            self.assertIn("network 25: 2 BEST runs span 0.20 in ln Z", text)
            self.assertIn("corrected ln Z of networks 15, 25: spread", text)
            # another prior range for one parameter is another convention, and is not compared
            saved = json.loads((run / "evidence" / "evidence_it_10_nlive2000.json").read_text())
            saved["priors"][1].update(lower=-0.5, source="user (replaces Cobaya's uniform prior)")
            saved["ln_prior_correction"] = 0.2876820724517809
            (run / "evidence" / "evidence_it_10_nlive2000.json").write_text(json.dumps(saved))
            text = self.summary(run)
            self.assertIn("| 10 | - | -15658.90 | 0.18 | - |", text)
            self.assertIn("networks 10: another prior convention than network 25's", text)

    def test_another_tolerance_or_a_capped_run_is_no_measure_of_the_sampler_s_scatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_folder(tmp)
            self.realistic(run)
            self.assertIn("network 25: 2 BEST runs span 0.20 in ln Z", self.summary(run))
            self.change(run, "evidence_it_25_nlive2000_seed43.json", tolerance=0.1, live_fraction=0.09)
            text = self.summary(run)
            self.assertNotIn("BEST runs span", text)
            self.assertIn("network 25: its BEST runs differ in target, prior convention or stopping rule", text)
            self.change(run, "evidence_it_25_nlive2000_seed43.json", tolerance=1e-3, live_fraction=9e-4, iteration_cap_reached=True)
            text = self.summary(run)
            self.assertNotIn("BEST runs span", text)                     # the same rule, but one stopped at its cap
            self.assertIn("network 25: BEST stopped at its iteration cap", text)
            self.assertIn("| 25 (seed43) | - | -15656.40 | 0.17 | +0.20", text)   # the row and its comparison stay

    def test_a_recorded_target_that_cannot_be_used_is_not_an_earlier_result(self):
        for damage in ("{not valid JSON", json.dumps(dict(target(), schema=2)), json.dumps(dict(schema=1, names=["x", "y"])),
                       json.dumps([1, 2])):
            with self.subTest(damage=damage[:24]), tempfile.TemporaryDirectory() as tmp:
                run = self.run_folder(tmp)
                saved = dict(np.load(run / "analysis" / "true_check_it_25.npz"))
                saved["target_json"] = damage
                np.savez(run / "analysis" / "true_check_it_25.npz", **saved)
                rows = evidence_summary.collect(run)[25]["best"]
                self.assertTrue(all("correction" not in row for row in rows))
                self.assertTrue(all("the true check's recorded target" in row["correction_unavailable"]["unavailable"] for row in rows))
                text = self.summary(run)
                self.assertNotIn("before they recorded their target", text)
                self.assertIn("network 25 (default): no correction: the true check's recorded target", text)
        # the same for the evidence's own record; one from before targets were (no entry, or null) is joined as before
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_folder(tmp)
            self.change(run, "evidence_it_25_nlive2000.json", target=dict(target("region_it_25.npz"), schema=7))
            self.change(run, "evidence_it_25_nlive2000_seed43.json", target=None)
            rows = {row["tag"]: row for row in evidence_summary.collect(run)[25]["best"]}
            self.assertIn("the evidence's recorded target has schema 7", rows[""]["correction_unavailable"]["unavailable"])
            self.assertIsNotNone(rows["seed43"]["correction"])
            self.assertFalse(rows["seed43"]["target_recorded"])
            self.assertNotIn("BEST runs span", self.summary(run))        # nor is the unusable one "the same target"

    def test_what_counts_as_the_sampler_s_scatter(self):
        native = dict(convention="native", priors=[1], ln_prior_correction=0.)
        a = dict(target=target(), normalisation=native, stop=["fraction of the evidence", 1e-3], region=None)
        self.assertTrue(evidence_summary.same_run_of_the_sampler(a, dict(a)))
        other_prior = dict(a, normalisation=dict(convention="normalised", priors=[2], ln_prior_correction=-1.))
        self.assertFalse(evidence_summary.same_convention(a, other_prior))
        self.assertFalse(evidence_summary.same_run_of_the_sampler(a, other_prior))
        self.assertFalse(evidence_summary.same_run_of_the_sampler(a, dict(a, target=dict(target(), model_sha256="other network"))))
        self.assertFalse(evidence_summary.same_run_of_the_sampler(a, dict(a, stop=["BEST's own", 1e-3])))
        self.assertFalse(evidence_summary.same_run_of_the_sampler(a, dict(a, stop=["fraction of the evidence", 0.1])))
        self.assertFalse(evidence_summary.same_run_of_the_sampler(a, dict(a, target=None, target_problem="recorded target cannot be read")))
        # without a recorded target on one side: by what they were integrated over
        earlier = dict(a, target=None)
        self.assertTrue(evidence_summary.same_run_of_the_sampler(a, earlier))
        self.assertFalse(evidence_summary.same_run_of_the_sampler(earlier, dict(earlier, region="region_it_25.npz")))
        self.assertEqual(evidence_summary.integrated_over(dict(region=dict(file="boost/region_it_25.npz"))), "region_it_25.npz")
        self.assertEqual(evidence_summary.integrated_over(dict(region="region_it_25.npz")), "region_it_25.npz")
        self.assertIsNone(evidence_summary.integrated_over(dict(region=None)))

    def test_a_run_without_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "evidence").mkdir()
            with mock.patch.object(sys, "argv", ["evidence_summary.py", tmp]), self.assertRaises(SystemExit):
                evidence_summary.main()


class Unsampled(unittest.TestCase):
    """An evidence whose own posterior lies where its true check has no sample is not corrected.

    The network: -10 - (x^2 + (y / 0.1)^2) / 2. Its true check has 500 draws of that posterior;
    the evidence's chain is written in the order (y, x), the check's in (x, y). With `false`, all
    but 1% of the evidence's posterior sits on a maximum at x = 6 that the truth lacks and that
    the check's chain never reached (the SINU branch's network 21 of 13 October)."""

    @staticmethod
    def network(x, y):
        return -10.0 - 0.5 * (x ** 2 + (y / 0.1) ** 2)

    def make(self, tmp, false):
        run = Path(tmp) / "run"
        for folder in ("evidence", "analysis"):
            (run / folder).mkdir(parents=True)
        rng = np.random.default_rng(3)
        x, y = rng.standard_normal(500), 0.1 * rng.standard_normal(500)
        value = self.network(x, y)
        np.savez(run / "analysis" / "true_check_it_5.npz", x=np.column_stack([x, y]).astype(np.float32), network=value,
                 truth=value + 0.1 * rng.standard_normal(500), names=np.array(["x", "y"]), target_json=json.dumps(target()))
        cx, cy = rng.standard_normal(20000), 0.1 * rng.standard_normal(20000)
        chain = self.network(cx, cy)
        if false:
            cx[:19800], cy[:19800] = 6.0 + 0.05 * rng.standard_normal(19800), 0.01 * rng.standard_normal(19800)
            chain[:19800] = -5.0 - 0.5 * ((cx[:19800] - 6.0) / 0.05) ** 2
        np.savetxt(run / "evidence" / "evidence_it_5_nlive2048.txt", np.column_stack([np.ones(20000), -chain, cy, cx]))
        (run / "evidence" / "evidence_it_5_nlive2048.json").write_text(json.dumps(dict(evidence(-12.0, 0.1), param_names=["y", "x"])))
        return run

    def test_an_evidence_where_its_check_sampled_is_corrected(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = evidence_summary.collect(self.make(tmp, false=False))[5]["best"][0]
        self.assertIn("correction", row)
        self.assertLess(row["unsampled"]["higher"], 0.02)
        self.assertLess(row["unsampled"]["farther"], 0.02)       # the columns are matched by name
        self.assertEqual(row["unsampled"]["samples"], 500)

    def test_the_measure_holds_in_many_parameters(self):
        # 500 draws of a 49-parameter Gaussian against 20 000 more of the same: about 0.2% of
        # them lie farther out than the farthest of the 500. Measured from the 500's own mean
        # and covariance, the 500 look closer than new draws do, and 2 to 5% did, which withheld
        # the correction of a sound 49-parameter network (network 17 of 13 October).
        rng = np.random.default_rng(7)
        d = 49
        mix = rng.normal(size=(d, d)) / np.sqrt(d) + np.eye(d)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            for folder in ("evidence", "analysis"):
                (run / folder).mkdir(parents=True)
            names = [f"p{i}" for i in range(d)]
            points, chain = rng.standard_normal((500, d)) @ mix, rng.standard_normal((20000, d)) @ mix
            value = lambda x: -0.5 * (np.linalg.solve(mix.T, x.T) ** 2).sum(axis=0)
            np.savez(run / "analysis" / "true_check_it_5.npz", x=points.astype(np.float32), network=value(points),
                     truth=value(points), names=np.array(names), target_json=json.dumps(target()))
            np.savetxt(run / "evidence" / "evidence_it_5_nlive2048.txt", np.column_stack([np.ones(20000), -value(chain), chain]))
            (run / "evidence" / "evidence_it_5_nlive2048.json").write_text(json.dumps(dict(evidence(-12.0, 0.1), param_names=names)))
            row = evidence_summary.collect(run)[5]["best"][0]
        self.assertIn("correction", row)
        self.assertLess(row["unsampled"]["farther"], 0.01)
        self.assertLess(row["unsampled"]["higher"], 0.01)

    def test_an_evidence_on_a_maximum_its_check_never_reached_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp, false=True)
            row = evidence_summary.collect(run)[5]["best"][0]
            self.assertNotIn("correction", row)
            self.assertGreater(row["unsampled"]["higher"], 0.98)
            self.assertGreater(row["unsampled"]["farther"], 0.98)
            self.assertRegex(row["correction_unavailable"]["unavailable"],
                             r"^its own posterior lies where its true check has no sample \(9\d\.\d% at network values above "
                             r"the highest of the check's 500 samples \(-10\.\d\d; the evidence's chain reaches -5\.\d\d\); "
                             r"9\d\.\d% farther from their mean than the farthest of them; about 0\.2% each")
            with mock.patch.object(sys, "argv", ["evidence_summary.py", str(run)]), contextlib.redirect_stdout(io.StringIO()):
                evidence_summary.main()
            text = (run / "analysis" / "evidence.md").read_text()
            saved = json.loads((run / "analysis" / "evidence.json").read_text())["rows"][0]
            self.assertIn("- network 5 (default): no correction: its own posterior lies where its true check has no sample", text)
            self.assertIsNone(saved["lnZ_corrected"])
            self.assertGreater(saved["unsampled"]["higher"], 0.98)

            # Without the chain beside the evidence (an output from before chains were written)
            # there is nothing to compare: the correction is made as before.
            (run / "evidence" / "evidence_it_5_nlive2048.txt").unlink()
            row = evidence_summary.collect(run)[5]["best"][0]
            self.assertIsNone(row["unsampled"])
            self.assertIn("correction", row)


class Correction(unittest.TestCase):
    """ln <exp(truth - network)> with points the likelihood refused: against the analytic value."""

    REFUSED_VALUE = -30700.0

    def check(self, tmp, truth, network, refused_value=None, **arrays):
        file = Path(tmp) / "true_check_it_1.npz"
        np.savez(file, network=network, truth=truth, x=np.zeros((len(truth), 2)), **arrays)
        return evidence_summary.correction(file, np.random.default_rng(1), repeats=200, refused_value=refused_value)

    def test_refused_points_weigh_nothing_and_stay_in_the_mean(self):
        network = -15520.0 - np.random.default_rng(4).random(600) * 20
        refused = np.zeros(600, bool)
        refused[::10] = True                                           # a refused share f = 0.1
        with tempfile.TemporaryDirectory() as tmp:
            # truth = network + 0.25 everywhere else: the mean over all points is exactly (1 - f) exp(0.25)
            truth = np.where(refused, self.REFUSED_VALUE, network + 0.25)
            found = self.check(tmp, truth, network, refused=refused)
            self.assertAlmostEqual(found["value"], 0.25 + np.log(0.9), places=9)
            self.assertEqual((found["points"], found["refused"], found["failed"]), (600, 60, 0))
            self.assertAlmostEqual(found["mean"], 0.25, places=9)
            self.assertAlmostEqual(found["scatter"], 0.0, places=9)
            self.assertAlmostEqual(found["effective"], 540.0, places=6)
            self.assertAlmostEqual(found["error"], np.sqrt(0.1 / 0.9 / 600), delta=0.004)    # the binomial error of the share
            # a scatter as well: the weights of the others, over every point drawn
            d = 0.7 * np.random.default_rng(5).standard_normal(600)
            found = self.check(tmp, np.where(refused, self.REFUSED_VALUE, network + d), network, refused=refused)
            self.assertAlmostEqual(found["value"], float(np.log(np.sum(np.exp(d[~refused])) / 600)), places=10)
            self.assertAlmostEqual(found["scatter"], d[~refused].std(), places=10)
            self.assertTrue(np.isfinite(found["error"]) and found["error"] > 0)
            # the same from -inf, and from a file without the mark when the run's refused value is given
            marked = found
            self.assertEqual(self.check(tmp, np.where(refused, -np.inf, network + d), network), marked)
            self.assertEqual(self.check(tmp, np.where(refused, self.REFUSED_VALUE, network + d), network, self.REFUSED_VALUE), marked)
            # without the mark and without the value the convention is taken as a likelihood: it weighs
            # nothing either, but it is in the scatter
            taken = self.check(tmp, np.where(refused, self.REFUSED_VALUE, network + d), network)
            self.assertEqual((taken["refused"], taken["failed"]), (0, 0))
            self.assertGreater(taken["scatter"], 4000.0)

    def test_without_refused_points_it_is_the_plain_mean(self):
        network = -15520.0 - np.random.default_rng(4).random(300) * 20
        d = 0.5 * np.random.default_rng(6).standard_normal(300)
        with tempfile.TemporaryDirectory() as tmp:
            found = self.check(tmp, network + d, network, self.REFUSED_VALUE, refused=np.zeros(300, bool))
            self.assertAlmostEqual(found["value"], float(np.log(np.mean(np.exp(d)))), places=12)
            self.assertEqual((found["points"], found["refused"], found["failed"]), (300, 0, 0))
            self.assertEqual(sorted(found), ["effective", "error", "failed", "mean", "points", "refused", "scatter", "value"])

    def test_a_failed_evaluation_leaves_no_correction(self):
        network = -15520.0 - np.random.default_rng(4).random(100) * 20
        truth = network + 0.1
        truth[[5, 6, 7]] = np.nan                                      # failed: what these points weigh is not known
        truth[[20, 21]] = -np.inf
        with tempfile.TemporaryDirectory() as tmp:
            found = self.check(tmp, truth, network)
            self.assertIsNone(found["value"])
            self.assertEqual((found["points"], found["refused"], found["failed"]), (100, 2, 3))
            self.assertIn("3 of the 100 true evaluations failed", found["unavailable"])
            truth[[5, 6, 7]] = np.inf                                  # neither a value nor a refusal
            self.assertEqual(self.check(tmp, truth, network)["failed"], 3)

    def test_every_reason_for_no_correction_is_named(self):
        # A true check whose samples cannot carry a mean (the weight of a cell of the region
        # was not reliable) and which also has failed evaluations: the note named the first only.
        network = -15520.0 - np.random.default_rng(4).random(100) * 20
        truth = network + 0.1
        marked = "unreliable region-cell mixture weights (cells [1])"
        with tempfile.TemporaryDirectory() as tmp:
            found = self.check(tmp, truth, network, sampling_correction_unavailable=marked)
            self.assertEqual((found["value"], found["unavailable"]), (None, marked))
            self.assertIsNotNone(self.check(tmp, truth, network, sampling_correction_unavailable="")["value"])
            truth[[5, 6, 7]] = np.nan
            found = self.check(tmp, truth, network, sampling_correction_unavailable=marked)
            self.assertIsNone(found["value"])
            self.assertIn(marked + "; 3 of the 100 true evaluations failed", found["unavailable"])
            found = self.check(tmp, truth, network, sampling_correction_unavailable="")
            self.assertTrue(found["unavailable"].startswith("3 of the 100 true evaluations failed"))

    def test_fewer_than_ten_points_with_a_true_value(self):
        network = -15520.0 - np.random.default_rng(4).random(40) * 20
        truth = np.where(np.arange(40) < 9, network + 0.1, -np.inf)
        with tempfile.TemporaryDirectory() as tmp:
            found = self.check(tmp, truth, network)
            self.assertIsNone(found["value"])
            self.assertEqual((found["refused"], found["failed"]), (31, 0))
            self.assertIn("9 of the 40 points have a true value: fewer than 10", found["unavailable"])
            truth[9] = network[9] + 0.1
            self.assertAlmostEqual(self.check(tmp, truth, network)["value"], 0.1 + np.log(10 / 40), places=9)

    def test_the_run_s_refused_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(evidence_summary.refused_value_of(Path(tmp)))
            (Path(tmp) / "refused_value.json").write_text(json.dumps(dict(value=-214.0, setting="design")))
            self.assertEqual(evidence_summary.refused_value_of(Path(tmp)), -214.0)
            (Path(tmp) / "refused_value.json").write_text("{")
            self.assertIsNone(evidence_summary.refused_value_of(Path(tmp)))


class WhatTheCommandsWrite(unittest.TestCase):
    """The summary on files made by the programs themselves, not on files shaped by a test.
    The comparison of prior conventions passed every test on made-up files and blanked the
    table of real ones, whose priors table the fixtures did not have."""

    def test_two_seeds_of_an_evidence_and_the_true_check_of_their_network(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_evidence import best_available, cobaya_run, command, quadratic_model

        if not best_available():
            self.skipTest("BEST is not installed or on $BEST_PATH")
        import tensorflow as tf
        from likelihood.surrogate import SurrogateMetadata
        from sampling.provenance import target_identity
        from sampling.targets import resolve

        with tempfile.TemporaryDirectory() as d:
            run = cobaya_run(d, quadratic_model([0.3, -0.2], [0.10, 0.20], 0.0), [-1.0, -1.0], [1.0, 1.0])
            command(run, "--n-live", "300", "--quiet")
            command(run, "--n-live", "300", "--quiet", "--seed", "43", "--tag", "seed43")
            # the true check's file, with the target as analysis/true_check.py records it (report.Run.target_identity)
            rng = np.random.default_rng(2)
            x = (np.array([0.3, -0.2]) + np.array([0.10, 0.20]) * rng.standard_normal((400, 2))).astype(np.float32)
            network = tf.keras.models.load_model(run / "trained_models" / "model_it_2.keras", compile=False)(x).numpy()[:, 0]
            recorded = target_identity(run, 2, SurrogateMetadata.load(run / "metadata.json"), resolve(run, 2, "run"))
            (run / "analysis").mkdir()
            np.savez(run / "analysis" / "true_check_it_2.npz", x=x, network=network, truth=network + 0.3 * rng.standard_normal(400),
                     refused=np.zeros(400, bool), names=np.array(["x0", "x1"]), target_json=json.dumps(recorded, sort_keys=True))
            saved = [json.loads(file.read_text()) for file in sorted((run / "evidence").glob("*.json"))]
            # what made the old comparison fail on every real pair: the table has the measured masses
            self.assertIn("edge_mass_lower", saved[0]["priors"][0])
            self.assertEqual(saved[0]["target"], recorded)             # the evidence and the check name the same target
            with mock.patch.object(sys, "argv", ["evidence_summary.py", str(run)]), contextlib.redirect_stdout(io.StringIO()):
                evidence_summary.main()
            text = (run / "analysis" / "evidence.md").read_text()
            rows = evidence_summary.collect(run)[2]["best"]
        self.assertEqual([row["tag"] for row in rows], ["", "seed43"])
        for row in rows:
            self.assertTrue(row["target_recorded"])
            self.assertIsNotNone(row["correction"])
            # BEST's own chain against the check's draws of the same posterior: both shares near 1 / 401
            self.assertEqual(row["unsampled"]["samples"], 400)
            self.assertLess(max(row["unsampled"]["higher"], row["unsampled"]["farther"]), 0.02)
        self.assertTrue(evidence_summary.same_convention(*rows))
        self.assertEqual(rows[0]["stop"], ["fraction of the evidence", 1e-3])
        self.assertIn("network 2: 2 BEST runs span", text)
        self.assertRegex(text, r"\| 2 \(seed43\) \| - \| [-+0-9.]+ \| [0-9.]+ \| [-+][0-9.]+ \([-+][0-9.]+ sigma\) \|")
        for unwanted in ("another prior convention", "before they recorded their target", "no correction", "differ in target"):
            self.assertNotIn(unwanted, text)


if __name__ == "__main__":
    unittest.main()
