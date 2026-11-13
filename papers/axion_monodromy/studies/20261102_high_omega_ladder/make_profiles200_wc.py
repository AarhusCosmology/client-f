"""The 2D profile studies of the H0WC combined emulator for omega <= 200 (7 November 2026): runs/axion_wc_combined200/
profiles_*.yaml, set up as those of PDP+ (make_profiles200.py, make_profiles200_narrow.py,
make_profiles200_aosc_phi_windows.py) with what they taught.

  (A_osc, omega) and (omega, phi): the four omega windows of PDP+ ([1, 50.75], [50.75, 100.5], [100.5, 150.25],
    [150.25, 200]), the hybrid scan; A_osc to 0.06 and 0.09 in the two lowest windows, where PDP+'s scans put node optima
    at the region's edge with 0.08 and 0.1, and to 0.1 (the training box) in the others. Fallbacks of (A_osc, omega) with
    the older scan in fallback/.
  (A_osc, phi): omega free over the prior, the older scan, A_osc to 0.1, as PDP+.
  windows/ (submitted separately): (A_osc, phi) with omega searched only around the highest maxima of the H0WC scan
    (Fig. scan, which starts at omega = 55; CLiENT's search_ranges), for the reason PDP+ needed them: its study with omega
    free was up to 23 too low in chi^2 at omega 179 and 196, the second and third maxima of its scan, and up to 3.6 at
    18-22 and 30, its next ones. H0WC's scan: 157.75 (6.2), 141.5 (5.6), 81.5 (5.0), 195.5 (5.0), 172.0 and 168.0 (4.9),
    146.25 (4.8), 165.25 (4.5), 170.5 (4.4), 193.5 (4.3) in chi^2 over no oscillation. Studies:
      om82  [79.5, 83.5], A_osc to 0.09 (PDP+'s nodes at omega 61-85, A_osc >= 0.098 were at the region's edge);
      om144 [139.5, 148.5], om158 [155.5, 160.5], om169 [163.5, 174.5], om195 [191.5, 197.5], A_osc to 0.1.
    A maximum moves along a ridge in omega and phi (PDP+'s 179 from 178 at phi +0.8 to 184 at phi -2.5), hence windows
    of 4 to 11. Below omega = 55, where there is no scan, the joined surface is checked against the other two planes'
    true optima once those are done, and a study is added where it fails.
  phi over H0WC's own window of the training box, [-2.66339265359, 3.61979265359] (PDP+'s is shifted by 0.0558).

H0WC's boost points reach the same A_osc as PDP+'s below omega = 100 (at most 0.068 and 0.064; PDP+ 0.074 and 0.064)
and the box's 0.1 above. A study whose scan puts a node optimum at the region's edge stops after the scan, before any
true call (profiles/surface_workflow.py); its window is then narrowed from the scan's own optima, never from a projection.

    python make_profiles200_wc.py KIT
"""
import sys
from pathlib import Path

kit = Path(sys.argv[1])
PP, WC = kit / "runs/axion_pp_combined200", kit / "runs/axion_wc_combined200"
PHI = "[-2.66339265359, 3.61979265359]"
WINDOWS = [(1.0, 50.75, 0.06), (50.75, 100.5, 0.09), (100.5, 150.25, 0.1), (150.25, 200.0, 0.1)]
RANGES = {"om82": (79.5, 83.5, 0.09, "the maximum of the scan at omega 81.5"),
          "om144": (139.5, 148.5, 0.1, "the maxima of the scan at omega 141.5 and 146.25"),
          "om158": (155.5, 160.5, 0.1, "the highest maximum of the scan, at omega 157.75"),
          "om169": (163.5, 174.5, 0.1, "the maxima of the scan at omega 165.25, 168, 170.5 and 172"),
          "om195": (191.5, 197.5, 0.1, "the maxima of the scan at omega 193.5 and 195.5")}
LEGACY = "surrogate_optimizer: legacy      # the hybrid_* settings below are then not used\n"


def body(template, ranges, search=None):
    text = (PP / template).read_text()
    text = text[text.index("dimension: 2"):]
    start, end = text.index("ranges:\n"), text.index("workers:")
    text = text[:start] + "ranges:\n" + "".join(f"  {k}: {v}\n" for k, v in ranges.items()) + text[end:]
    if "search_ranges:" in text:
        text = text[:text.index("search_ranges:")]
    if search:
        text += f"search_ranges:              # omega searched only here (CLiENT's search_ranges)\n  omega: {search}\n"
    return text


def head(pair, what, nodes, path, study):
    return (f"# The true 2D profile of H0WC in {pair} for omega <= 200, from the combined emulator of the low part and the seven\n"
            "# zones at their final networks (results/<time>_axion_wc_combined200, network 0). At every node the third\n"
            "# oscillation parameter and all others are optimised with the true likelihood (runs/axion_wc_combined200/job.env).\n"
            f"# 7 November 2026, make_profiles200_wc.py in results/20261102_high_omega_ladder. {what}\n"
            "#\n"
            f"#   sbatch true_profiles/submit_profiles.sh results/<time>_axion_wc_combined200 runs/axion_wc_combined200/{path} results_profiles/{study} --iteration 0\n"
            "#\n"
            f"# 50 x 50 nodes: {nodes}\n"
            "# region: run (the combined region, one cell per part, edge 1.5). Settings as PDP+'s studies.\n"
            "#\n")


def fallback(path, study, text):
    fpath, fstudy = f"fallback/{Path(path).stem}_legacy.yaml", f"{study}_legacy"
    line = f"runs/axion_wc_combined200/{path} results_profiles/{study} "
    assert text.count(line) == 1 and text.count("surrogate_optimizer: hybrid\n") == 1
    text = text.replace(line, f"runs/axion_wc_combined200/{fpath} results_profiles/{fstudy} ").replace("surrogate_optimizer: hybrid\n", LEGACY)
    (WC / fpath).write_text(f"# The fallback of {path}: the same study with CLiENT's older scan, used only if its hybrid scan ends at\n"
                            "# its 20 hops with nodes still improving. Its own output directory.\n#\n" + text)
    return fpath


(WC / "fallback").mkdir(exist_ok=True)
(WC / "windows").mkdir(exist_ok=True)
written = []
for k, (lo, hi, amax) in enumerate(WINDOWS, 1):
    step = (hi - lo) / 49
    path, study = f"profiles_aosc_omega_w{k}.yaml", f"axion_wc200_aosc_omega_w{k}"
    text = head("(A_osc, omega)", f"Window {k} of 4 in omega, [{lo:g}, {hi:g}].",
                f"A_osc [0, {amax:g}] in steps of {amax / 49:.4f}, omega in steps of {step:.3f}.", path, study) \
        + body("profiles_aosc_omega_w3.yaml", {"A_osc": f"[0.0, {amax}]", "omega": f"[{lo}, {hi}]"})
    (WC / path).write_text(text)
    written += [path, fallback(path, study, text)]
    path, study = f"profiles_omega_phi_w{k}.yaml", f"axion_wc200_omega_phi_w{k}"
    (WC / path).write_text(head("(omega, phi)", f"Window {k} of 4 in omega, [{lo:g}, {hi:g}].",
                                f"omega in steps of {step:.3f}, phi in steps of 0.13 rad over the runs' window (its two ends are the same phase).",
                                path, study)
                           + body("profiles_omega_phi_w1.yaml", {"omega": f"[{lo}, {hi}]", "phi": PHI}))
    written.append(path)
path, study = "profiles_aosc_phi.yaml", "axion_wc200_aosc_phi"
(WC / path).write_text(head("(A_osc, phi)", "omega free over the prior [1, 200]; the older scan (surrogate_optimizer: legacy), as PDP+.",
                            "A_osc [0, 0.1] in steps of 0.0020, phi in steps of 0.13 rad.", path, study)
                       + body("profiles_aosc_phi.yaml", {"A_osc": "[0.0, 0.1]", "phi": PHI}))
written.append(path)
for tag, (lo, hi, amax, what) in RANGES.items():
    path, study = f"windows/profiles_aosc_phi_{tag}.yaml", f"axion_wc200_aosc_phi_{tag}"
    text = head("(A_osc, phi)", f"omega searched only in [{lo:g}, {hi:g}], {what} (CLiENT's search_ranges); the surface is the "
                "largest of the (A_osc, phi) studies' values at each node.",
                f"A_osc [0, {amax:g}] in steps of {amax / 49:.4f}, phi in steps of 0.13 rad"
                f"{' (the grid of profiles_aosc_phi.yaml)' if amax == 0.1 else ' (the phi grid of profiles_aosc_phi.yaml)'}.", path, study) \
        + body("profiles_aosc_phi_om180.yaml", {"A_osc": f"[0.0, {amax}]", "phi": PHI}, search=f"[{lo}, {hi}]")
    (WC / path).write_text(text)
    written += [path, fallback(path, study, text)]
print("\n".join(written))
