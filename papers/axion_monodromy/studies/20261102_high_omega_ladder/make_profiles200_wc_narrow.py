"""Windows 1 and 2 of the H0WC (A_osc, omega) profiles for omega <= 200 with A_osc ranges inside the region (8 November
2026): runs/axion_wc_combined200/profiles_aosc_omega_w{1,2}_a{040,070}.yaml and their fallbacks with the older scan in
fallback/.

The hybrid scans of make_profiles200_wc.py (A_osc to 0.06 and 0.09, the ranges that passed for PDP+) did not pass: in window
1 (omega [1, 50.75]) 44 node optima lay at the region's edge (rho 1.50 against 0.98 x 1.5) at A_osc >= 0.047, omega 3-6,
270-540 below the window's best in chi^2, and 63 nodes were nonstationary after the 20 hops (13 still improving); in window 2
(omega [50.75, 100.5]) 64 optima lay beyond the edge and 17 nodes at A_osc >= 0.0735, omega 50-62, 90-150 below the best,
did not settle. A 2D study stops after its scan when any node is at the edge (profiles/surface_workflow.py).

From the same scans (failed_surrogate_diagnostics.json, the scans' own optima):
  window 1, A_osc <= 0.04: every optimum at rho <= 1.28 (0.87 of the edge), all stationary; what is lost is the column
    omega = 1 (the prior's end) between A_osc 0.04 and 0.06, 2.6-3.4 below the window's best in chi^2 (the low-frequency
    degeneracy with n_s and the running), which (A_osc, phi), with omega free, shows;
  window 2, A_osc <= 0.07: every optimum at rho <= 1.15, all stationary; nothing above lies within 19.5 of the best.
The other settings are those of make_profiles200_wc.py.

    python make_profiles200_wc_narrow.py KIT
"""
import sys
from pathlib import Path

kit = Path(sys.argv[1])
NEW = kit / "runs/axion_wc_combined200"
NARROW = {1: (0.06, 0.04, "a040", 44), 2: (0.09, 0.07, "a070", 64)}
NOTE = ("# A_osc narrowed from {old:g} to {new:g} (8 November 2026, make_profiles200_wc_narrow.py): with {old:g} the scan put the\n"
        "# optimum of {edge} nodes beyond the region's edge, and the study stopped after its scan. Replaces profiles_aosc_omega_w{k}.yaml.\n")
written = []
for k, (old, new, tag, edge) in NARROW.items():
    src = (NEW / f"profiles_aosc_omega_w{k}.yaml").read_text()
    name, study = f"profiles_aosc_omega_w{k}_{tag}.yaml", f"axion_wc200_aosc_omega_w{k}_{tag}"
    reps = [(f"runs/axion_wc_combined200/profiles_aosc_omega_w{k}.yaml results_profiles/axion_wc200_aosc_omega_w{k} ",
             f"runs/axion_wc_combined200/{name} results_profiles/{study} "),
            (f"A_osc [0, {old:g}] in steps of {old / 49:.4f},", f"A_osc [0, {new:g}] in steps of {new / 49:.4f},"),
            (f"  A_osc: [0.0, {old}]\n", f"  A_osc: [0.0, {new}]\n")]
    text = src
    for a, b in reps:
        assert text.count(a) == 1, (k, a)
        text = text.replace(a, b)
    text = NOTE.format(old=old, new=new, edge=edge, k=k) + "#\n" + text
    (NEW / name).write_text(text)
    written.append(name)
    hybrid = "surrogate_optimizer: hybrid\n"
    assert text.count(hybrid) == 1
    fname, fstudy = f"profiles_aosc_omega_w{k}_{tag}_legacy.yaml", f"axion_wc200_aosc_omega_w{k}_{tag}_legacy"
    fb = text.replace(hybrid, "surrogate_optimizer: legacy      # the hybrid_* settings below are then not used\n")
    line = f"#   sbatch true_profiles/submit_profiles.sh results/<time>_axion_wc_combined200 runs/axion_wc_combined200/{name} results_profiles/{study} --iteration 0\n"
    assert fb.count(line) == 1
    fb = fb.replace(line, line.replace(f"combined200/{name}", f"combined200/fallback/{fname}").replace(study, fstudy))
    fb = (f"# The fallback of {name}: the same study with CLiENT's older scan, used only if its hybrid scan ends at its 20\n"
          "# hops with nodes still improving. Its own output directory.\n#\n") + fb
    (NEW / "fallback" / fname).write_text(fb)
    written.append("fallback/" + fname)
print("\n".join(written))
