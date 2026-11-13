"""Windows 1 and 2 of the PDP+ (A_osc, omega) profiles for omega <= 200 with A_osc ranges inside the region (6 November
2026, evening): runs/axion_pp_combined200/profiles_aosc_omega_w{1,2}_a{060,090}.yaml and their fallbacks with the older
scan in fallback/.

make_profiles200.py chose the A_osc ranges (0.08 in window 1, 0.1 in window 2) from the region's projection onto
(A_osc, omega), the most favourable point of a node. The scans' own optima are less favourable. The fallback scans
(results_profiles/axion_pp200_aosc_omega_w{1,2}_legacy, five passes each, snapshot of 6 November 16:41) put the optimum
of 57 nodes of window 1 (A_osc >= 0.0718, omega 2-17) and 31 of window 2 (A_osc >= 0.098, omega 61-85) at the region's
edge (rho > 0.98 x 1.5), and the hybrid scan of window 2 put 30 there. A 2D study stops after its scan when any node is
there (profiles/surface_workflow.py), so neither fallback could reach its true stage; they also did not settle, since
the nodes at and next to the edge stayed nonstationary (projected gradient 1.4-1.8 against 1e-4) at an unchanged value.
These nodes lie 126-411 (window 1) and 96-146 (window 2) below the best fit in chi^2.

From the same scans:
  window 1, A_osc <= 0.06: every optimum at rho <= 1.33 (0.89 of the edge, below the 0.9-0.98 where nodes stall);
    what is lost is the column omega = 1 (the prior's end) between A_osc 0.06 and 0.078, at 7.7-11.8 below the best fit
    in chi^2 (the low-frequency degeneracy with n_s and the running), which (A_osc, phi), with omega free, still shows;
  window 2, A_osc <= 0.09: every optimum at rho <= 1.20; the top row is 32 below the best fit in chi^2.
The other settings, the grids in omega and the hybrid scan are those of make_profiles200.py.

    python make_profiles200_narrow.py KIT
"""
import sys
from pathlib import Path

kit = Path(sys.argv[1])
NEW = kit / "runs/axion_pp_combined200"
NARROW = {1: (0.08, 0.06, "a060"), 2: (0.1, 0.09, "a090")}
NOTE = ("# A_osc narrowed from {old:g} to {new:g} (6 November 2026, make_profiles200_narrow.py): with {old:g} the scans put the optimum\n"
        "# of {edge} nodes at the region's edge, and the study would stop after its scan. Replaces profiles_aosc_omega_w{k}.yaml.\n")
written = []
for k, (old, new, tag) in NARROW.items():
    src = (NEW / f"profiles_aosc_omega_w{k}.yaml").read_text()
    name, study = f"profiles_aosc_omega_w{k}_{tag}.yaml", f"axion_pp200_aosc_omega_w{k}_{tag}"
    reps = [(f"runs/axion_pp_combined200/profiles_aosc_omega_w{k}.yaml results_profiles/axion_pp200_aosc_omega_w{k} ",
             f"runs/axion_pp_combined200/{name} results_profiles/{study} "),
            (f"A_osc [0, {old:g}] in steps of {old / 49:.4f},", f"A_osc [0, {new:g}] in steps of {new / 49:.4f},"),
            (f"  A_osc: [0.0, {old}]\n", f"  A_osc: [0.0, {new}]\n")]
    text = src
    for a, b in reps:
        assert text.count(a) == 1, (k, a)
        text = text.replace(a, b)
    text = NOTE.format(old=old, new=new, edge={1: 57, 2: 31}[k], k=k) + "#\n" + text
    (NEW / name).write_text(text)
    written.append(name)
    hybrid = "surrogate_optimizer: hybrid\n"
    assert text.count(hybrid) == 1
    fname, fstudy = f"profiles_aosc_omega_w{k}_{tag}_legacy.yaml", f"axion_pp200_aosc_omega_w{k}_{tag}_legacy"
    fb = text.replace(hybrid, "surrogate_optimizer: legacy      # the hybrid_* settings below are then not used\n")
    line = f"#   sbatch true_profiles/submit_profiles.sh results/<time>_axion_pp_combined200 runs/axion_pp_combined200/{name} results_profiles/{study} --iteration 0\n"
    assert fb.count(line) == 1
    fb = fb.replace(line, line.replace(f"combined200/{name}", f"combined200/fallback/{fname}").replace(study, fstudy))
    fb = (f"# The fallback of {name}: the same study with CLiENT's older scan, used only if its hybrid scan ends at its 20\n"
          "# hops with nodes still improving. Its own output directory.\n#\n") + fb
    (NEW / "fallback" / fname).write_text(fb)
    written.append("fallback/" + fname)
print("\n".join(written))
