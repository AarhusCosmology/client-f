"""The 2D profile studies of the PDP+ combined emulator for omega <= 200 (6 November 2026; Steen chose four omega
windows): runs/axion_pp_combined200/profiles_{aosc_omega,omega_phi}_w{1..4}.yaml and profiles_aosc_phi.yaml.

A 2D study has at most 50 x 50 nodes. Over omega [1, 200] that is a step of 4 in omega, coarser than the peaks (the
main PDP+ peak spans 49-52.5), so each plane with omega is split into four windows of 50 nodes, a step of 1.015
(1.2 in the omega <= 60 study). The settings are those of the omega <= 60 studies (runs/axion_pp_combined/profiles_*.yaml),
with only the ranges changed: the hybrid scan for (A_osc, omega) and (omega, phi), the older scan for (A_osc, phi) (the
hybrid one did not converge there in 60 rounds).

A_osc reaches higher than 0.05 at high omega: within Delta chi^2 < 11.83 of the best training value, 0.094 in zone 4
and 0.063 in the low part. A 2D study stops after its scan if a node's optimum is outside the region, and the
region's projection onto (A_osc, omega) (the smallest rho of a point with the node's two values) reaches 1.54 at
A_osc = 0.1 in the lowest window (20 nodes beyond 0.98 x 1.5) but at most 1.26 at 0.08, and 1.25, 0.89 and 0.69 at 0.1 in
the other three. So A_osc runs to 0.08 in window 1 and to 0.1 (the box) in windows 2-4 and in (A_osc, phi), where it
reaches 0.55. (omega, phi) reaches at most 0.52. A projection is the most favourable point of a node, so a study can
still stop; it does so after the scan, before any true call.

Fallbacks (6 November 2026, Steen): runs/axion_pp_combined200/fallback/profiles_aosc_omega_w{1..4}_legacy.yaml, the same
four (A_osc, omega) studies with CLiENT's older scan (surrogate_optimizer: legacy), as runs/axion_wc_combined's
profiles_aosc_omega_legacy.yaml for omega <= 60: for a window whose hybrid scan ends at its 20 hops with nodes still
improving (at hop 6 of the first submission 37-188 of 2501 nodes still improved per hop). In a subfolder so that
analysis/queue_axion_combined200.sh, which submits runs/<NAME>/profiles_*.yaml, never submits them.

    python make_profiles200.py KIT
"""
import sys
from pathlib import Path

kit = Path(sys.argv[1])
OLD, NEW = kit / "runs/axion_pp_combined", kit / "runs/axion_pp_combined200"
PHI = "[-2.71919265359, 3.56399265359]"
WINDOWS = [(1.0, 50.75, 0.08), (50.75, 100.5, 0.1), (100.5, 150.25, 0.1), (150.25, 200.0, 0.1)]


def body(old, ranges):
    text = (OLD / old).read_text()
    text = text[text.index("dimension: 2"):]
    start = text.index("ranges:\n")
    end = text.index("workers:")
    return text[:start] + "ranges:\n" + "".join(f"  {k}: {v}\n" for k, v in ranges.items()) + text[end:]


def head(pair, what, nodes, name):
    out = f"results_profiles/axion_pp200_{name[len('profiles_'):-len('.yaml')]}"
    return (f"# The true 2D profile of PDP+ in {pair} for omega <= 200, from the combined emulator of the low part and the five\n"
            "# zones at their final networks (results/<time>_axion_pp_combined200, network 0). At every node the third\n"
            "# oscillation parameter and all others are optimised with the true likelihood (CLASS++ 0a57bbc6,\n"
            f"# runs/axion_pp_combined200/job.env). 6 November 2026, make_profiles200.py in results/20261102_high_omega_ladder.\n"
            f"# {what}\n"
            "#\n"
            f"#   sbatch true_profiles/submit_profiles.sh results/<time>_axion_pp_combined200 runs/axion_pp_combined200/{name} {out} --iteration 0\n"
            "#\n"
            f"# 50 x 50 nodes: {nodes}\n"
            "# region: run (the combined region, one cell per part, edge 1.5). Settings as runs/axion_pp_combined's studies.\n"
            "#\n")


written = []
for k, (lo, hi, amax) in enumerate(WINDOWS, 1):
    step = (hi - lo) / 49
    name = f"profiles_aosc_omega_w{k}.yaml"
    (NEW / name).write_text(head("(A_osc, omega)", f"Window {k} of 4 in omega, [{lo:g}, {hi:g}].", f"A_osc [0, {amax:g}] in steps of {amax / 49:.4f}, omega in steps of {step:.3f}.", name)
                            + body("profiles_aosc_omega.yaml", {"A_osc": f"[0.0, {amax}]", "omega": f"[{lo}, {hi}]"}))
    name2 = f"profiles_omega_phi_w{k}.yaml"
    (NEW / name2).write_text(head("(omega, phi)", f"Window {k} of 4 in omega, [{lo:g}, {hi:g}].", f"omega in steps of {step:.3f}, phi in steps of 0.13 rad over the runs' window (its two ends are the same phase).", name2)
                             + body("profiles_omega_phi.yaml", {"omega": f"[{lo}, {hi}]", "phi": PHI}))
    written += [name, name2]
name = "profiles_aosc_phi.yaml"
(NEW / name).write_text(head("(A_osc, phi)", "omega free over the prior [1, 200]; the older scan (surrogate_optimizer: legacy), as for omega <= 60.",
                             "A_osc [0, 0.1] in steps of 0.0020, phi in steps of 0.13 rad.", name)
                        + body("profiles_aosc_phi_legacy.yaml", {"A_osc": "[0.0, 0.1]", "phi": PHI}))
written.append(name)
FALL = NEW / "fallback"
FALL.mkdir(exist_ok=True)
HYBRID = "surrogate_optimizer: hybrid\n"
for k, (lo, hi, amax) in enumerate(WINDOWS, 1):
    src = (NEW / f"profiles_aosc_omega_w{k}.yaml").read_text()
    assert src.count(HYBRID) == 1
    name = f"profiles_aosc_omega_w{k}_legacy.yaml"
    out = f"results_profiles/axion_pp200_aosc_omega_w{k}_legacy"
    note = (f"# The fallback of profiles_aosc_omega_w{k}.yaml (6 November 2026): the same study with CLiENT's older scan, used only if\n"
            "# the hybrid scan ends at its 20 hops with nodes still improving (make_profiles200.py). Its own output directory:\n"
            "#\n"
            f"#   sbatch true_profiles/submit_profiles.sh results/<time>_axion_pp_combined200 runs/axion_pp_combined200/fallback/{name} {out} --iteration 0\n"
            "#\n")
    body = src.replace(HYBRID, "surrogate_optimizer: legacy      # the hybrid_* settings below are then not used\n")
    body = body.replace(f"#   sbatch true_profiles/submit_profiles.sh results/<time>_axion_pp_combined200 runs/axion_pp_combined200/profiles_aosc_omega_w{k}.yaml results_profiles/axion_pp200_aosc_omega_w{k} --iteration 0\n#\n", "")
    (FALL / name).write_text(note + body)
    written.append("fallback/" + name)
print("\n".join(written))
