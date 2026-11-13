"""The PDP+ (A_osc, phi) profile for omega <= 200 in four studies with omega searched in a narrow range each
(7 November 2026): runs/axion_pp_combined200/profiles_aosc_phi_om{18,30,180,196}.yaml and their fallbacks with the
older scan in fallback/. They need CLiENT's profile setting search_ranges (branch profile-search-ranges).

The study of make_profiles200.py (profiles_aosc_phi.yaml, omega free over [1, 200], the older scan) did not find the
highest maximum in omega at many nodes. Checked against the true optima of the (A_osc, omega) and (omega, phi) planes
(snapshot of 7 November 07:43, every node of the three planes; each other plane's optimum compared with the (A_osc, phi)
node within half a grid step of it): 523 of them, within Delta chi^2 < 11.83 of the best fit, beat the (A_osc, phi)
node by more than 1 in chi^2, 260 by more than 4, at most 23.4. They lie at omega 17-22 and 29-31 (A_osc <= 0.017,
up to 3.6) and at 178-184 and 194-200 (A_osc 0.008-0.1, up to 23). The two other planes pass the same test (at most
0.8). The network has these maxima higher too (at A_osc = 0.049: -6178.35 at omega 195.9, against -6180.82 at the
study's omega 100.7), so it is the search: the older scan continues each node from its neighbours, and the hybrid
scan's walkers all start at the best point of the box and step locally in omega.

Each study searches omega only in a range around one group of those maxima (search_ranges), with the hybrid scan:
  om18   omega [16.5, 23.5], A_osc [0, 0.04]: the maxima at 18 and 21;
  om30   omega [27.5, 32.5], A_osc [0, 0.04]: the maximum at 30;
  om180  omega [176.5, 185.5], A_osc [0, 0.1]: the ridge from 178 (phi +0.8) to 184 (phi -2.5);
  om196  omega [192.5, 200], A_osc [0, 0.1]: 194-197 (phi +1.6 to +0.7) and the prior's end at 200 (phi -1.5).
The two high studies have the grid of profiles_aosc_phi.yaml, node for node; the surface is the largest of the five
studies' values at each node. The two low ones end at A_osc = 0.04 (their own grid, joined by interpolation in A_osc):
their maxima are 12-100 below the best fit in chi^2 above A_osc = 0.03 ((A_osc, omega) plane), and the region does not
allow more (the window-1 scan's optima at omega 16-24 reach rho 1.5 at A_osc = 0.08 against the study's edge 1.47;
0.8-0.99 at 0.034-0.046). In the two high ranges the optima reach rho 0.58.

    python make_profiles200_aosc_phi_windows.py KIT
"""
import sys
from pathlib import Path

kit = Path(sys.argv[1])
NEW = kit / "runs/axion_pp_combined200"
WINDOWS = {"om18": (16.5, 23.5, 0.04, "the maxima at omega 18 and 21"),
           "om30": (27.5, 32.5, 0.04, "the maximum at omega 30"),
           "om180": (176.5, 185.5, 0.1, "the ridge of the maximum at omega 179 (178 at phi +0.8 to 184 at phi -2.5)"),
           "om196": (192.5, 200.0, 0.1, "the maximum at omega 196 (194-197) and the prior's end at 200")}
src = (NEW / "profiles_aosc_phi.yaml").read_text()
body = src[src.index("dimension: 2"):]
LEGACY = "surrogate_optimizer: legacy      # see the header; the hybrid_* settings below are then not used\n"
HOPS = "hybrid_hops: 60             # hops of the network scan at most (20 were not enough: see the header)\n"
assert body.count(LEGACY) == 1 and body.count(HOPS) == 1 and body.count("  A_osc: [0.0, 0.1]\n") == 1
written = []
for tag, (lo, hi, amax, what) in WINDOWS.items():
    name, study = f"profiles_aosc_phi_{tag}.yaml", f"axion_pp200_aosc_phi_{tag}"
    head = ("# The true 2D profile of PDP+ in (A_osc, phi) for omega <= 200 with omega searched in a narrow range, from the\n"
            "# combined emulator of the low part and the five zones at their final networks (results/<time>_axion_pp_combined200,\n"
            "# network 0), with the true likelihood (CLASS++ 0a57bbc6, runs/axion_pp_combined200/job.env). 7 November 2026,\n"
            "# make_profiles200_aosc_phi_windows.py in results/20261102_high_omega_ladder: profiles_aosc_phi.yaml (omega free\n"
            f"# over [1, 200]) missed the maxima of other omega ranges at many nodes. This one: omega [{lo:g}, {hi:g}], {what}.\n"
            "# Needs CLiENT's search_ranges. The surface is the largest of the studies' values at each node.\n"
            "#\n"
            f"#   sbatch true_profiles/submit_profiles.sh results/<time>_axion_pp_combined200 runs/axion_pp_combined200/{name} results_profiles/{study} --iteration 0\n"
            "#\n"
            f"# 50 x 50 nodes: A_osc [0, {amax:g}] in steps of {amax / 49:.4f}, phi in steps of 0.13 rad (the grid of profiles_aosc_phi.yaml"
            f"{'' if amax == 0.1 else ' in phi'}).\n"
            "# region: run (the combined region, one cell per part, edge 1.5).\n"
            "#\n")
    text = (body.replace("  A_osc: [0.0, 0.1]\n", f"  A_osc: [0.0, {amax}]\n")
                .replace(LEGACY, "surrogate_optimizer: hybrid\n")
                .replace(HOPS, "hybrid_hops: 20             # hops of the network scan at most (CLiENT's default 12), as the other planes\n"))
    text += f"search_ranges:              # omega searched only here (CLiENT's search_ranges)\n  omega: [{lo}, {hi}]\n"
    (NEW / name).write_text(head + text)
    written.append(name)
    fname, fstudy = f"profiles_aosc_phi_{tag}_legacy.yaml", f"axion_pp200_aosc_phi_{tag}_legacy"
    line = f"#   sbatch true_profiles/submit_profiles.sh results/<time>_axion_pp_combined200 runs/axion_pp_combined200/{name} results_profiles/{study} --iteration 0\n"
    fb = (head + text).replace(line, line.replace(f"combined200/{name}", f"combined200/fallback/{fname}").replace(study, fstudy))
    fb = fb.replace("surrogate_optimizer: hybrid\n", LEGACY.replace("see the header; ", ""))
    fb = (f"# The fallback of {name}: the same study with CLiENT's older scan, used only if its hybrid scan ends at its 20\n"
          "# hops with nodes still improving. Its own output directory.\n#\n") + fb
    (NEW / "fallback" / fname).write_text(fb)
    written.append("fallback/" + fname)
print("\n".join(written))
