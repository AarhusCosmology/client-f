"""Specs for the kit's smoke/profile_chain.py: the World Cup profiles without SH0ES, with the cut, by the true likelihood
alone, node by node (27 October 2026):

    python wc_chain_setup.py [PROFILE ...]        # default: the twelve cosmological parameters

The refinement of the cut studies' nodes (..._dncut_{cosmo,fidm_wide,logGeff_wide}) started from the emulator and ended
mostly at its limit of 156 calls per node, and four passes of moving known points to the nodes (wc_polish_setup.py)
left several profiles jagged by 2-3 in chi^2. There are two branches: the best fit (Delta N_eff = 0.22, f_idm = 0.01,
log10 G_eff = -2.85, H0 = 69.5) and, only 0.7 above it in chi^2, a branch at the cut (Delta N_eff = 0.01, f_idm = 0.2,
log10 z_stop and log10 (G/H)_ini on their prior boundaries, log10 G_eff = -3.3, H0 = 68.4). For every profiled parameter p this writes the kit's
smoke/profile_chain_wc_<p>.json with the cut studies' nodes of p and four chains, one up and one down from each seed's
value of p:
    main     the best true point found outside the cut (the global best fit)
    branch   the best true point with Delta N_eff < 0.05 and f_idm > 0.15 (the region of dark acoustic oscillations)
Cobaya file: the cut's (smoke/cobaya_wc_dncut.yaml, Delta N_eff >= 0.01), whose log-posterior is the run's plus
build_wc.CUT_PRIOR_SHIFT; nuisance parameters as smoke/minimise_drmd_wc_cut.json; covariance of the twelve slow
parameters from the posterior samples in the best fit's mode (wc_polish_setup.mode_covariance), which whitens each
node's search and gives the shift along the degeneracy from node to node; reference_logpost, the best fit's, for
--stop-above. On Grendel, two jobs per profile (one per seed, two chains each, 46 of 48 workers):

    sbatch smoke/submit_profile_chain.sh smoke/profile_chain_wc_<p>.json wc_<p>_main --chains main_up main_down --stop-above 12
    sbatch smoke/submit_profile_chain.sh smoke/profile_chain_wc_<p>.json wc_<p>_branch --chains branch_up branch_down --stop-above 12

chain_results() reads the jobs' logs (logs/profile_chain_wc_*_<job>.json) back onto the run's objective.
"""
import json
import sys

import build_wc as W
import wc_polish_setup as S

KIT = W.KIT
SLOW = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "m_int", "logGeff", "delta_Neff_drmd", "f_idm_drmd",
        "log10_zstop", "log10_G_over_aH"]
COBAYA = "smoke/cobaya_wc_dncut.yaml"


def chain_results(names):
    """Every node of every chain job: (profile, coordinate, parameters, ln L on the run's objective)."""
    out = []
    for f in sorted(KIT.glob("logs/profile_chain_wc_*_[0-9]*.json")):
        r = json.loads(f.read_text())
        if r.get("cobaya") != COBAYA:
            continue
        for ch in r["chains"].values():
            for node in ch.get("nodes", []):
                b = node.get("best")
                if b is not None:
                    out.append((ch["profile"], float(node["coordinate"]), {n: float(b["params"][n]) for n in names},
                                float(b["logpost"]) - W.CUT_PRIOR_SHIFT))
    return out


def main():
    profiles = sys.argv[1:] or SLOW
    unknown = set(profiles) - set(SLOW)
    if unknown:
        raise SystemExit(f"not slow parameters: {sorted(unknown)}")
    names, _ = S.meta()
    src = S.sources(names, npass=len(S.PASSES) + 1)
    main_seed = max(src, key=lambda s: s[3])
    branch = [s for s in src if s[2]["delta_Neff_drmd"] < 0.05 and s[2]["f_idm_drmd"] > 0.15]
    branch_seed = max(branch, key=lambda s: s[3])
    coords = {}
    for p, c, *_ in S.new_nodes(names):
        coords.setdefault(p, set()).add(round(c, 6))
    C = S.mode_covariance(SLOW)
    old = json.loads((KIT / "smoke" / "minimise_drmd_wc_cut.json").read_text())
    print(f"seeds: main ln L {main_seed[3]:.3f} ({main_seed[4]}), branch ln L {branch_seed[3]:.3f} ({branch_seed[4]}), "
          f"{2 * (main_seed[3] - branch_seed[3]):.2f} above in chi^2")
    for p in profiles:
        chains = []
        for tag, seed in (("main", main_seed), ("branch", branch_seed)):
            v = seed[2][p]
            up = sorted(c for c in coords[p] if c > v + 1e-9)
            down = sorted((c for c in coords[p] if c <= v + 1e-9), reverse=True)
            for direction, nodes in (("up", up), ("down", down)):
                if nodes:
                    chains.append(dict(label=f"{tag}_{direction}", profile=p, seed=dict(seed[2]), nodes=nodes,
                                       note=f"{tag} seed: {seed[4]} (ln L {seed[3]:.3f} on the run's objective, {p} = {v:.6g})"))
        spec = dict(made="wc_chain_setup.py", cobaya=COBAYA, job_env=old["job_env"], fixed={}, nuisance=old["nuisance"],
                    covariance=dict(names=SLOW, matrix=C.tolist()), reference_logpost=main_seed[3] + W.CUT_PRIOR_SHIFT,
                    chains=chains)
        out = KIT / "smoke" / f"profile_chain_wc_{p}.json"
        out.write_text(json.dumps(spec, indent=1))
        print(f"{out.relative_to(KIT)}: " + ", ".join(f"{c['label']} {len(c['nodes'])}" for c in chains))


if __name__ == "__main__":
    main()
