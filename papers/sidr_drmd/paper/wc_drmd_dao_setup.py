"""A true-likelihood minimisation of DRMD on the World Cup data without SH0ES, started in the region of dark acoustic
oscillations at the cut (27 October 2026):

    python wc_drmd_dao_setup.py

After the second polish pass (wc_polish_setup.py) the profile of log10 G_eff with the cut reaches Delta chi^2 = 3.5 at
log10 G_eff = -5.5, i.e. in DRMD, at Delta N_eff = 0.01 (the cut), f_idm = 0.31 and log10 z_stop = 5 (the prior's upper
bound). That point is 2.0 lower in chi^2 than the true minimum of LCDM, whereas the DRMD minimisations above the cut of
wc_region_minimise_setup.py (kit jobs 6962431 and 6962595), started near LCDM, ended at LCDM. This writes the kit's
smoke/minimise_drmd_wc_cutdao.json for smoke/minimise_true.py, with the same Cobaya file (the cut's), fixed log10 G_eff,
nuisance parameters and whitening as smoke/minimise_drmd_wc_cut.json, from two starts:
    dao_node    the best true point with log10 G_eff <= -5.4 (a node of the log10 G_eff profile)
    dao_moved   the best true point with -5.4 < log10 G_eff < -4, f_idm > 0.15 and log10 z_stop > 4 (2.3 above the best
                fit, at log10 G_eff = -4.04), moved to log10 G_eff = -5.5
Both lie at three boundaries: Delta N_eff = 0.01, log10 z_stop = 5 and log10 (G/H)_ini = 2.
Its result is read by build_wc.table_minimisations (logs/minimise_drmd_wc_cutdao_<job>.json). On Grendel:

    sbatch smoke/submit_minimise.sh smoke/minimise_drmd_wc_cutdao.json drmd_wc_cutdao
"""
import json

import build_wc as W
import wc_polish_setup as S
import wc_region_minimise_setup as R

KIT = W.KIT


def main():
    names, _ = S.meta()
    src = S.sources(names, npass=len(S.PASSES) + 1)
    weak = sorted((s for s in src if s[2]["logGeff"] < -4 and s[2]["f_idm_drmd"] > 0.15 and s[2]["log10_zstop"] > 4),
                  key=lambda s: -s[3])
    node = next(s for s in weak if s[2]["logGeff"] <= -5.4)
    other = next(s for s in weak if s[2]["logGeff"] > -5.4)

    def start(label, s, **moved):
        q = dict({n: float(s[2][n]) for n in names}, **moved)
        assert q["delta_Neff_drmd"] >= R.CUT, (label, q["delta_Neff_drmd"])
        return dict(label=label, params=q, note=f"{s[4]} (true ln L {s[3]:.2f})" + (f", moved to {moved}" if moved else ""))

    starts = [start("dao_node", node, logGeff=-5.5), start("dao_moved", other, logGeff=-5.5)]
    old = json.loads((KIT / "smoke" / "minimise_drmd_wc_cut.json").read_text())
    spec = dict(old, made="wc_drmd_dao_setup.py", starts=starts)
    out = KIT / "smoke" / "minimise_drmd_wc_cutdao.json"
    out.write_text(json.dumps(spec, indent=1))
    lcdm = max(b["logpost"] for _, _, b in W.table_minimisations("wc", "lcdm"))
    print(f"{out.relative_to(KIT)} (cobaya {spec['cobaya']}, fixed {spec['fixed']}):")
    for x, s in zip(starts, (node, other)):
        q = x["params"]
        print(f"   {x['label']:10s} dNeff {q['delta_Neff_drmd']:.4f} f_idm {q['f_idm_drmd']:.3f} log10 zstop {q['log10_zstop']:.2f} "
              f"H0 {q['H0']:.2f}; chi^2 {2 * (lcdm - s[3]):+.2f} relative to LCDM <- {x['note']}")


if __name__ == "__main__":
    main()
