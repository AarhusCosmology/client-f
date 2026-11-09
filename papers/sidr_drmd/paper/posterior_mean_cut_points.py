"""The Pantheon+ posterior mean with the cut at Delta N_eff = 0.01, for the kit's chi^2 job (26 October 2026):

    python posterior_mean_cut_points.py

Sec. 4.1 quotes the true chi^2 at the posterior mean (over all 21 sampled parameters). With the cut (build.py CUT) the
mean moves (Delta N_eff 0.33 -> 0.38, f_idm 0.040 -> 0.020), so it is evaluated again. Pantheon+SH0ES has no posterior
below the cut and keeps its point. Writes the kit's smoke/chi2_points_posterior_mean_cut.json (set pp_posterior_mean_cut):

    sbatch smoke/submit_chi2_points.sh smoke/chi2_points_posterior_mean_cut.json
"""
import json

import numpy as np

import build as B


def main():
    sets = {}
    for d in ("pp",):
        run = B.latest(f"results/*_sidr_drmd2_{d}_combined_boost10")
        z = np.load(run / "analysis" / "best_posterior_it_0.npz", allow_pickle=True)
        names = [str(n) for n in z["names"]]
        x, w = z["x"].astype(float), z["weights"].astype(float)
        keep = x[:, names.index("delta_Neff_drmd")] >= B.CUT
        mean = (x[keep] * w[keep, None]).sum(axis=0) / w[keep].sum()
        params = dict(zip(names, map(float, mean)))
        sets[f"{d}_posterior_mean_cut"] = dict(cobaya=f"runs/sidr_drmd2_{d}_combined_boost10/cobaya.yaml",
                                               job_env=f"runs/sidr_drmd2_{d}_combined_boost10/job.env",
                                               points=[dict(label="posterior_mean", params=params,
                                                            note=f"mean of {run.name} with Delta N_eff >= {B.CUT}")])
        print(d, {k: round(params[k], 4) for k in ("H0", "delta_Neff_drmd", "f_idm_drmd", "logGeff")})
    path = B.KIT / "smoke" / "chi2_points_posterior_mean_cut.json"
    path.write_text(json.dumps(dict(made="posterior_mean_cut_points.py", sets=sets), indent=1))
    print("written:", path)


if __name__ == "__main__":
    main()
