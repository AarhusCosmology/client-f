"""Points and Cobaya set-ups for two per-likelihood chi^2 jobs on Grendel (24 October 2026, Steen):

    python results/20261024_chi2_compare/make_points.py

1. Why does the preferred self-interaction move from the strong mode (Planck alone: log10 Geff ~ -1.7) to intermediate
   coupling (~ -2.9) on the H0 World Cup data? Data or code? Four points per data set, each evaluated with every
   likelihood in four set-ups:
     A  new_best     the best true value of the new runs (both parts)
     B  new_strong   the best true value of the new runs with log10 Geff in [-2.0, -1.4)
     C  old_best     the global best fit of the earlier runs (their best profile node, the paper's star)
     D  old_interm   the earlier runs' log10 Geff profile node closest to -2.9 (their best at that coupling)
   set-ups:
     <wc>_new        World Cup data, CLASS++ bfe6729e, the new runs' settings (runs/sidr_drmd_<wc>_low)
     <pp>_newcode    the earlier data, CLASS++ bfe6729e, the new runs' theory block (runs/chi2cmp_<pp>_newcode)
     <pp>_newbuild   the earlier data and settings, CLASS++ bfe6729e (runs/chi2cmp_<pp>_newbuild)
     <pp>_old        the earlier data, settings and patched CLASS++ 7cc41a51 (runs/sidr_drmd2_<pp>_combined_boost10)
   Old points on the new data get the ACT and SPT calibrations of A, and, with SH0ES, M_B moved with H0 so that the
   supernova magnitudes stay those of A: M_B = M_B(A) + 5 log10(H0 / H0(A)). New points on the old data get the
   CamSpec nuisances of C (theirs were fitted to the cut CamSpec and are free above l = 1000); the ACT, SPT and M_B
   parameters the old files do not have are ignored. Nuisances are not re-optimised anywhere.

2. Which likelihoods drive the true profiles of the paper? Every node of the earlier runs' profile studies (12
   cosmological parameters x 25 nodes per data set, and the SH0ES f_idm profile over the whole prior), evaluated
   likelihood by likelihood with the runs' own set-up. The total must reproduce each node's logl.

3. Follow-ups (smoke/chi2_followup_points.json, same day):
   <pp>_cut_logGeff  the earlier log10 Geff profile nodes with CamSpec cut as in the World Cup data, everything else
                     as the earlier runs (runs/chi2cmp_<pp>_camspeccut): does the cut alone remove the strong mode's
                     lead, which came from CamSpec (+3.8 / +3.1 at the weak plateau)?
   <pp>_nuisopt      points A-D on the full earlier data and set-up, CamSpec's nine nuisances re-optimised at each
                     point: is the new best fit A really ~230 worse in full-l Planck than the earlier best fit C?

Writes, in the kit: smoke/chi2_compare_points.json, smoke/chi2_profile_points.json, smoke/chi2_followup_points.json,
runs/chi2cmp_<pp>_{newcode,newbuild,camspeccut}/.
"""
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1] / "results" / "20261007_grendel_kit" / "hpc_client"
PAPER = HERE.parents[1] / "results" / "20261022_preprint_client"
PAIRS = {"pp": "wc", "ppshoes": "wcmb"}            # earlier data set -> World Cup data set
STRONG = (-2.0, -1.4)
INTERMEDIATE = -2.9
CAMSPEC = ["A_planck", "amp_143", "amp_217", "amp_143x217", "n_143", "n_217", "n_143x217", "calTE", "calEE"]
CAMSPEC_CUT = """  planck_NPIPE_highl_CamSpec.TTTEEE:     # cut as in the H0 World Cup data (runs/sidr_drmd_wc_low/cobaya.yaml)
    dataset_params:
      use_cl: 143x143 217x217 143x217 TE EE
      use_range:
        143x143: 30-999
        217x217: 500-999
        143x217: 500-999
        TE: 30-599
        EE: 30-599
"""


def latest_training(run_glob):
    """The latest training set of each part of a new run, together, finite values only."""
    frames = []
    for run in sorted(KIT.glob(f"results/*_{run_glob}")):
        files = sorted(run.glob("training_data/data_it_*.csv"), key=lambda f: int(re.findall(r"\d+", f.name)[-1]))
        d = pd.read_csv(files[-1])
        d["source"] = f"{run.name}/{files[-1].name}"
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    v = pd.to_numeric(d["loglkl"], errors="coerce")
    return d[np.isfinite(v) & (v > -1e29)].assign(loglkl=v)


def row_point(row, names, label, note):
    return dict(label=label, note=note, params={n: float(row[n]) for n in names}, logl=float(row["loglkl"]))


def old_names():
    params = json.loads(next(KIT.glob("results/*_sidr_drmd2_pp_combined_boost10/metadata.json")).read_text())["parameters"]
    return [q["name"] if isinstance(q, dict) else q for q in params]


def node_point(path, names, label):
    j = json.loads(Path(path).read_text())
    return dict(label=label, params=dict(zip(names, map(float, j["x"]))), logl=float(j["logl"]),
                coordinate=float(j["coordinate"]))


def cobaya_newcode(old_file, new_file):
    """The earlier Cobaya file with the theory block of the new runs (CLASS++ settings), as text."""
    old, new = old_file.read_text(), new_file.read_text()
    theory = new[new.index("\ntheory:\n") + 1:new.index("\nparams:\n") + 1]     # line starts: "dataset_params:" is not one
    head = ("# chi^2 comparison (results/20261024_chi2_compare/make_points.py, 24 October 2026): the earlier data set of\n"
            f"# {old_file.relative_to(KIT)} with the theory block of {new_file.relative_to(KIT)} (CLASS++ bfe6729e settings).\n")
    return head + old[:old.index("\ntheory:\n") + 1] + theory + old[old.index("\nparams:\n") + 1:]


def main():
    names_old = old_names()
    best_fits = json.loads((PAPER / "plots" / "data" / "best_fits.json").read_text())
    compare = dict(made="24 October 2026, results/20261024_chi2_compare/make_points.py", sets={})
    profiles = dict(made=compare["made"], sets={})
    followup = dict(made=compare["made"], sets={})
    for pp, wc in PAIRS.items():
        new_file = KIT / "runs" / f"sidr_drmd_{wc}_low" / "cobaya.yaml"
        names_new = [k for k, v in yaml.safe_load(new_file.read_text())["params"].items()
                     if isinstance(v, dict) and v.get("prior") is not None]
        d = latest_training(f"sidr_drmd_{wc}_*")
        a = d.loc[d.loglkl.idxmax()]
        s = d[(d.logGeff >= STRONG[0]) & (d.logGeff < STRONG[1])]
        b = s.loc[s.loglkl.idxmax()]
        pa = row_point(a, names_new, "A_new_best", f"best true value of the new runs ({a.source})")
        pb = row_point(b, names_new, "B_new_strong", f"best with log10 Geff in [{STRONG[0]}, {STRONG[1]}) ({b.source})")
        bf = best_fits[pp]
        pc = dict(label="C_old_best", note=bf["source"], params={n: float(bf["values"][n]) for n in names_old},
                  logl=float(bf["logl"]))
        nodes = sorted(glob.glob(str(KIT / "logs" / f"profiles_{pp}_boost10_nodes" / "node_007_*.json")))
        coords = [json.loads(Path(f).read_text())["coordinate"] for f in nodes]
        pick = nodes[int(np.argmin(np.abs(np.array(coords) - INTERMEDIATE)))]
        pd_ = node_point(pick, names_old, "D_old_interm")
        pd_["note"] = f"log10 Geff profile node at {pd_['coordinate']:.3f} ({Path(pick).name})"

        # the old points on the new data: A's ACT and SPT calibrations, and M_B moved with H0
        extra = [n for n in names_new if n not in names_old]
        new_points = [pa, pb]
        for p in (pc, pd_):
            q = dict(p, params=dict(p["params"]))
            for n in extra:
                q["params"][n] = pa["params"][n]
            if "Mb" in extra:
                q["params"]["Mb"] = pa["params"]["Mb"] + 5 * math.log10(q["params"]["H0"] / pa["params"]["H0"])
            new_points.append(q)
        compare["sets"][f"{wc}_new"] = dict(cobaya=str(new_file.relative_to(KIT)),
                                            job_env=f"runs/sidr_drmd_{wc}_low/job.env", points=new_points)

        old_file = KIT / "runs" / f"sidr_drmd2_{pp}_combined_boost10" / "cobaya.yaml"
        # the new points on the earlier data: their cosmology, with the CamSpec nuisances of the earlier best fit C.
        # Their own were fitted to the cut CamSpec (l < 1000) and are free above it: with the full CamSpec they cost
        # thousands in chi^2 (laptop test of 24 October: +7200 at B).
        old_points = [dict(p, note=p["note"] + "; CamSpec nuisances of C",
                           params={n: (pc["params"][n] if n in CAMSPEC else p["params"][n]) for n in names_old})
                      for p in (pa, pb)] + [pc, pd_]
        for variant, text in (("newcode", cobaya_newcode(old_file, new_file)),
                              ("newbuild", "# chi^2 comparison: the earlier file unchanged, with CLASS++ bfe6729e (job.env)\n"
                                           + old_file.read_text())):
            folder = KIT / "runs" / f"chi2cmp_{pp}_{variant}"
            folder.mkdir(exist_ok=True)
            (folder / "cobaya.yaml").write_text(text)
            (folder / "job.env").write_text((KIT / "runs" / f"sidr_drmd_{wc}_low" / "job.env").read_text())
            compare["sets"][f"{pp}_{variant}"] = dict(cobaya=str((folder / "cobaya.yaml").relative_to(KIT)),
                                                      job_env=f"runs/chi2cmp_{pp}_{variant}/job.env", points=old_points)
        compare["sets"][f"{pp}_old"] = dict(cobaya=str(old_file.relative_to(KIT)),
                                            job_env=f"runs/sidr_drmd2_{pp}_combined_boost10/job.env", points=old_points)

        # every profile node of the earlier runs
        pts = []
        for f in sorted(glob.glob(str(KIT / "logs" / f"profiles_{pp}_boost10_nodes" / "node_*.json"))):
            i, k = map(int, re.findall(r"node_(\d+)_(\d+)", f)[0])
            p = node_point(f, names_old, f"{names_old[i]}:{k:02d}")
            p["profile"] = names_old[i]
            pts.append(p)
        if pp == "ppshoes":
            for f in sorted(glob.glob(str(KIT / "logs" / "profiles_ppshoes_fidm_wide" / "node_*.json"))):
                k = int(re.findall(r"node_\d+_(\d+)", f)[0])
                p = node_point(f, names_old, f"f_idm_drmd_wide:{k:02d}")
                p["profile"] = "f_idm_drmd_wide"
                pts.append(p)
        profiles["sets"][f"{pp}_profiles"] = dict(cobaya=str(old_file.relative_to(KIT)),
                                                  job_env=f"runs/sidr_drmd2_{pp}_combined_boost10/job.env", points=pts)

        # follow-up 1: the earlier log10 Geff profile nodes with CamSpec cut as in the World Cup data (l < 1000 in TT,
        # l < 600 in TE and EE), everything else as the earlier runs: does the cut alone remove the strong mode's lead?
        folder = KIT / "runs" / f"chi2cmp_{pp}_camspeccut"
        folder.mkdir(exist_ok=True)
        text = old_file.read_text()
        entry = "  planck_NPIPE_highl_CamSpec.TTTEEE: null\n"
        assert text.count(entry) == 1
        text = text.replace(entry, CAMSPEC_CUT)
        (folder / "cobaya.yaml").write_text("# chi^2 comparison: the earlier file with CamSpec cut as in the World Cup data "
                                            "(results/20261024_chi2_compare/make_points.py)\n" + text)
        (folder / "job.env").write_text((KIT / "runs" / f"sidr_drmd2_{pp}_combined_boost10" / "job.env").read_text())
        followup["sets"][f"{pp}_cut_logGeff"] = dict(cobaya=str((folder / "cobaya.yaml").relative_to(KIT)),
                                                     job_env=f"runs/chi2cmp_{pp}_camspeccut/job.env",
                                                     points=[q for q in pts if q["profile"] == "logGeff"])
        # the Pantheon+ omega_cdm profile computed again with twice the calls per node (its better values are the
        # paper's): its node checkpoints, when copied on Grendel into logs/profiles_pp_ocdm_maxfun_nodes/, fill the 10
        # nodes of the per-likelihood profile figure that have no breakdown yet (that study profiles omega_cdm alone)
        extra_nodes = KIT / "logs" / f"profiles_{pp}_ocdm_maxfun_nodes"
        if extra_nodes.is_dir():
            more = []
            for f in sorted(extra_nodes.glob("node_*.json")):
                k = int(re.findall(r"node_\d+_(\d+)", f.name)[0])
                q = node_point(f, names_old, f"omega_cdm_maxfun:{k:02d}")
                q["profile"] = "omega_cdm"
                more.append(q)
            followup["sets"][f"{pp}_profiles_ocdm"] = dict(cobaya=str(old_file.relative_to(KIT)),
                                                          job_env=f"runs/sidr_drmd2_{pp}_combined_boost10/job.env",
                                                          points=more)
        # follow-up 2: the four points on the full earlier data, CamSpec's nine nuisances re-optimised at each
        followup["sets"][f"{pp}_nuisopt"] = dict(cobaya=str(old_file.relative_to(KIT)),
                                                 job_env=f"runs/sidr_drmd2_{pp}_combined_boost10/job.env",
                                                 points=old_points, optimise=CAMSPEC)
        print(f"{pp}/{wc}: A {pa['logl']:.1f} at log10 Geff {pa['params']['logGeff']:.2f}, dNeff "
              f"{pa['params']['delta_Neff_drmd']:.3f}; B {pb['logl']:.1f} at {pb['params']['logGeff']:.2f}; "
              f"C {pc['logl']:.1f} at {pc['params']['logGeff']:.2f}; D {pd_['logl']:.1f} at {pd_['params']['logGeff']:.2f}; "
              f"{len(pts)} profile nodes")

    (KIT / "smoke" / "chi2_compare_points.json").write_text(json.dumps(compare, indent=1))
    (KIT / "smoke" / "chi2_profile_points.json").write_text(json.dumps(profiles, indent=1))
    (KIT / "smoke" / "chi2_followup_points.json").write_text(json.dumps(followup, indent=1))
    print("written: smoke/chi2_{compare,profile,followup}_points.json, runs/chi2cmp_*")


if __name__ == "__main__":
    main()
