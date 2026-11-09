"""The four SIDR + DRMD runs on the data of the H0 World Cup papers (arXiv:2607.13282, 2607.13283), made from the
second set's files (Steen, 23 October 2026: the same cosmological model and priors as the earlier runs, which they
complement, not replace).

    python results/20261023_wc_runs/make_runs.py          # writes the kit's runs/sidr_drmd_wc{,mb}_{low,high}/

Each folder gets cobaya.yaml (byte-identical within a data set, as client.py combine requires), client.yaml,
domain_bounds.json and job.env. What changes from runs/sidr_drmd2_pp_low:
  - the likelihoods (LIKELIHOOD below), and the nuisance parameters they bring (NUISANCE);
  - CLASS++: master bfe6729e (install/10_classpp_build.sh, job.env) with pre/act_dr6.pre's lines minus
    nonlinear_min_k_max (Thomas Tram's suggestion: since #470 the sources are continued past the last mode, and the
    preset's 70/Mpc made SIDR + DRMD points take more than 45 minutes; results/20261023_classpp_act_preset_timing);
  - the two data sets: "wc", the World Cup baseline CMB+BAO+SN; "wcmb", the same plus SH0ES as a Gaussian prior on
    the supernova absolute magnitude, M_B = -19.253 +- 0.027 (their "+M_B").
Everything else (the cosmological parameters and their priors, the derived parameters, the split in delta_Neff,
client.yaml's settings) is the second set's.
"""
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / "20261007_grendel_kit" / "hpc_client"
SOURCE = KIT / "runs" / "sidr_drmd2_pp_low"
CLASSPP = Path(os.environ.get("CLASSPP_REPO", "CLASSpp"))     # a CLASS++ tree of the same commit, for the preset
COMMIT = "bfe6729e"
# The k up to which CLASS++ solves the modes for halofit (nonlinear_min_k_max; None: the default 2/Mpc). 10/Mpc, Steen's
# decision of 23 October 2026. At points with dark acoustic oscillations the continuation past 2/Mpc is off:
# - on the laptop (5 points), C_L^phiphi 6% (L <= 400) and 16% (L <= 2500) from modes solved to 30/Mpc, against 0.4%
#   and 1.2% at 10/Mpc, for 15% more time;
# - on Grendel (kit smoke/submit_build_check.sh, job 6950688; the 21 of the first run's 41 NaN points inside the
#   new prior), C_L^phiphi up to 8.8% (median 2.9%, L <= 400) from 10/Mpc, for 11% more time.
# At the best fit the two agree to 1e-4.
NONLINEAR_MIN_K_MAX = 10
PARTS = {"low": [0.0, 0.13], "high": [0.07, 2.0]}
COMBINED = ["boost1", "boost5", "boost10"]   # the analysis packages: networks S + 1, S + 5, S + 10 of each part

HEADER = """# SIDR + DRMD on the data of the H0 World Cup papers (Paper I arXiv:2607.13282, Paper II arXiv:2607.13283), {what}
# Made 23 October 2026 by results/20261023_wc_runs/make_runs.py from runs/sidr_drmd2_pp_low (Steen: the same
# cosmological model and priors as the earlier runs, which these complement, not replace).
#
# The data, as the World Cup's baseline CMB+BAO+SN ("A"){mb}:
#   - Planck PR4 NPIPE CamSpec TTTEEE (Cobaya's, v12.6; the World Cup names v12.7), cut to l < 1000 in TT and l < 600
#     in TE and EE: ACT's P-ACT cut, and the World Cup Paper I's (Paper II writes "l<1000 in TT,TE");
#   - Planck low-l TT (Commander) and SRoll2 low-l EE;
#   - ACT DR6 ACT-lite (foreground-marginalised; the package's default TT/TE/EE 600-6500), its calibration tied to
#     Planck's (A_act = A_planck, as ACT's P-ACT and SPT's SPA files do), with A_planck's CamSpec prior only;
#   - SPT-3G D1 T&E lite through candl, its internal priors cleared and Tcal, Ecal given here (SPT's SPA file);
#   - CMB lensing: ACT DR6 + Planck PR4 (act_dr6_lenslike, variant actplanck_baseline) and SPT-3G 2019-2020 MUSE
#     (phiphi only); SPT's own combination of the two (muselens_actplanckbaselens);
#   - DESI DR2 BAO, all tracers; Pantheon+ (z > 0.01), {sn}.
# Installed by install/11_lite_likelihoods.sh (pins in env.sh).
#
# CLASS++ master {commit} (22 October 2026), unpatched: install/10_classpp_build.sh, job.env. Its own fix (#467)
# replaces our extrapolation guard. Precision:
# - pre/act_dr6.pre's lines (ACT DR6 to l = 8500, tuned on ACT-lite) without its nonlinear_min_k_max = 70, with which
#   a SIDR + DRMD point did not finish in 45 minutes (results/20261023_classpp_act_preset_timing);
# - {kmax_text};
# - halofit, as the earlier runs, and their evolver_perturbations 1 and tol_perturb_integration 1e-7;
# - l_max_scalars is Cobaya's, from ACT-lite's 9000.
#
# The two parts of this data set, runs/{name}_low and runs/{name}_high, have byte-identical copies of this file,
# as client.py combine requires: the same objective, the training box differing in delta_Neff only.
# {n} sampled parameters: the second set's 21 and {extra}.
"""

LIKELIHOOD = """likelihood:
  planck_2018_lowl.TT: null
  planck_2018_lowl.EE_sroll2: null
  planck_NPIPE_highl_CamSpec.TTTEEE:
    dataset_params:
      use_cl: 143x143 217x217 143x217 TE EE
      use_range:                 # l < 1000 in TT, l < 600 in TE and EE (217 and 143x217 start at 500)
        143x143: 30-999
        217x217: 500-999
        143x217: 500-999
        TE: 30-599
        EE: 30-599
  act_dr6_cmbonly: null          # ACT-lite, DR6-ACT-lite at ACT_LITE_COMMIT (env.sh)
  candl.interface.CandlCobayaLikelihood:
    data_set_file: spt_candl_data.SPT3G_D1_TnE_lite
    clear_internal_priors: true  # its tau and Tcal priors; Tcal's is in params below, tau has none (SPT's SPA file)
    feedback: false
  act_dr6_lenslike.ACTDR6LensLike:
    lens_only: false
    lmax: 4000
    variant: actplanck_baseline
  muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse:
    components: ["ϕϕ"]           # lensing only, not the delensed EE
  bao.desi_dr2.desi_bao_all: null
"""
SN = {"wc": "  sn.pantheonplus: null           # uncalibrated: M_B marginalised analytically\n",
      "wcmb": "  sn.pantheonplus:\n    use_abs_mag: true            # M_B sampled, with the SH0ES prior (params)\n"}

NUISANCE = """  A_act:
    value: 'lambda A_planck: A_planck'
    latex: A_{\\rm ACT}
  P_act:
    prior:
      min: 0.9
      max: 1.1
    ref:
      dist: norm
      loc: 1.0
      scale: 0.01
    proposal: 0.01
    latex: p_{\\rm ACT}
  Tcal:
    prior:
      dist: norm
      loc: 1.0
      scale: 0.0036
    ref: 1.0
    proposal: 0.005
    latex: T_{\\rm cal}
  Ecal:
    prior:
      min: 0.8
      max: 1.2
    ref: 1.0
    proposal: 0.01
    latex: E_{\\rm cal}
"""
MB = """  Mb:
    prior:                       # SH0ES as the World Cup uses it: M_B = -19.253 +- 0.027 (Riess et al. 2022)
      dist: norm
      loc: -19.253
      scale: 0.027
    ref:
      dist: norm
      loc: -19.3
      scale: 0.02
    proposal: 0.01
    latex: M_B
"""
KMAX_TEXT = ("the default 2/Mpc, the sources continued past the last mode (#470)" if NONLINEAR_MIN_K_MAX is None else
             f"nonlinear_min_k_max = {NONLINEAR_MIN_K_MAX}: the continuation past the default 2/Mpc (#470) is off at "
             "dark acoustic oscillations (C_L^phiphi by up to 9%; make_runs.py)")
BOUNDS = {"P_act": [0.9, 1.1], "Tcal": [0.9784, 1.0216], "Ecal": [0.8, 1.2]}
BOUNDS_MB = {"Mb": [-19.6, -19.0]}


def theory():
    pre = {}
    for line in (CLASSPP / "pre" / "act_dr6.pre").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = (x.strip() for x in line.split("=", 1))
            pre[k] = v
    assert pre.pop("nonlinear_min_k_max") == "70"
    lines = ["      # pre/act_dr6.pre of CLASS++ " + COMMIT + ", every line but nonlinear_min_k_max = 70:"]
    if NONLINEAR_MIN_K_MAX is not None:
        lines = [f"      nonlinear_min_k_max: {NONLINEAR_MIN_K_MAX}   # modes solved to this k (1/Mpc) for halofit: "
                 "the dark acoustic oscillations (make_runs.py)"] + lines
    for k, v in pre.items():
        value = int(v) if re.fullmatch(r"-?\d+", v) else repr(float(v))   # 1e-4 as 0.0001: a float to any YAML reader
        lines.append(f"      {k}: {value}")
    return "\n".join(lines) + "\n"


def cobaya_text(data):
    src = (SOURCE / "cobaya.yaml").read_text()
    body = src[src.index("theory:\n"):]
    # the theory block: the earlier runs' SIDR inputs and settings, the halofit comment brought up to date, the preset
    body = body.replace("""      non_linear: halofit        # for the lensing potential (PR4 lensing) and the lensed spectra
                                 # with CLASS++ patched (job.env, install/09_classpp_guard.sh): at points with
                                 # dark acoustic oscillations the unpatched source extrapolation gave NaN spectra
""", "      non_linear: halofit        # as the earlier runs (ACT, SPT and their likelihoods' authors use HMcode 2020)\n")
    body = body.replace("      tol_perturb_integration: 1.0e-07\n", "      tol_perturb_integration: 1.0e-07\n" + theory())
    # the new nuisance parameters after the CamSpec ones
    anchor = "    latex: c_{EE}\n"
    assert body.count(anchor) == 1
    body = body.replace(anchor, anchor + NUISANCE + (MB if data == "wcmb" else ""))
    n = 24 + (data == "wcmb")
    extra = "P_act, Tcal, Ecal" + (", Mb" if data == "wcmb" else "")
    header = HEADER.format(
        what="WITH the SH0ES calibration of the supernovae (the World Cup's \"+M_B\")." if data == "wcmb"
        else "WITHOUT SH0ES (the World Cup's baseline \"A\").",
        mb=" plus SH0ES as a Gaussian prior on M_B" if data == "wcmb" else "",
        sn="calibrated by the Gaussian prior on M_B (Mb below)" if data == "wcmb" else "uncalibrated",
        commit=COMMIT, name=f"sidr_drmd_{data}", n=n, extra=extra, kmax_text=KMAX_TEXT)
    return header + LIKELIHOOD + SN[data] + body


def client_text(data, part):
    src = (SOURCE / "client.yaml").read_text()
    head, rest = src[:src.index("likelihood:\n")], src[src.index("likelihood:\n"):]
    rest = rest.replace("input: runs/sidr_drmd2_pp_low/cobaya.yaml", f"input: runs/sidr_drmd_{data}_{part}/cobaya.yaml")
    box = "[0, 0.13]" if part == "low" else "[0.07, 2]"
    other = "high" if part == "low" else "low"
    head = f"""# SIDR + DRMD on the H0 World Cup data ({'with SH0ES as a prior on M_B' if data == 'wcmb' else 'baseline, without SH0ES'}):
# runs/sidr_drmd_{data}_{part}/cobaya.yaml, identical in both parts; {24 + (data == 'wcmb')} sampled parameters.
# The {part.upper()} part of a domain split in delta_Neff: training box delta_Neff {box} (prior [0, 2]); every other
# cosmological range is its prior. The two parts are combined with
#   python client.py combine OUTPUT --parameter delta_Neff_drmd --part results/<time>_sidr_drmd_{data}_low \\
#       --part results/<time>_sidr_drmd_{data}_high --band 0.08:0.12
#
#   sbatch --time=7-00:00:00 submit_client.sh sidr_drmd_{data}_{part}
#
# Settings: exactly the second set's (runs/sidr_drmd2_pp_low/client.yaml; made by results/20261023_wc_runs/make_runs.py).
# Paths are relative to hpc_client/. The training box is this folder's domain_bounds.json; its other part is
# runs/sidr_drmd_{data}_{other}.
"""
    return head + rest


JOB_ENV = """# Read by submit_client.sh, smoke/submit_timing.sh and analysis/submit_true_check.sh.
# Made by results/20261023_wc_runs/make_runs.py (23 October 2026).
#
# CLIENT_REFUSED_VALUE is NOT set, as in the second set: a point CLASS++ or a likelihood cannot evaluate is -inf,
# left out of training, weight zero in the corrections.
unset CLIENT_REFUSED_VALUE

# CLASS++ master {commit}, UNPATCHED, built by install/10_classpp_build.sh into a folder of its own; neither the env's
# classy (7cc41a51) nor the second set's patched build is used. Its #467 replaces our extrapolation guard (NaN at dark
# acoustic oscillations), its #470 continues the sources past the last mode. CLASSPP_BUILD_DIR and
# CLASSPP_BUILD_COMMIT also tell install/check.py, launch.py and the profile backend to require this build
# (boltzmann_build.py). Without it the job stops here rather than run with another CLASS++.
export CLASSPP_BUILD_DIR="$HOME/codes/classpp_{commit}" CLASSPP_BUILD_COMMIT={commit}
ls "$CLASSPP_BUILD_DIR"/lib/classy*.so >/dev/null 2>&1 \\
  || {{ echo "FATAL: no CLASS++ {commit} in $CLASSPP_BUILD_DIR/lib: run install/10_classpp_build.sh {commit} first" >&2; return 1; }}
export PYTHONPATH="$CLASSPP_BUILD_DIR/lib${{PYTHONPATH:+:$PYTHONPATH}}"
"""


def main():
    source_bounds = json.loads((SOURCE / "domain_bounds.json").read_text())
    for data in ("wc", "wcmb"):
        text = cobaya_text(data)
        for part, box in PARTS.items():
            folder = KIT / "runs" / f"sidr_drmd_{data}_{part}"
            folder.mkdir(exist_ok=True)
            (folder / "cobaya.yaml").write_text(text)
            (folder / "client.yaml").write_text(client_text(data, part))
            bounds = dict(source_bounds)
            bounds["delta_Neff_drmd"] = box
            bounds.update(BOUNDS)
            if data == "wcmb":
                bounds.update(BOUNDS_MB)
            (folder / "domain_bounds.json").write_text(json.dumps(bounds, indent=1) + "\n")
            (folder / "job.env").write_text(JOB_ENV.format(commit=COMMIT))
            print("written:", folder.relative_to(KIT))
        # the folders of the combined emulators (analysis/queue_sidr_drmd_wc_combined.sh), as the second set's: the parts'
        # cobaya.yaml and job.env, the combined box (delta_Neff over its whole prior), a client.yaml naming the folder's file
        for label in COMBINED:
            folder = KIT / "runs" / f"sidr_drmd_{data}_combined_{label}"
            folder.mkdir(exist_ok=True)
            (folder / "cobaya.yaml").write_text(text)
            (folder / "job.env").write_text(JOB_ENV.format(commit=COMMIT))
            bounds = dict(source_bounds)
            bounds["delta_Neff_drmd"] = [0.0, 2.0]
            bounds.update(BOUNDS)
            if data == "wcmb":
                bounds.update(BOUNDS_MB)
            (folder / "domain_bounds.json").write_text(json.dumps(bounds, indent=1) + "\n")
            client = client_text(data, "low").replace(f"input: runs/sidr_drmd_{data}_low/cobaya.yaml",
                                                      f"input: runs/sidr_drmd_{data}_combined_{label}/cobaya.yaml")
            (folder / "client.yaml").write_text(f"# The combined emulator {label} of sidr_drmd_{data}: a client.yaml for the analysis "
                                                f"jobs (they read its likelihood input), from the low part's.\n" + client)
            print("written:", folder.relative_to(KIT))


if __name__ == "__main__":
    main()
