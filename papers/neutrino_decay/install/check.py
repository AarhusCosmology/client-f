"""Installation guard. Exits non-zero on the first thing that is not what the kit assumes.

    python install/check.py --stage classy            # after step 3
    python install/check.py --stage camb              # after step 4
    python install/check.py --stage all               # after step 5, and at the top of every job
    python install/check.py --stage all --run RUN     # ... checking only runs/RUN
    python install/check.py --stage runs --run RUN    # the run inputs alone (also works off the cluster)

A check that cannot run is a FAILURE, not a skip.

Adapted from the hpc_client harness's check.py. Added: CAMB, the explicit domain of a run,
the comparison of the two Boltzmann codes, and the test that Cobaya evaluates with the
environment's own classy and camb and not with a copy under packages_path/code.
"""
import argparse
import os
import re
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENV = os.environ

# A Planck-like LambdaCDM point with massless neutrinos (N_eff = 3.045), as the runs here use.
FIDUCIAL = dict(omega_b=0.02238, omega_cdm=0.1201, H0=67.4, A_s=2.1e-9, n_s=0.9659, tau=0.0543)


def fail(msg):
    print(f"FATAL: {msg}", flush=True)
    sys.exit(1)


def ok(msg):
    print(f"  ok  {msg}", flush=True)


def need_env(name):
    value = ENV.get(name)
    if not value:
        fail(f"${name} is not set -- source env.sh")
    return value


def inside_env(path):
    return os.path.realpath(path).startswith(os.path.realpath(sys.prefix) + os.sep)


def check_environment():
    prefix = need_env("CLIENT_ENV")
    if os.path.realpath(sys.prefix) != os.path.realpath(prefix):
        fail(f"python is {sys.prefix}, not the kit's env {prefix}")
    ok(f"python {sys.version.split()[0]} from {sys.prefix}")
    import numpy
    import scipy
    import sklearn
    import tensorflow as tf
    import keras
    ok(f"numpy {numpy.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, "
       f"tensorflow {tf.__version__}, keras {keras.__version__}")
    pinned = ENV.get("KERAS_VERSION")
    if pinned and keras.__version__ != pinned:
        fail(f"keras is {keras.__version__}, env.sh pins {pinned}")
    gpus = tf.config.list_physical_devices("GPU")
    asked = ENV.get("SLURM_GPUS_ON_NODE") or ENV.get("CUDA_VISIBLE_DEVICES")
    if asked and not gpus:
        fail(f"the job was given GPUs ({asked}) but TensorFlow sees none: CPU wheels, or a driver/CUDA mismatch")
    ok(f"TensorFlow sees {len(gpus)} GPU(s){': ' + ', '.join(g.name for g in gpus) if gpus else ''}")
    # MPI must be the cluster's module runtime, never a conda MPI inside the env.
    import shutil
    import mpi4py
    from mpi4py import MPI
    mpicc = (mpi4py.get_config() or {}).get("mpicc", "")
    if mpicc.startswith(sys.prefix):
        fail(f"mpi4py was built with conda's mpicc ({mpicc}), not the module's")
    ldd = subprocess.run(["ldd", MPI.__file__], capture_output=True, text=True)
    if ldd.returncode != 0:
        fail(f"ldd failed on {MPI.__file__} -- cannot tell which MPI library mpi4py loads")
    libmpi = [line.split("=>")[-1].split()[0] for line in ldd.stdout.splitlines()
              if "libmpi" in line and "=>" in line]
    if not libmpi:
        fail(f"mpi4py's extension links no libmpi:\n{ldd.stdout}")
    if any(path.startswith(sys.prefix) for path in libmpi):
        fail(f"mpi4py loads a conda MPI {libmpi}: load the MPI module, rebuild mpi4py (step 2)")
    launcher = shutil.which("mpirun") or ""
    if not launcher or launcher.startswith(sys.prefix):
        fail(f"mpirun is {launcher or 'missing'}: the MPI module is not loaded")
    ok(f"MPI: {MPI.Get_library_version().splitlines()[0]} via {libmpi[0]}, launcher {launcher}")


def classy_spectrum(lmax=2000):
    """Lensed D_l^TT in muK^2 from classy at the fiducial point, and the seconds it took."""
    import numpy as np
    from classy import Class

    c = Class()
    c.set({"output": "tCl,pCl,lCl", "lensing": "yes", "l_max_scalars": lmax + 500,
           "omega_b": FIDUCIAL["omega_b"], "omega_cdm": FIDUCIAL["omega_cdm"], "H0": FIDUCIAL["H0"],
           "A_s": FIDUCIAL["A_s"], "n_s": FIDUCIAL["n_s"], "tau_reio": FIDUCIAL["tau"],
           "N_ncdm": 0, "N_ur": 3.045})
    t = time.perf_counter()
    c.compute()
    cl = c.lensed_cl(lmax)
    seconds = time.perf_counter() - t
    ell = np.arange(lmax + 1)
    dl = cl["tt"][: lmax + 1] * ell * (ell + 1) / (2 * np.pi) * (c.T_cmb() * 1e6) ** 2
    return dl, seconds


def camb_spectrum(lmax=2000):
    """The same from CAMB."""
    import camb

    pars = camb.set_params(H0=FIDUCIAL["H0"], ombh2=FIDUCIAL["omega_b"], omch2=FIDUCIAL["omega_cdm"],
                           As=FIDUCIAL["A_s"], ns=FIDUCIAL["n_s"], tau=FIDUCIAL["tau"], mnu=0.0,
                           num_massive_neutrinos=0, nnu=3.045, lmax=lmax + 500, lens_potential_accuracy=1)
    t = time.perf_counter()
    results = camb.get_results(pars)
    dl = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=lmax)["total"][: lmax + 1, 0]
    return dl, time.perf_counter() - t


def classy_location_problem(classy_file, site, guard_dir, kit_patch, expected, build_dir="", build_commit=""):
    """Why the classy found is not the one this job may use, or None. Without CLASSPP_GUARD_DIR or
    CLASSPP_BUILD_DIR it must be the env's own extension (step 3). A run whose job.env sets CLASSPP_GUARD_DIR
    (the SIDR + DRMD second set, 21 October 2026) must import the patched build of install/09_classpp_guard.sh
    from that folder's lib/, built from exactly the kit's patch at the pinned commit. One that sets
    CLASSPP_BUILD_DIR (the SIDR + DRMD World Cup runs, 23 October 2026) must import the unpatched build of
    install/10_classpp_build.sh from that folder's lib/, of the commit CLASSPP_BUILD_COMMIT names."""
    if guard_dir and build_dir:
        return "CLASSPP_GUARD_DIR and CLASSPP_BUILD_DIR are both set: a run evaluates with one CLASS++ build"
    if build_dir:
        lib = str(Path(build_dir, "lib")) + os.sep
        if not classy_file.startswith(lib) or not classy_file.endswith(".so"):
            return f"CLASSPP_BUILD_DIR={build_dir} is set, but classy is imported from {classy_file}, not from {lib}"
        if not build_commit:
            return "CLASSPP_BUILD_DIR is set without CLASSPP_BUILD_COMMIT, the commit the run asks for"
        if Path(build_dir, "applied.patch").exists():
            return f"{build_dir} holds a patch (applied.patch): a patched build belongs to CLASSPP_GUARD_DIR"
        provenance = Path(build_dir, "PROVENANCE.txt")
        if not provenance.exists() or not provenance.read_text().startswith(f"CLASS++ {build_commit} "):
            return f"{provenance} does not record a build of CLASS++ {build_commit} (run install/10_classpp_build.sh)"
        return None
    if not guard_dir:
        if not classy_file.startswith(site) or not classy_file.endswith(".so"):
            return f"classy imported from {classy_file}, not an extension in {site}"
        return None
    lib = str(Path(guard_dir, "lib")) + os.sep
    if not classy_file.startswith(lib) or not classy_file.endswith(".so"):
        return f"CLASSPP_GUARD_DIR={guard_dir} is set, but classy is imported from {classy_file}, not from {lib}"
    applied = Path(guard_dir, "applied.patch")
    if not applied.exists() or applied.read_bytes() != Path(kit_patch).read_bytes():
        return f"{guard_dir} was not built from the kit's {Path(kit_patch).name} (run install/09_classpp_guard.sh)"
    provenance = Path(guard_dir, "PROVENANCE.txt")
    if not provenance.exists() or not provenance.read_text().startswith(f"CLASS++ {expected} "):
        return f"{provenance} does not record a build of CLASS++ {expected}"
    return None


def check_classy():
    expected = (HERE / "EXPECTED_COMMIT").read_text().strip()
    build_commit = HERE / "BUILD_COMMIT"
    if not build_commit.exists():
        fail("no BUILD_COMMIT -- step 3 has not completed")
    if build_commit.read_text().strip() != expected:
        fail(f"classy built from {build_commit.read_text().strip()}, EXPECTED_COMMIT is {expected}")
    repo = need_env("CLASSPP_REPO")
    head = subprocess.run(["git", "-C", repo, "rev-parse", "--short=8", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != expected:
        fail(f"{repo} is at {head!r}, expected {expected}")

    import classy
    site = sysconfig.get_paths()["purelib"]
    guard = ENV.get("CLASSPP_GUARD_DIR", "").strip()
    build = ENV.get("CLASSPP_BUILD_DIR", "").strip()
    build_commit = ENV.get("CLASSPP_BUILD_COMMIT", "").strip()
    problem = classy_location_problem(classy.__file__, site, guard, Path(need_env("CLIENT_REPO")) / "hpc" / "patches" / "classpp_extrap_guard.patch", expected,
                                      build, build_commit)
    if problem:
        fail(problem)
    if guard:
        ok(f"classy: the patched build of install/09_classpp_guard.sh (CLASSPP_GUARD_DIR, the run's job.env): {guard}")
    if build:
        ok(f"classy: CLASS++ {build_commit} built by install/10_classpp_build.sh (CLASSPP_BUILD_DIR, the run's job.env): {build}")
    if Path(site, "classy", "__init__.py").exists():
        fail(f"stale {site}/classy/__init__.py would shadow the extension")
    # the version the source of the classy found declares: the clone's (steps 3 and 9), or the build's own tree
    source = Path(build, "src") if build else Path(repo)
    pyproject = Path(source, "pyproject.toml").read_text()
    version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    if getattr(classy, "__version__", None) != version:
        fail(f"classy.__version__ = {getattr(classy, '__version__', None)!r}, {source} says {version}")
    ok(f"classy {classy.__version__} (CLASS++) at {build_commit if build else expected}: {classy.__file__}")

    comment = subprocess.run(["readelf", "-p", ".comment", classy.__file__],
                             capture_output=True, text=True)
    if comment.returncode != 0:
        fail("readelf failed -- cannot tell which compiler built classy")
    compilers = set(re.findall(r"GCC: \(GNU\) ([0-9.]+)", comment.stdout))
    if not any(int(v.split(".")[0]) >= 9 for v in compilers):
        fail(f"classy has no GCC >= 9 in .comment ({sorted(compilers)}): built by the system compiler")
    ok(f"compilers in .comment: {sorted(compilers)} (an 8.5 entry alongside is glibc's crt files)")

    dl, seconds = classy_spectrum()
    if not dl[100] > 0:
        fail("classy: lensed D_l^TT(100) is not positive")
    ok(f"classy: lensed TT to l = 2000 in {seconds:.1f} s, D_100 = {dl[100]:.1f} muK^2 "
       f"(threads: OMP_NUM_THREADS={ENV.get('OMP_NUM_THREADS')})")


def check_camb():
    import camb

    pinned = ENV.get("CAMB_VERSION")
    if pinned and camb.__version__ != pinned:
        fail(f"camb is {camb.__version__}, env.sh pins {pinned}")
    if not inside_env(camb.__file__):
        fail(f"camb imported from {camb.__file__}, not from the env")
    dl, seconds = camb_spectrum()
    if not dl[100] > 0:
        fail("camb: lensed D_l^TT(100) is not positive")
    ok(f"camb {camb.__version__}: {camb.__file__}; lensed TT to l = 2000 in {seconds:.1f} s, "
       f"D_100 = {dl[100]:.1f} muK^2")


def check_agreement():
    """The two codes on one LambdaCDM point. They agree to a few parts in a thousand; a wrong
    build, wrong units or a different neutrino sector shows as per cents."""
    import numpy as np

    a, _ = classy_spectrum()
    b, _ = camb_spectrum()
    rel = (a[30:2001] - b[30:2001]) / b[30:2001]
    rms, worst = float(np.sqrt(np.mean(rel ** 2))), float(np.abs(rel).max())
    if not rms < 0.01:
        fail(f"CLASS++ and CAMB lensed TT differ by {100 * rms:.2f}% rms over l = 30..2000")
    ok(f"CLASS++ against CAMB, lensed TT over l = 30..2000: {100 * rms:.3f}% rms, {100 * worst:.3f}% at worst")


def check_client():
    repo = need_env("CLIENT_REPO")
    branch = subprocess.run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    if not commit:
        fail(f"{repo} is not a git checkout (sync_to_hpc.sh mirrors .git on purpose)")
    # CLiENT is the branch of client-f this folder lies in: any branch is taken, and its commit recorded
    expected_branch = (HERE / "CLIENT_BRANCH").read_text().strip() if (HERE / "CLIENT_BRANCH").exists() else branch
    if branch not in ("HEAD", expected_branch):
        fail(f"CLiENT checkout is on {branch!r}, CLIENT_BRANCH says {expected_branch!r}")
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    (HERE / "CLIENT_COMMIT").write_text(commit + (" (DIRTY)" if dirty else "") + "\n")
    sys.path.insert(0, repo)
    from likelihood.cobaya import CobayaLikelihood  # noqa: F401
    from model.network import load_model  # noqa: F401
    ok(f"CLiENT {branch} @ {commit}{' (DIRTY)' if dirty else ''}, recorded in CLIENT_COMMIT")


def check_runs(only=None, strict=True):
    """Every run input loads in CLiENT's own CobayaLikelihood, varies what Cobaya samples, has
    a finite box, and gives a finite log-posterior at its reference point with the env's code.
    strict=False (--stage runs, off the cluster) only reports where the Boltzmann module is."""
    import numpy as np
    import yaml

    repo = need_env("CLIENT_REPO")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    sys.path.insert(0, str(HERE))
    import client_paths  # noqa: E402,F401  CLiENT's hpc/ tools
    from domain import apply_domain, read_domain
    from likelihood.cobaya import CobayaLikelihood

    packages = need_env("COBAYA_PACKAGES_PATH")
    for code in ("classy", "CAMB", "camb"):
        if Path(packages, "code", code).exists():
            fail(f"{packages}/code/{code} exists: Cobaya would evaluate with it instead of the env's build")
    runs = sorted(p for p in (HERE / "runs").iterdir() if (p / "client.yaml").exists())
    if only is not None:
        runs = [p for p in runs if p.name == only]
    if not runs:
        fail(f"no runs/{only or '*'}/client.yaml")
    for run in runs:
        cfg = yaml.safe_load((run / "client.yaml").read_text())
        cobaya_yaml = HERE / cfg["likelihood"]["input"]
        if cobaya_yaml.resolve() != (run / "cobaya.yaml").resolve():
            fail(f"{run.name}: client.yaml's likelihood.input is {cfg['likelihood']['input']}, "
                 f"not runs/{run.name}/cobaya.yaml")
        info = yaml.safe_load(cobaya_yaml.read_text())
        if "packages_path" in info:
            fail(f"{run.name}: cobaya.yaml hard-codes packages_path; leave it out ($COBAYA_PACKAGES_PATH is used)")
        lik = CobayaLikelihood(str(cobaya_yaml))
        # Every parameter the Cobaya model samples must be one CLiENT varies,
        # or logposterior() is called without it.
        sampled = set(lik.cobaya_model.parameterization.sampled_params())
        varied = set(lik.param_names)
        if sampled != varied:
            fail(f"{run.name}: Cobaya samples {sorted(sampled - varied)} that the input does not "
                 f"list, or lists {sorted(varied - sampled)} that Cobaya does not sample")
        domain = run / "domain_bounds.json"
        if domain.exists():
            apply_domain(lik, read_domain(domain))
            box = "domain_bounds.json"
        elif cfg["prior"].get("n_sigma") is not None:
            missing = [p.name for p in lik._params if p.center is None or p.sigma is None]
            if missing:
                fail(f"{run.name}: no centre/width for {missing} (give each a ref: {{loc, scale}})")
            lik.restrict_prior_bounds(cfg["prior"]["n_sigma"])
            box = f"ref +- {cfg['prior']['n_sigma']} sigma"
        elif only is None and (run / "centre.env").exists():
            # A run whose box is made on the cluster (smoke/submit_centre.sh) and has not been
            # made yet does not stop a check of every run; asked for by name, it fails below.
            print(f"  note  {run.name}: no training box yet (sbatch smoke/submit_centre.sh {run.name}); not checked", flush=True)
            continue
        else:
            fail(f"{run.name}: neither runs/{run.name}/domain_bounds.json nor prior.n_sigma: no training box"
                 + (f" (make it: sbatch smoke/submit_centre.sh {run.name})" if (run / "centre.env").exists() else ""))
        bounds = np.array([lik.prior_bounds[n] for n in lik.param_names], dtype=float)
        if not (np.isfinite(bounds).all() and (bounds[:, 0] < bounds[:, 1]).all()):
            fail(f"{run.name}: the training box is not finite in every parameter")
        # The reference point of the input, moved into the box where it lies outside.
        point = np.array([bounds[k].mean() if p.center is None else float(p.center)
                          for k, p in enumerate(lik._params)])
        point = np.clip(point, bounds[:, 0], bounds[:, 1])
        t = time.perf_counter()
        value = lik.loglkl(point)
        dt = time.perf_counter() - t
        if not np.isfinite(value):
            fail(f"{run.name}: log-posterior at the reference point is {value}")
        codes = []
        for name in lik.cobaya_model.theory:
            root = name.split(".")[0]
            module = sys.modules.get(root)
            if root in ("classy", "camb") and module is not None and root not in [c[0] for c in codes]:
                codes.append((root, module.__file__, getattr(module, "__version__", "?")))
                if strict and root == "classy" and (ENV.get("CLASSPP_GUARD_DIR", "").strip()
                                                    or ENV.get("CLASSPP_BUILD_DIR", "").strip()):
                    sys.path.insert(0, str(HERE))
                    import client_paths  # noqa: E402,F401  CLiENT's hpc/ tools
                    from boltzmann_build import location_problem
                    problem = location_problem(root, module.__file__)
                    if problem:
                        fail(f"{run.name}: {problem}")
                elif strict and not inside_env(module.__file__):
                    fail(f"{run.name}: Cobaya evaluates with {root} from {module.__file__}, not the env's")
        ok(f"{run.name}: {lik.ndim} parameters ({box}), likelihoods {', '.join(lik.cobaya_model.likelihood)}; "
           f"logpost(reference) = {value:.2f} in {dt:.1f} s with "
           + (", ".join(f"{n} {v}" for n, _, v in codes) or "no Boltzmann code")
           + ("" if strict else "  [" + "; ".join(f for _, f, _ in codes) + "]"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["classy", "camb", "runs", "all"], required=True)
    parser.add_argument("--run", help="check only runs/<RUN> (default: every run)")
    args = parser.parse_args()
    if args.stage == "runs":
        check_runs(args.run, strict=False)
    else:
        check_environment()
        if args.stage in ("classy", "all"):
            check_classy()
        if args.stage in ("camb", "all"):
            check_camb()
        if args.stage == "all":
            check_agreement()
            check_client()
            check_runs(args.run)
    print("ALL CHECKS PASSED", flush=True)
