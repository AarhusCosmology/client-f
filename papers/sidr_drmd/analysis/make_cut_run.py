"""A copy of a run with a narrower box in one parameter, for profile studies inside it (26 October 2026):

    python analysis/make_cut_run.py results/<timestamp>_<RUN> --parameter delta_Neff_drmd --lower 0.01 [--suffix dncut]

On the World Cup data without SH0ES the best fit lies in a narrow, fine-tuned valley at Delta N_eff ~ 3e-4 (f_idm 0.15:
the interacting dark matter, coupled to a negligible amount of dark radiation, has a small sound speed), and the
profile nodes that end there make the profiles jagged. The paper leaves the valley out with the cut Delta N_eff >= 0.01;
this makes the run its profiles are computed from. Writes results/<timestamp>_<RUN>_<suffix>/ and runs/<RUN>_<suffix>/:
  metadata.json                  the parameter's bound moved. The network is unchanged: it takes physical parameters,
                                 and the profile code reads the box from metadata.json as the space it searches (the true
                                 stage refuses points outside it);
  training_data/data_it_N.csv    without the rows outside the new box (the profile code refuses training rows outside);
  likelihood_input/*.yaml        the Cobaya prior of the parameter narrowed the same way (the true objective is -inf
                                 outside), and runs/<RUN>_<suffix>/cobaya.yaml likewise;
  analysis/posterior_it_N.npz    the report's chain without its samples outside (make_config.py reads the profile ranges
                                 off it; give the cut parameter's own range with --range);
  everything else                a symbolic link into the original run; runs/<RUN>_<suffix>/job.env a copy.
The original run is not touched. --dest DIR writes the copy elsewhere (a test).
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent.parent
SPECIAL = ("metadata.json", "training_data", "likelihood_input", "analysis")


def cut_prior(text, name, lower, upper):
    """The Cobaya YAML with the prior of `name` narrowed (exactly one prior block of that parameter)."""
    for key, value in (("min", lower), ("max", upper)):
        if value is None:
            continue
        pattern = rf"(?m)^(  {re.escape(name)}:\n    prior:\n(?:      \w+: .*\n)*?      {key}: )([^\n]+)$"
        text, n = re.subn(pattern, rf"\g<1>{value}", text)
        if n != 1:
            raise SystemExit(f"FATAL: no single prior {key} of {name} in the Cobaya file")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--parameter", required=True)
    p.add_argument("--lower", type=float)
    p.add_argument("--upper", type=float)
    p.add_argument("--suffix", default="cut")
    p.add_argument("--dest", type=Path, help="parent directory of the copy (default: beside the run; runs/ is then not written)")
    a = p.parse_args()
    if a.lower is None and a.upper is None:
        raise SystemExit("give --lower and/or --upper")
    src = a.run.resolve()
    m = re.fullmatch(r"(\d{8}_\d{6})_(.+)", src.name)
    if not m:
        raise SystemExit(f"FATAL: {src.name} is not <timestamp>_<RUN>")
    run_name = m.group(2)
    dst = (a.dest.resolve() if a.dest else src.parent) / f"{src.name}_{a.suffix}"
    if dst.exists():
        raise SystemExit(f"FATAL: {dst} exists")

    md = json.loads((src / "metadata.json").read_text())
    par = next((q for q in md["parameters"] if q["name"] == a.parameter), None)
    if par is None:
        raise SystemExit(f"FATAL: no parameter {a.parameter} in {src}/metadata.json")
    lo = par["lower"] if a.lower is None else a.lower
    hi = par["upper"] if a.upper is None else a.upper
    if not par["lower"] <= lo < hi <= par["upper"]:
        raise SystemExit(f"FATAL: [{lo}, {hi}] is not inside the box [{par['lower']}, {par['upper']}]")
    names = [q["name"] for q in md["parameters"]]
    col = names.index(a.parameter)
    inside = lambda v: lo <= v <= hi  # noqa: E731
    # The Cobaya files are cut before anything is written: a file whose prior cannot be cut stops the program with
    # nothing left behind, and it can be run again.
    cut_inputs = {f.name: cut_prior(f.read_text(), a.parameter, a.lower, a.upper)
                  for f in sorted((src / "likelihood_input").iterdir()) if f.suffix in (".yaml", ".yml")}
    if a.dest is None:
        runs_src, runs_dst = KIT / "runs" / run_name, KIT / "runs" / f"{run_name}_{a.suffix}"
        if runs_dst.exists():
            raise SystemExit(f"FATAL: {runs_dst} exists")
        runs_cobaya = cut_prior((runs_src / "cobaya.yaml").read_text(), a.parameter, a.lower, a.upper)

    dst.mkdir(parents=True)
    for entry in sorted(src.iterdir()):
        if entry.name not in SPECIAL:
            (dst / entry.name).symlink_to(entry)
    par["lower"], par["upper"] = lo, hi
    (dst / "metadata.json").write_text(json.dumps(md, indent=2))

    (dst / "training_data").mkdir()
    for f in sorted((src / "training_data").iterdir()):
        if not re.fullmatch(r"data_it_\d+\.csv", f.name):
            (dst / "training_data" / f.name).symlink_to(f)
            continue
        kept = dropped = 0
        with f.open() as fin, (dst / "training_data" / f.name).open("w") as fout:
            header = fin.readline()
            if header.strip().split(",")[:-1] != names:
                raise SystemExit(f"FATAL: {f} columns differ from metadata.json")
            fout.write(header)
            for line in fin:
                if inside(float(line.split(",", col + 1)[col])):
                    fout.write(line)
                    kept += 1
                else:
                    dropped += 1
        print(f"training_data/{f.name}: {kept} rows kept, {dropped} outside left out")

    (dst / "likelihood_input").mkdir()
    for f in sorted((src / "likelihood_input").iterdir()):
        if f.suffix in (".yaml", ".yml"):
            (dst / "likelihood_input" / f.name).write_text(cut_inputs[f.name])
            print(f"likelihood_input/{f.name}: prior of {a.parameter} now [{lo}, {hi}]")
        else:
            (dst / "likelihood_input" / f.name).symlink_to(f)

    (dst / "analysis").mkdir()
    for f in sorted((src / "analysis").iterdir()):
        if re.fullmatch(r"posterior_it_\d+\.npz", f.name):
            z = np.load(f, allow_pickle=True)
            zn = [str(n) for n in z["names"]]
            keep = np.array([inside(v) for v in z["x"][:, zn.index(a.parameter)]])
            out = {k: z[k] for k in z.files}
            out["x"] = z["x"][keep]
            np.savez(dst / "analysis" / f.name, **out)
            print(f"analysis/{f.name}: {int(keep.sum())} samples kept, {int((~keep).sum())} outside left out")
        else:
            (dst / "analysis" / f.name).symlink_to(f)

    if a.dest is None:
        runs_dst.mkdir()
        shutil.copy2(runs_src / "job.env", runs_dst / "job.env")
        (runs_dst / "cobaya.yaml").write_text(runs_cobaya)
        print(f"{runs_dst.relative_to(KIT)}: job.env and cobaya.yaml (prior of {a.parameter} [{lo}, {hi}])")
    print(f"written: {dst}")


if __name__ == "__main__":
    main()
