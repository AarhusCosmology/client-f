#!/bin/bash
# The test of smoke/profile_chain.py on an analytic toy, through smoke/submit_profile_chain.sh (27 October 2026):
#
#   smoke/profile_chain_toy/run.sh SCRATCH        # SCRATCH: a new or empty folder; PY: the env's Python (default python)
#
# Copies the program, its helpers, the job script and env.sh into SCRATCH (a kit of its own), writes the toy there
# (make_toy.py) and runs: a full run (5 chains on 4 workers, seconds), the profile from both seeds against the exact one,
# a resume from a log cut after three nodes per chain (must equal the full run), a resume with a wrong node (refused),
# the node budget, the stall rule, an unknown chain (refused), --stop-above without and with a reference, the edge
# cases (a chain that fails, a node outside the prior) and a covariance whose shift goes the wrong way (the unshifted
# start must be taken and the profile stay exact). Every line must say PASS or "refused".
set -u
T=${1:?usage: run.sh SCRATCH}; S=$(cd "$(dirname "$0")" && pwd); KIT=$(cd "$S/../.." && pwd); PY=${PY:-python}
mkdir -p "$T/smoke" "$T/logs" "$T/toy"
cp "$KIT/env.sh" "$T/"
cp "$KIT"/smoke/{profile_chain.py,minimise_true.py,chi2_points_mp.py,submit_profile_chain.sh} "$T/smoke/"
rm -f "$T"/logs/profile_chain_*
$PY -I "$S/make_toy.py" "$T" > /dev/null
export HPC_CLIENT_DIR=$T SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-4}
cd "$T"
SLURM_JOB_ID=local bash smoke/submit_profile_chain.sh toy/spec.json toy > logs/run.out 2>&1 || echo "FAIL full run"
$PY -I - "$T" <<'PYEOF'
import json, sys
from pathlib import Path
T = Path(sys.argv[1]); ex = json.loads((T / "toy/exact.json").read_text()); const = -4.526126978647639
r = json.loads((T / "logs/profile_chain_toy_local.json").read_text())
env = {}
for ch in r["chains"].values():
    if ch["profile"] == "a":
        for nd in ch["nodes"]:
            x = round(nd["coordinate"], 4); env[x] = max(env.get(x, -1e9), nd["best"]["logpost"])
e = ex["a"]
worst = max(abs(2 * (v - (max(-0.5 * (x - e["mu1"]) ** 2 / e["var1"], -0.5 * (x - e["mu2"]) ** 2 / e["var2"] - e["off2"])
                          + e["nuis"] + const))) for x, v in env.items())
print("profile of a from both seeds:", len(env), "nodes, worst |chi2 error| %.1e" % worst, "PASS" if worst < 1e-3 else "FAIL")
(T / "logs/fake_previous.json").write_text(json.dumps({"chains": {k: dict(v, nodes=v["nodes"][:3]) for k, v in r["chains"].items()}}))
(T / "logs/fake_bad.json").write_text(json.dumps({"chains": {k: dict(v, nodes=[dict(v["nodes"][0], coordinate=9.9)]) for k, v in r["chains"].items()}}))
s = json.loads((T / "toy/spec.json").read_text()); s["reference_logpost"] = e["nuis"] + const
(T / "toy/spec_ref.json").write_text(json.dumps(s))
PYEOF
SLURM_JOB_ID=resume bash smoke/submit_profile_chain.sh toy/spec.json toy --resume logs/fake_previous.json > logs/resume.out 2>&1 || echo "FAIL resume"
SLURM_JOB_ID=bad bash smoke/submit_profile_chain.sh toy/spec.json toy --resume logs/fake_bad.json > logs/bad.out 2>&1 && echo "FAIL bad resume accepted" || echo "bad resume refused: $(tail -1 logs/bad.out)"
SLURM_JOB_ID=budget bash smoke/submit_profile_chain.sh toy/spec.json toy --chains main_up --node-budget 10 > logs/budget.out 2>&1 || echo "FAIL budget"
SLURM_JOB_ID=stall bash smoke/submit_profile_chain.sh toy/spec.json toy --chains main_up --stall 2 --stall-gain 1e9 > logs/stall.out 2>&1 || echo "FAIL stall"
SLURM_JOB_ID=badcov bash smoke/submit_profile_chain.sh toy/spec_badcov.json toy > logs/badcov.out 2>&1 || echo "FAIL badcov"
SLURM_JOB_ID=nochain bash smoke/submit_profile_chain.sh toy/spec.json toy --chains nonexistent > logs/nochain.out 2>&1 && echo "FAIL unknown chain accepted" || echo "unknown chain refused: $(tail -1 logs/nochain.out)"
SLURM_JOB_ID=noref bash smoke/submit_profile_chain.sh toy/spec.json toy --stop-above 3 > logs/noref.out 2>&1 && echo "FAIL stop without reference accepted" || echo "stop without reference refused: $(tail -1 logs/noref.out)"
SLURM_JOB_ID=stop bash smoke/submit_profile_chain.sh toy/spec_ref.json toy --stop-above 3 > logs/stop.out 2>&1 || echo "FAIL stop"
SLURM_JOB_ID=edge bash smoke/submit_profile_chain.sh toy/spec_edge.json edge > logs/edge.out 2>&1 || echo "FAIL edge"
$PY -I - "$T" <<'PYEOF'
import json, sys
from pathlib import Path
T = Path(sys.argv[1]); L = T / "logs"
a = json.loads((L / "profile_chain_toy_local.json").read_text())["chains"]
b = json.loads((L / "profile_chain_toy_resume.json").read_text())["chains"]
ok = all([round(n["best"]["logpost"], 6) for n in a[k]["nodes"]] == [round(n["best"]["logpost"], 6) for n in b[k]["nodes"]] for k in a)
print("resume equals the full run:", "PASS" if ok else "FAIL")
c = json.loads((L / "profile_chain_toy_budget.json").read_text())["chains"]["main_up"]["nodes"]
print("node budget:", "PASS" if all(n["ended"] == "node budget" and n["evaluations"] <= 10 + 2 + 7 for n in c) else "FAIL")
ref = json.loads((T / "toy/spec_ref.json").read_text())["reference_logpost"]
okst = True
for k, v in json.loads((L / "profile_chain_toy_stop.json").read_text())["chains"].items():
    d = [2 * (ref - n["best"]["logpost"]) for n in v["nodes"]]
    if "stopped_above" in v and not (d[-1] > 3 and all(x <= 3 for x in d[:-1])):
        okst = False
    if "stopped_above" not in v and any(x > 3 for x in d):
        okst = False
print("stop rule:", "PASS" if okst else "FAIL")
st = json.loads((L / "profile_chain_toy_stall.json").read_text())["chains"]["main_up"]["nodes"]
print("stall rule:", "PASS" if all((n["ended"] == "stalled" and n["evaluations"] == 2 + 21) or (n["ended"] != "stalled" and n["evaluations"] <= 2 + 14)
                                   for n in st) and any(n["ended"] == "stalled" for n in st) else "FAIL",
      [(n["evaluations"], n["ended"][:10]) for n in st][:3])
ex = json.loads((T / "toy/exact.json").read_text())["a"]; const = -4.526126978647639
def envelope_error(log):
    env = {}
    for ch in json.loads((L / log).read_text())["chains"].values():
        if ch["profile"] == "a":
            for nd in ch["nodes"]:
                x = round(nd["coordinate"], 4); env[x] = max(env.get(x, -1e9), nd["best"]["logpost"])
    return max(abs(2 * (v - (max(-0.5 * (x - ex["mu1"]) ** 2 / ex["var1"], -0.5 * (x - ex["mu2"]) ** 2 / ex["var2"] - ex["off2"])
                             + ex["nuis"] + const))) for x, v in env.items())
def starts(log, label):
    return [n["start"]["chosen"] for n in json.loads((L / log).read_text())["chains"][label]["nodes"]]
good, bad = starts("profile_chain_toy_local.json", "main_up"), starts("profile_chain_toy_badcov.json", "main_up")
err = envelope_error("profile_chain_toy_badcov.json")
# a wrong covariance also whitens the search badly, so the nodes end less precisely: 0.05 in chi^2, far below the noise of
# a real likelihood, is the test of a chain that is not thrown off
print("wrong covariance:", "PASS" if err < 0.05 and good.count("shifted") > len(good) / 2 and bad.count("unshifted") > len(bad) / 2 else "FAIL",
      "error %.1e; main_up starts with the right covariance %s, with the wrong one %s" % (err, good, bad))
e = json.loads((L / "profile_chain_edge_edge.json").read_text())["chains"]
print("edge cases:", "PASS" if ("error" in e["broken"] and e["outside"]["nodes"][0]["ended"] == "no finite value"
                                 and e["outside"]["nodes"][1]["best"] and len(e["fine"]["nodes"]) == 2) else "FAIL")
PYEOF
