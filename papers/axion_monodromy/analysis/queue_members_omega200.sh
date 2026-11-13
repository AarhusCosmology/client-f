#!/bin/bash
# Several trainings per network on the omega-200 parts (4 November 2026; Steen: this is where the new training
# feature should show its worth). Both omega-200 runs, warm-started as by default, ran into what training.members with
# training.member_pilot cured on the high parts (analysis/member_pilot/README.md; CLiENT's docs/members.md): a setback
# with false maxima, then a plateau with the network too high on average at its own posterior points:
#   pp_omega200  results/20261103_084403_axion_pp_omega200  boosts from 4, setback at network 7 (scatter 4.2 at its
#                boost points), 1.29 at network 13, the last boost; ended at network 14
#   wc_omega200  results/20261103_084447_axion_wc_omega200  boosts from 5, setback at network 8 (10.1, false maxima at
#                omega 185-200), 2.41 at 9; still running (job 7011041), to network 15
#
#   bash analysis/queue_members_omega200.sh            # on the login node; ONLY=pp_omega200 (or wc_omega200) for one
#
# For each: a branch made with analysis/branch_run.py --train-from N --set training.members=12 --set
# training.member_pilot=100, N the network of the setback. The branch takes the source's networks 0 to N - 1 and the
# training set of N, trains N and every later network itself (the warm member from the branch's network N - 1, eleven
# from random weights, chosen by 1200 true evaluations at their own posterior points), and runs to the source's last
# network. Two nodes, 96 ranks: twelve members fit at submit_client.sh's 8 TensorFlow threads (the log warns if not).
# The source may still be running. ANALYSIS=1 also queues the usual analysis after the branch's job
# (analysis/queue_after_boosts.sh). OUT=DIR puts the branches in DIR (a dry run elsewhere) instead of results/.
# Needs CLiENT with training.members (dev-sth1509 2b4f77a or later; the member tests ran 91bdd1a).
set -euo pipefail
cd "$(dirname "$0")/.."
ONLY=${ONLY:-}
ANALYSIS=${ANALYSIS:-0}
OUT=${OUT:-}

# The network the boosts of RUN begin with: boost/start.json, or a branch's acquisition.boost.start_iteration.
START='
import json, sys
from pathlib import Path
import yaml
run = Path(sys.argv[1])
if (run / "boost" / "start.json").exists():
    print(json.loads((run / "boost" / "start.json").read_text())["start_iteration"])
else:
    settings = sorted(run.glob("*.yaml"))
    print(yaml.safe_load(settings[0].read_text())["acquisition"]["boost"]["start_iteration"])
'

branch() {   # SOURCE N NODES TASKS [branch_run.py --set ...]: make and submit one branch
  local src=$1 n=$2 nodes=$3 tasks=$4 out line dir count job s
  shift 4
  out=$(python analysis/branch_run.py "$src" --train-from "$n" ${OUT:+--out "$OUT"} "$@") \
    || { echo "FATAL: branch_run.py refused $src" >&2; return 1; }
  line=$(printf '%s\n' "$out" | head -1)
  dir=${line%%:*}
  count=$(printf '%s\n' "$out" | sed -n 's/.* -i \([0-9][0-9]*\).*/\1/p' | head -1)
  [ -d "$dir" ] && [[ "$count" =~ ^[0-9]+$ ]] || { echo "FATAL: no branch or no -i from branch_run.py: $out" >&2; return 1; }
  job=$(sbatch --parsable --nodes="$nodes" --ntasks="$tasks" submit_client.sh "$dir" -i "$count")
  echo "  $dir: networks $n to $((n + count)), job ${job%%;*}"
  if [ "$ANALYSIS" = 1 ]; then
    s=$(python -c "$START" "$src")
    bash analysis/queue_after_boosts.sh "$dir" "${job%%;*}" "$s" | sed 's/^/    /'
  fi
  sleep 2                       # two branches made in one second would have one name
}

for item in "pp_omega200 results/20261103_084403_axion_pp_omega200 7 2 96" \
            "wc_omega200 results/20261103_084447_axion_wc_omega200 8 2 96"; do
  read -r key src n nodes tasks <<< "$item"
  [ -z "$ONLY" ] || [ "$ONLY" = "$key" ] || continue
  if [ ! -d "$src" ]; then echo "$key: no $src here: skipped"; continue; fi
  echo "$key: $src"
  branch "$src" "$n" "$nodes" "$tasks" --set training.members=12 --set training.member_pilot=100
done
