#!/bin/bash
# The extra settings of 2 November 2026 (after run_all.sh): 4 processes at a time.
D=$(cd "$(dirname "$0")" && pwd)
B=${BUILD:?BUILD=CLASS++ 0a57bbc6 install directory}
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1
one() { "$PY" -I "$D/ladder3.py" "$B" "$D/generate_Pk_kpd.py" "$1" "$2" "$D/${1}_$2.npz" > "$D/${1}_$2.log" 2>&1; echo "$1 $2 exit $?"; }
export -f one; export D B PY
printf "wc WLQ\npp L4Q200\npp L8Q\npp L8Q200\nwc WL200\n" | xargs -P 4 -L 1 bash -c 'one "$0" "$1"'
echo "ALL DONE"
