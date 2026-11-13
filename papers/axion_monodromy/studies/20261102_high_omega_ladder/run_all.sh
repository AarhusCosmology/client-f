#!/bin/bash
# The whole ladder on the laptop: 3 PDP+ settings at a time plus the H0WC settings one after another (4 cores).
D=$(cd "$(dirname "$0")" && pwd)
B=${BUILD:?BUILD=CLASS++ 0a57bbc6 install directory}
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1
one() { "$PY" -I "$D/ladder3.py" "$B" "$D/generate_Pk_kpd.py" "$1" "$2" "$D/${1}_$2.npz" > "$D/${1}_$2.log" 2>&1; echo "$1 $2 exit $?"; }
export -f one; export D B PY
( printf "%s\n" REFP REF PQ L4Q L4K PK | xargs -P 3 -I{} bash -c 'one pp {}' ) &
( for s in W WK WL WQ WREF; do one wc $s; done ) &
wait
echo "ALL DONE"
