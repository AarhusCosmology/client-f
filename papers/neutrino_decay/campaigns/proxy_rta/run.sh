#!/bin/bash
# Runs the generated proxy inis on ONE binary, cheapest rung first, across
# WORKERS parallel processes.
#
# Parallel is safe here and not merely convenient: CLASS parallelises over k and
# the modes are independent, so the output is BIT-IDENTICAL across thread counts
# (verified on px_ours_G06.0_m0.3 at 8 vs 4 threads). One CLASS process reaches
# only ~1.9 cores on this machine, so 4 workers x 2 threads fills the box where
# 1 worker x 8 threads left three quarters of it idle.
#
# Usage: ./run.sh [workers]   (default 4)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# CLASS: a class binary built from CLASS++ 85deb9fc (the commit in EXPECTED_COMMIT).
CLASS="${CLASS:?set CLASS to a class binary built from CLASS++ $(cut -c1-8 "$HERE/EXPECTED_COMMIT")}"
[ -x "$CLASS" ] || { echo "FATAL: $CLASS is not executable"; exit 1; }
cd "$HERE"                     # the inis write to out/, relative to this folder
export OMP_NUM_THREADS=2
mkdir -p "$HERE/out"

# Sort on the BASENAME fields: model_G<gl>_m<mass>.ini -> key 2 is the rung.
# Sorting the absolute path instead puts field 3 inside the directory name and
# orders by model, which runs one whole model before the next one starts.
# (a plain read loop, not mapfile: macOS ships bash 3.2, which has no mapfile
# and would leave JOBS empty -- the pool would then report "0 jobs" and exit 0.)
JOBS=()
while IFS= read -r line; do JOBS+=("$line"); done < <(
  ls "$HERE"/px_*.ini | awk -F/ '{print $NF"\t"$0}' \
  | sed 's/^px_//' | sort -t_ -k2,2 -k1,1 | cut -f2)
[ ${#JOBS[@]} -gt 0 ] || { echo "FATAL: no jobs found"; exit 1; }

worker() {  # $1 = worker index, strided over the Gamma-sorted list so the
            # expensive top rungs land on different workers
  local w=$1 i=0 ini tag cl u t0
  for ini in "${JOBS[@]}"; do
    i=$((i+1))
    [ $(( (i-1) % WORKERS )) -eq "$w" ] || continue
    tag=$(basename "$ini" .ini)
    cl="$HERE/out/${tag}_cl_lensed.dat"
    if [ -s "$cl" ] && [ "$cl" -nt "$ini" ]; then
      printf '%-34s skip\n' "$tag"; continue
    fi
    t0=$SECONDS
    if "$CLASS" "$ini" > "$HERE/out/${tag}.log" 2>&1 && [ -s "$cl" ]; then
      printf '%-34s ok    %5ds\n' "$tag" "$((SECONDS-t0))"
    else
      printf '%-34s FAIL  %5ds\n' "$tag" "$((SECONDS-t0))"
    fi
    # CLASS silently ignores keys it does not recognise; an unused rta key means
    # the model that ini names is NOT the model that ran.
    u="$HERE/out/${tag}_unused_parameters"
    if [ -s "$u" ] && grep -qE 'dr_rta|dr_representation|N_ur' "$u" 2>/dev/null; then
      printf '  !! %s: MODEL KEY UNUSED -> %s\n' "$tag" \
             "$(grep -E 'dr_rta|dr_representation|N_ur' "$u" | tr '\n' ' ')"
    fi
  done
}

echo "$WORKERS workers, ${#JOBS[@]} jobs, OMP_NUM_THREADS=$OMP_NUM_THREADS"
for w in $(seq 0 $((WORKERS-1))); do worker "$w" & done
wait
echo "ALL DONE: $(ls "$HERE"/out/*_cl_lensed.dat 2>/dev/null | wc -l) cells on disk"
