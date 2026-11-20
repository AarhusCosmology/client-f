#!/bin/bash
# The proxy column of table 1: the wall-clock time of one spectrum on one thread, on the inputs of figure 2.
#
#   CLASS=/path/to/class bash time_proxy.sh > proxy_timings_new.txt     # a class binary built from CLASS++ cb2255f4
#
# Run it on an otherwise idle machine, one cell at a time. The outputs go to a temporary folder and are deleted.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
INI="$HERE/../campaigns/proxy_rta"
CLASS="${CLASS:?set CLASS to a class binary built from CLASS++ cb2255f4}"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/time_proxy_XXXXXX")
trap 'rm -rf "$TMP"' EXIT
export OMP_NUM_THREADS=1
echo "# mass  log10Gamma  seconds   ($(date), $CLASS, one thread)"
for cell in 0.3:05.0 0.3:07.0 0.3:09.0 0.3:10.0 0.3:11.0 0.3:12.0 0.06:09.0 0.06:10.0; do
  mass=${cell%%:*}; gl=${cell#*:}
  sed "s|^root = .*|root = $TMP/px_|" "$INI/px_ours_G${gl}_m${mass}.ini" > "$TMP/cell.ini"
  t0=$(date +%s)
  "$CLASS" "$TMP/cell.ini" > "$TMP/cell.log" 2>&1
  echo "$mass  $gl  $(( $(date +%s) - t0 ))"
done
