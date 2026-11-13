#!/bin/bash
# The H0WC profile studies of the omega <= 200 paper, queued again so that each continues where it stopped after the
# 72 h limit of its current job. Copy it into the kit's analysis/ on Grendel and run it from the kit:
#
#   DRY=1 bash analysis/resubmit_wc_profiles200.sh      # print what it would do
#   bash analysis/resubmit_wc_profiles200.sh
#
# Per study: nothing if its status.txt says "complete"; if its current job is still in the queue, a new job of
# true_profiles/submit_profiles.sh with --dependency=afterany on it (it starts when that job ends, never beside it);
# otherwise a new job at once. submit_profiles.sh resumes a study that exists (finished nodes kept, true values already
# computed served from its journal). Run it again after the new jobs end if a study is still not complete.
#
# --iteration 0 as when the studies were begun: until CLiENT 38c506b (dev-sth1509, 11 November 2026) a resume compared
# the setting, and the follow-on jobs this script queued on 9 November without it were all refused at once
# ("source/configuration changed; resume refused"). The job numbers below are those resubmitted by hand on
# 11 November at about 08:25 (the first of them is (A_osc, phi), the last (A_osc, omega) window 4); (omega, phi) window 2
# is complete.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=results/20261107_084828_axion_wc_combined200
RUNS=runs/axion_wc_combined200
while read -r job cfg; do
  name=axion_wc200_$(basename "$cfg" .yaml | sed 's/^profiles_//')
  out=results_profiles/$name
  [ -f "$RUNS/$cfg" ] || { echo "FATAL: no $RUNS/$cfg" >&2; exit 1; }
  if [ "$(cat "$out/status.txt" 2>/dev/null)" = "complete" ]; then echo "skip $name (complete)"; continue; fi
  if [ -n "$(squeue -h -j "$job" 2>/dev/null)" ]; then dep=(--dependency="afterany:$job"); else dep=(); fi
  cmd=(sbatch --parsable ${dep[@]+"${dep[@]}"} true_profiles/submit_profiles.sh "$TARGET" "$RUNS/$cfg" "$out" --iteration 0)
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else echo "$("${cmd[@]}")  $name ${dep[@]+"${dep[@]}"}"; fi
done <<'EOF'
7077752 profiles_aosc_phi.yaml
7077753 profiles_aosc_omega_w1_a040.yaml
7077754 profiles_aosc_omega_w2_a070.yaml
7077755 profiles_aosc_omega_w3.yaml
7077764 profiles_aosc_omega_w4.yaml
7077756 profiles_omega_phi_w1.yaml
7053598 profiles_omega_phi_w2.yaml
7077757 profiles_omega_phi_w3.yaml
7077758 profiles_omega_phi_w4.yaml
7077760 windows/profiles_aosc_phi_om144.yaml
7077761 windows/profiles_aosc_phi_om158.yaml
7077762 windows/profiles_aosc_phi_om169.yaml
7077763 windows/profiles_aosc_phi_om195.yaml
7077759 windows/profiles_aosc_phi_om82.yaml
EOF
