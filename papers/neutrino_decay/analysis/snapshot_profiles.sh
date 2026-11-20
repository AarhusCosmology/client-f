#!/bin/bash
# Copy profile studies, finished or still running, from results_profiles/ (not synced) to logs/ (synced), as
# analysis/submit_profile_breakdown.sh does in its first step (25 October 2026: preliminary profiles for the paper's
# draft while the World Cup studies run):
#
#   bash analysis/snapshot_profiles.sh STUDY [STUDY ...]       # on the login node; seconds
#
# Per 1D study: logs/profiles_<STUDY>_nodes/ (true/node_*.json, the nodes done so far, and surrogate_profiles.json and
# ranges.json) and, for a finished study, logs/profiles_<STUDY>.csv. The breakdown job later copies the complete
# studies over the same files. The paper's build_wc.py reads a study's csv when there is one, else its nodes.
#
# Per 2D study (29 October 2026: the axion runs' surfaces): logs/profiles_<STUDY>_surface/, with true_surface/ (its
# node_*.json, the nodes done so far, anchor.json and method.json), the scan's surrogate_surface.json, manifest.json,
# ranges.json, status.txt and, for a finished study, surface.csv, true_surface.json, surface_diagnostics.json, the
# README and its figures; a study stopped by an error also brings failure.json and failed_surrogate_diagnostics.json.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || { echo "usage: bash analysis/snapshot_profiles.sh STUDY [STUDY ...]" >&2; exit 1; }
for study in "$@"; do
  dir=results_profiles/$study
  if [ "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["configuration"].get("dimension", 1))' \
          "$dir/manifest.json" 2>/dev/null)" = 2 ]; then
    out=logs/profiles_${study}_surface
    mkdir -p "$out"
    nodes=0
    if ls "$dir"/true_surface/node_*.json >/dev/null 2>&1; then
      mkdir -p "$out/true_surface"
      cp "$dir"/true_surface/*.json "$out/true_surface/"
      nodes=$(ls "$dir"/true_surface/node_*.json | wc -l | tr -d ' ')
    fi
    for extra in manifest.json ranges.json status.txt surrogate_surface.json surface.csv true_surface.json \
                 surface_diagnostics.json README.md surface.png surface.pdf isocontours.png isocontours.pdf \
                 failure.json failed_surrogate_diagnostics.json; do
      [ ! -f "$dir/$extra" ] || cp "$dir/$extra" "$out/"
    done
    echo "$study (2D, $(cat "$dir/status.txt" 2>/dev/null || echo 'no status yet')): $nodes nodes of the true stage in $out/"
    continue
  fi
  if ! ls "$dir"/true/node_*.json >/dev/null 2>&1; then
    echo "$study: no node checkpoints in $dir/true/ yet"
    continue
  fi
  mkdir -p "logs/profiles_${study}_nodes"
  cp "$dir"/true/node_*.json "logs/profiles_${study}_nodes/"
  for extra in surrogate_profiles.json ranges.json; do
    [ ! -f "$dir/$extra" ] || cp "$dir/$extra" "logs/profiles_${study}_nodes/"
  done
  if [ -f "$dir/profiles.csv" ]; then
    cp "$dir/profiles.csv" "logs/profiles_$study.csv"
    state="finished"
  else
    state="running"
  fi
  echo "$study ($state): $(ls "logs/profiles_${study}_nodes"/node_*.json | wc -l) nodes in logs/profiles_${study}_nodes/"
done
