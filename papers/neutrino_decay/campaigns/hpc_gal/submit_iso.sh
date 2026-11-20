#!/bin/bash -l
#SBATCH --job-name=dncdm_iso
#SBATCH --partition=q48,q40,q64,q128,q144
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=359:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/iso_%A_%a.out
#SBATCH --error=logs/iso_%A_%a.out
#SBATCH --array=1-57%24
# ^ gen.py rewrites EVERY #SBATCH line above from config.py.  Re-run gen.py after
#   editing config.py and these stay consistent; editing them by hand does not
#   survive the next gen.py.
#
# family B -- a4PiDot for the isotropisation fit.  cpus MUST equal the
#   number of k modes the cell solves (config.K_OUTPUT_ISO + 1): one mode per
#   core, so the wall time is the slowest single mode and extra modes are free.
#   Submitted to a partition LIST so it lands wherever there is a gap.
#
# All the actual work -- binary selection, the four guards, resume-safety,
# post-processing -- is in runcell.sh, shared by all four families so they cannot
# drift apart.
exec ./runcell.sh iso
