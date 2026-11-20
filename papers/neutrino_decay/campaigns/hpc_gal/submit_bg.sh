#!/bin/bash -l
#SBATCH --job-name=dncdm_bg
#SBATCH --partition=q48,q40
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/bg_%A_%a.out
#SBATCH --error=logs/bg_%A_%a.out
#SBATCH --array=1-26%12
# ^ gen.py rewrites EVERY #SBATCH line above from config.py.  Re-run gen.py after
#   editing config.py and these stay consistent; editing them by hand does not
#   survive the next gen.py.
#
# BACKGROUND PRE-FLIGHT.  Run this FIRST -- it costs minutes and it is the
#   only thing that says whether a Gamma is worth submitting at all.  At an
#   under-resolved daughter grid the sector's comoving radiation content reads a
#   dramatic runaway in Delta N_eff that is entirely grid error and that NOTHING
#   in the run warns you about.  The background is serial, so 4 cores is plenty.
#
# All the actual work -- binary selection, the four guards, resume-safety,
# post-processing -- is in runcell.sh, shared by all four families so they cannot
# drift apart.
exec ./runcell.sh bg
