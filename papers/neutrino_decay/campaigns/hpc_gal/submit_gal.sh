#!/bin/bash -l
#SBATCH --job-name=dncdm_gal
#SBATCH --partition=q40,q48
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=359:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/gal_%A_%a.out
#SBATCH --error=logs/gal_%A_%a.out
#SBATCH --array=1-14%12
# ^ gen.py rewrites EVERY #SBATCH line above from config.py.  Re-run gen.py after
#   editing config.py and these stay consistent; editing them by hand does not
#   survive the next gen.py.
#
# family C -- the Galerkin moment ladder.  q40/q48 are the quiet partitions.
#
# All the actual work -- binary selection, the four guards, resume-safety,
# post-processing -- is in runcell.sh, shared by all four families so they cannot
# drift apart.
exec ./runcell.sh gal
