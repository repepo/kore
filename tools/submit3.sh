#!/bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=kore3
#SBATCH --nodes=1
#SBATCH --partition=batch

ncpus=$SLURM_NTASKS

mpiexec -n $ncpus ./bin/solve.py "$@" >> out3
