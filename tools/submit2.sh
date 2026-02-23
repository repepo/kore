#!/bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=kore2
#SBATCH --nodes=1
#SBATCH --partition=batch

ncpus=$SLURM_NTASKS

mpiexec -n $ncpus ./bin/assemble.py >> out0
mpiexec -n $ncpus ./bin/solve.py "$@" >> out1
