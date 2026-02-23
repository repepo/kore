#!/bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=kore3
#SBATCH --partition=batch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ncpus=$SLURM_NTASKS

srun ../bin/spin_doctor.py $ncpus >> out2
