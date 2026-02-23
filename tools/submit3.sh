#!/bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=kore3
#SBATCH --partition=batch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ncpus=$SLURM_NTASKS

srun ./bin/spin_doctor.py $ncpus >> out2

# copy results back to global scratch

cp -r bin/parameters.py $1/
cp -r *out* $1/
cp -r *.dat $1/

rm *.field
rm *.npz
rm *.mtx
rm *.dat
rm *out*

