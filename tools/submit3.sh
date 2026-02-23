#!/bin/bash
# Submission script for Lemaitre4
#SBATCH --job-name=kore3
#SBATCH --partition=batch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ncpus=$SLURM_NTASKS

srun ../bin/spin_doctor.py $ncpus >> out2

# copy results back to global scratch
result_folder=$GLOBALSCRATCH/results/kore/$1/$folder
mkdir -p $result_folder/

cp -r bin/parameters.py $result_folder/
cp -r *out* $result_folder/
cp -r *.dat $result_folder/

rm *.field
rm *.npz
rm *.mtx
rm *.dat
rm *out*

