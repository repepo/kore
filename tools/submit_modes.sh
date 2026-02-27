#!/bin/bash
#SBATCH --job-name=kore_modes
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100

### Parameters
### 1 : ncpus
### 2 : time_run
### 3 : mem-per-cpu
### 4 : opts
### 5 : run folder
### 6 : result folder

ncpus=$1
time_run=$2
mem_per_cpu_run=$3
otps=$4

rnd1=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
rnd2=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
echo $rnd1; echo $rnd2; 

# Create the run directory (one per eigenvalue tracked)
run_folder=$LOCALSCRATCH/$5
result_folder=$6/$SLURM_ARRAY_TASK_ID

mkdir $run_folder/$SLURM_ARRAY_TASK_ID
cd $run_folder/$SLURM_ARRAY_TASK_ID
cp -r $run_folder/* . # copies the source files and the assembled matrices

# Change eigenvalue track
sed -i 's,^\(rnd1[ ]*=\).*,\1'$rnd1',g' bin/parameters.py
sed -i 's,^\(rnd2[ ]*=\).*,\1'$rnd2',g' bin/parameters.py

srun sleep 0.2

# Solve 
ID3=$(sbatch --parsable --time=$time_run --ntasks=$ncpus --cpus-per-task=1 --mem-per-cpu=$mem_per_cpu_run ./tools/submit3.sh $4)
# Results and Postprocessing
if [ -f no_conv_solution ]; then
    echo 'No converged solution'
else 
    # Results and Postprocessing
    ID4=$(sbatch --parsable --time=$time_run --ntasks=1 --cpus-per-task=$ncpus --mem-per-cpu=$mem_per_cpu_run --dependency=afterok:${ID3} ./tools/submit4.sh $result_folder)
fi




