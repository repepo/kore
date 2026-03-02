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
### 4 : parent run folder
### 5 : parent result folder

ncpus=$1
time_run=$2
mem_per_cpu_run=$3

### For eigenvalue problems use:
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'
export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -st_pc_factor_mat_solver_type mumps -st_mat_mumps_icntl_14 3000 -st_mat_mumps_icntl_23 14000 -eps_balance twoside'
#export opts='-st_type cayley -eps_error_relative ::ascii_info_detail'

rnd1=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
rnd2=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
echo $rnd1; echo $rnd2; 

# Create the run directory (one per eigenvalue tracked)
run_folder=$4/$SLURM_ARRAY_TASK_ID
result_folder=$5/$SLURM_ARRAY_TASK_ID

mkdir -p $run_folder
cd $run_folder
cp -r ../bin . 
cp -r ../tools . 
cp -r ../*.npz . 

# Change eigenvalue track
sed -i 's,^\(rnd1[ ]*=\).*,\1'$rnd1',g' bin/parameters.py
sed -i 's,^\(rnd2[ ]*=\).*,\1'$rnd2',g' bin/parameters.py
srun sleep 0.2

# Solve 
ID3=$(sbatch --parsable --time=$time_run --ntasks=$ncpus --cpus-per-task=1 --mem-per-cpu=$mem_per_cpu_run ./tools/submit3.sh $opts)
# Results and Postprocessing
ID4=$(sbatch --parsable --time=$time_run --ntasks=1 --cpus-per-task=$ncpus --mem-per-cpu=$mem_per_cpu_run --dependency=afterok:${ID3} ./tools/submit4.sh $result_folder)





