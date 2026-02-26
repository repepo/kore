#!/bin/bash
#
# Script to run Kore simulations on a SLURM-managed cluster
# with a variable parameter to generate a map of modes.
#
# Call : sbatch --array 0-<num> ./doModes.sh somename var d startvalue step
# 
# Example calls: 
#   sbatch --array 0-10 ./doModes.sh run_ricb_ ricb d 0.3 0.1
#   sbatch --array 0-10 ./doModes.sh run_Ek_ Ek e -5 -0.1
#
# Where --array can be specified in the sbatch or change in the file
#
# Also possible to make a simple run : sbatch ./doModes.sh run_name, with the current parameter file
#SBATCH --job-name=kore
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000

#---------- Ressource allocation ----------------------------------------------------------------------
export time_run=00:10:00
export mem_per_cpu_run=4000
# Number of OpenMP threads for submatrices and postprocess and MPI processes for assemble and solve
export ncpus=10
# Number of modes to track (random sampling)
export nModes=100
#------------------------------------------------------------------------------------------------------  

#---------- Solve Options -----------------------------------------------------------------------------
### For eigenvalue problems use:
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'
export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -st_pc_factor_mat_solver_type mumps -st_mat_mumps_icntl_14 3000 -st_mat_mumps_icntl_23 14000 -eps_balance twoside'
#export opts='-st_type cayley -eps_error_relative ::ascii_info_detail'
#------------------------------------------------------------------------------------------------------ 

#---------- Parameters and run directories ------------------------------------------------------------
# Load modules (foss release), python environment, PETSc/SLEPc variables and $KORE_HOME directory
source ./tools/load_env.sh

# Check number of arguments
if [ $# -eq 1 ]; then
    folder='.'
    mkdir $LOCALSCRATCH/$folder
    cd $LOCALSCRATCH/$folder
    cp -r $KORE_HOME/* . # copies the source files
    sed -i 's,^\('ncpus'[ ]*=\).*,\1'$ncpus',' bin/parameters.py	

elif [ $# -eq 5 ]; then
    var=$2
    exp=$3
    startvalue=$4
    step=$5

    k=$(echo "$startvalue + ($SLURM_ARRAY_TASK_ID * $step)" | bc | awk '{printf "%f", $0}')
    if [ "$exp" = 'e' ]; then
        value='10**'$k # powers of ten
    else
        value=$k # linear
    fi

    # Create the run directories
    folder=${var}_${value}
    echo $folder $var=$value
    mkdir $LOCALSCRATCH/$folder
    cd $LOCALSCRATCH/$folder
    cp -r $KORE_HOME/* . # copies the source files

    # modify variables
    sed -i 's,^\('$var'[ ]*=\).*,\1'$value',' bin/parameters.py	
    sed -i 's,^\('ncpus'[ ]*=\).*,\1'$ncpus',' bin/parameters.py   

    srun sleep 0.2
else
    echo "Wrong number of arguments. Either one or five arguments are required."
    exit 1
fi

#------------------------------------------------------------------------------------------------------  
#---------- Run Kore ---------------------------------------------------------------------------------- 
# Submatrices
ID1=$(sbatch --parsable --time=$time_run --ntasks=1 --cpus-per-task=$ncpus --mem-per-cpu=$mem_per_cpu_run ./tools/submit1.sh)
# Assemble 
ID2=$(sbatch --parsable --time=$time_run --ntasks=$ncpus --cpus-per-task=1 --mem-per-cpu=$mem_per_cpu_run --dependency=afterok:${ID1} ./tools/submit2.sh)

for i in $(seq 1 1 $nModes)
do
    rnd1=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
    rnd2=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
    sed -i 's,^\(rnd1[ ]*=\).*,\1'$rnd1',g' bin/parameters.py
    sed -i 's,^\(rnd2[ ]*=\).*,\1'$rnd2',g' bin/parameters.py	

    # Solve
    ID3=$(sbatch --parsable --time=$time_run --ntasks=$ncpus --cpus-per-task=1 --mem-per-cpu=$mem_per_cpu_run --dependency=afterok:${ID2} ./tools/submit3.sh $opts)

    if [ -f no_conv_solution ]; then
        echo 'No converged solution'
        rm no_conv_solution
    else 
        # Results and Postprocessing
        ID4=$(sbatch --parsable --time=$time_run --ntasks=1 --cpus-per-task=$ncpus --mem-per-cpu=$mem_per_cpu_run --dependency=afterok:${ID3} ./tools/submit4.sh $result_folder)
    fi

    rm *.field
done

result_folder=$GLOBALSCRATCH/results/kore/$1/$folder
mkdir -p $result_folder/ 
# copy results back to global scratch

cp -r bin/parameters.py $result_folder/
cp -r *out* $result_folder/

rm *.npz
rm *.mtx
rm *.dat
rm *out*

