#!/bin/bash
#
# Script to run Kore simulations on a SLURM-managed cluster
# with a variable parameter.
#
# Call : sbatch --array 0-<num> ./srunKore.sh somename var d startvalue step
# 
# Example calls: 
#   sbatch --array 0-10 ./srunKore.sh run_ricb_ ricb d 0.3 0.1
#   sbatch --array 0-10 ./srunKore.sh run_Ek_ Ek e -5 -0.1
#
# Where --array can be specified in the sbatch or change in the file
#
# Also possible to make a simple run : sbatch ./srunKore.sh, with the current parameter file

#---------- Ressource allocation ----------------------------------------------------------------------
export time_run=00:10:00
export mem_per_cpu_run=1000
# MPI processes for assemble and solve
export mpi_processes=8
# OpenMP threads for submatrices and postprocess
export openmp_threads=8
#------------------------------------------------------------------------------------------------------  

#---------- Solve Options -----------------------------------------------------------------------------
### For eigenvalue problems use:
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'
export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -st_pc_factor_mat_solver_type mumps -st_mat_mumps_icntl_14 3000 -st_mat_mumps_icntl_23 14000 -eps_balance twoside'
#export opts='-st_type cayley -eps_error_relative ::ascii_info_detail'

### For forced problems use:
### use for simple test problems
#export opts='-ksp_type preonly -pc_type lu'
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason'
### use for standard problems with mumps (fast but requires more memory) amd an iterative solver (less memory but no guaranteed convergence)
#export opts='-ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor_true_residual -ksp_monitor -ksp_converged_reason'
### use for standard problems with mumps (fast but requires more memory) and a direct solver (more memory)
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor_true_residual -ksp_monitor -ksp_converged_reason'
### use for standard problems with superlu dist (should always work)
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1'
#------------------------------------------------------------------------------------------------------ 

#---------- Parameters and run directories ------------------------------------------------------------
# Load modules (foss release), python environment, PETSc/SLEPc variables and $KORE_HOME directory
source ./tools/load_env

# Check number of arguments
if [ $# -eq 1 ]; then
    folder='.'
    sed -i 's,^\('ncpus'[ ]*=\).*,\1'$mpi_processes',' bin/parameters.py	

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
    #------------------------------------------------------------------------------------------------------  
    #------------------------------------------------------------------------------------------------------  

    # Create the run directories
    folder=${var}_${value}
    echo $folder $var=$value
    mkdir $LOCALSCRATCH/$folder
    cd $LOCALSCRATCH/$folder
    cp -r $KORE_HOME/* . # copies the source files

    # modify variables
    sed -i 's,^\('$var'[ ]*=\).*,\1'$value',' bin/parameters.py	
    sed -i 's,^\('ncpus'[ ]*=\).*,\1'$mpi_processes',' bin/parameters.py

    srun sleep 0.2
else
    echo "Wrong number of arguments. Either one or five arguments are required."
    exit 1
fi


ID0=$(sbatch --parsable --time=$time_run --ntasks=1 --cpus-per-task=$openmp_threads --mem-per-cpu=$mem_per_cpu_run ./tools/submit1.sh)
ID1=$(sbatch --parsable --time=$time_run --ntasks=$mpi_processes --mem-per-cpu=$mem_per_cpu_run --dependency=afterok:${ID0} ./tools/submit2.sh $opts)
#ID2=$(sbatch --parsable --time=$time_run --ntasks=1 --cpus-per-task=$openmp_threads --mem-per-cpu=$mem_per_cpu_run --dependency=afterok:${ID1} ./tools/submit3.sh)

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
