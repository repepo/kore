#!/bin/bash
#
# Script to run Kore simulations on a SLURM-managed cluster
#
# Calls:
#   sbatch ./srunKore.sh run_name                         one run with the current bin/parameters.py
#   sbatch --array 0-10 ./srunKore.sh name var d start step   a scan: var = start + i*step for task i
#   sbatch --array 0-10 ./srunKore.sh name var e start step   a scan: var = 10**(start + i*step)
# Examples:
#   sbatch --array 0-10 ./srunKore.sh run_ricb_ ricb d 0.3 0.1
#   sbatch --array 0-10 ./srunKore.sh run_Ek_ ViscosD e -5 -0.1
# The resources below can be overridden on the command line, e.g. sbatch --ntasks 8 --mem 60G ./srunKore.sh run_name
#
# Resources:
# - ntasks: the MPI ranks of assemble.py and solve.py, and the processes of submatrices.py's pool. It is written
#   to par.ncpus, and lmax - m + 1 must be a multiple of 2*ntasks (assemble.py stops otherwise).
# - mem: per node, for the whole job.
# - time: the sum of the stages. 
#
#SBATCH --job-name=kore
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#---------- Solve options -----------------------------------------------------------------------------
# Solver options are taken from petsc_opts in bin/params_default.py (override them in bin/parameters.py).
# Anything put in opts is added on the command line and takes precedence, e.g.:
#opts='-eps_balance twoside'
opts=''

#------------------------------------------------------------------------------------------------------

# Load modules (foss release), python environment, PETSc/SLEPc variables and $KORE_HOME directory
source ./tools/load_env.sh

ncpus=$SLURM_NTASKS
stage() { echo "$(date '+%F %T')  $*"; }

#---------- Run directory and parameters --------------------------------------------------------------
if [ $# -eq 1 ]; then
    folder='.'
elif [ $# -eq 5 ]; then
    var=$2
    exp=$3
    startvalue=$4
    step=$5
    if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
        echo "A scan needs a job array: sbatch --array 0-<num> ./srunKore.sh $*"
        exit 1
    fi
    k=$(echo "$startvalue + ($SLURM_ARRAY_TASK_ID * $step)" | bc | awk '{printf "%f", $0}')
    if [ "$exp" = 'e' ]; then
        value='10**'$k # powers of ten
    else
        value=$k # linear
    fi
    folder=${var}_${value}
else
    echo "Wrong number of arguments. Either one or five arguments are required."
    exit 1
fi

run_folder=$GLOBALSCRATCH/runs/kore/$1/$folder
result_folder=$GLOBALSCRATCH/results/kore/$1/$folder
mkdir -p $run_folder $result_folder
cd $run_folder
cp -r $KORE_HOME/* . # copies the source files

# tools/setpar.py replaces the statement par.<name> = ... of bin/parameters.py (or adds one at the end)
if [ $# -eq 5 ]; then
    echo "$folder: $var = $value"
    python3 tools/setpar.py $var "$value" || exit 1
fi
python3 tools/setpar.py ncpus $ncpus || exit 1

#---------- Run Kore ----------------------------------------------------------------------------------
# Each stage runs only if the ones before it succeeded; status is the number of the stage that failed (0: none)
status=0

# One BLAS thread per process: the pool of submatrices.py and the MPI ranks already use every CPU
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# submatrices.py: a pool of ncpus processes (python multiprocessing), run here in the batch step, which has
# every CPU of the job on this node
stage "submatrices.py"
python3 ./bin/submatrices.py $ncpus > out1 2>&1 || status=1

if [ $status -eq 0 ]; then
    stage "assemble.py"
    mpiexec -n $ncpus python3 ./bin/assemble.py > out2 2>&1 || status=2
fi

if [ $status -eq 0 ]; then
    stage "solve.py"
    mpiexec -n $ncpus python3 ./bin/solve.py $opts > out3 2>&1 || status=3
fi

if [ $status -eq 0 ] && [ -f no_conv_solution ]; then
    echo "No converged solution"
elif [ $status -eq 0 ]; then
    stage "spin_doctor.py"
    # spin_doctor.py is serial (its ncpus argument is no longer used): its only parallelism is numpy's BLAS threads
    OMP_NUM_THREADS=$ncpus OPENBLAS_NUM_THREADS=$ncpus MKL_NUM_THREADS=$ncpus \
        python3 ./bin/spin_doctor.py $ncpus > out4 2>&1 || status=4
fi

if [ $status -ne 0 ]; then
    echo "stage $status failed: see $run_folder/out$status"
fi

#---------- Copy the results --------------------------------------------------------------------------
# the parameters, the logs, the tables and the solutions: flow.npy, thermal.npy, ... (field_format 'npy', the
# default) or *.field ('text')
stage "copying the results to $result_folder"
cp bin/parameters.py bin/params_default.py $result_folder/
shopt -s nullglob
cp out* *.dat *.npy *.field $result_folder/
if [ -f no_conv_solution ]; then
    cp no_conv_solution $result_folder/
fi

stage "done"
exit $status
