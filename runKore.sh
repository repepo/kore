#!/bin/bash

ncpus=4

#opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'

# # Iterative solver  
# opts='-pc_type lu -ksp_type gmres -ksp_monitor -memory_view' 
# # opts='-pc_type gamg -ksp_type gmres -ksp_monitor -memory_view' 

# Direct solver
opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 2000 -mat_mumps_icntl_14 2000 -ksp_monitor -memory_view' 

# opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 2000' 
# -mat_mumps_icntl_14 2000 -ksp_monitor_true_residual -ksp_monitor -ksp_converged_reason'
# opts='-ksp_type preonly -pc_type lu'

if [ "$1" == "purge" ]; then
    echo "Purging old matrices..."
    rm *.mtx *.npz
fi

./bin/submatrices.py $ncpus
mpiexec -n $ncpus ./bin/assemble.py
mpiexec -n $ncpus ./bin/solve.py $opts

# mpiexec -n $ncpus ./bin/solve.py

./get_pressure.py 0 100

#./postprocess.py
