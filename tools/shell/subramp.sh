#!/bin/bash
#PBS -l select=1:ncpus=24:mem=60gb
#PBS -N somename
#PBS -l walltime=0:30:00

ncpus=24

dir=somename

cd $HOME/data/$dir

source $HOME/sou1
source $HOME/env_petsc/bin/activate

### For forced problems use:

# for simple test problems
#export opts='-ksp_type preonly -pc_type lu'
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason'
 
# use for standard problems with mumps (fast but requires more memory) amd an iterative solver (less memory but no guaranteed convergence)
#export opts='-ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor_true_residual -ksp_monitor -ksp_converged_reason'

# use for standard problems with mumps (fast but requires more memory) and a direct solver (more memory)
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor_true_residual -ksp_monitor -ksp_converged_reason'

# use for standard problems with superlu dist (should always work)
# export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1'

# use for induction problems
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS'

### For eigenvalue problems use:

#export opts='-st_type cayley -eps_error_relative ::ascii_info_detail'
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -eps_balance oneside -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 300'
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 1000 -mat_mumps_icntl_23 8000'
#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -st_pc_factor_mat_solver_type superlu_dist'
#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -st_pc_factor_mat_solver_type superlu_dist'
#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -eps_error_relative ::ascii_info_detail -st_pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1'
export opts='-st_type sinvert -st_pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 3000 -eps_true_residual -eps_converged_reason -eps_conv_rel -eps_monitor_conv -eps_error_relative ::ascii_info_detail -eps_balance twoside'

for k in $(seq 1 1 1)
do
	
	./bin/submatrices.py $ncpus >> out0
	mpiexec -n $ncpus ./bin/assemble.py >> out1
	mpiexec -n $ncpus ./bin/solve_nopp.py $opts >> out2
	./bin/spin_doctor.py $ncpus >> out3
			
done
	
deactivate

#rm *.field
rm *.npz
rm *.mtx

