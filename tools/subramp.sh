#!/bin/bash
###PBS -l nodes=1:ppn=24
###PBS -l mem=245g
###PBS -l select=1:ncpus=24:mem=54gb
#PBS -l select=1:ncpus=24:mem=248gb
#PBS -N somename
#PBS -l walltime=03:00:00



source $HOME/venv00/bin/activate


ncpus=24

dir=somedir

cd $HOME/data/$dir



if [ -f no_conv_solution ]; then
	rm no_conv_solution
fi
if [ -f big_error ]; then
	rm big_error
fi

#./submatrices.py $ncpus >> out00
#mpiexec -n $ncpus ./assemble.py >> out0

#rm *.dat out*

# For forced problems use the following options:

#export opts='-ksp_type preonly -pc_type lu'
#export opts='-ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor_true_residual -ksp_monitor -ksp_converged_reason'
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 2000 -ksp_monitor_true_residual -ksp_monitor -ksp_converged_reason'
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason'
#export opts='-ksp_type gmres -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1'
#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1'


# For eigenvalue problems use:

#export opts='-st_type cayley -eps_error_relative ::ascii_info_detail'
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -eps_balance oneside -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 1000'
#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 1000 -mat_mumps_icntl_23 8000'
export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 10000 -eps_balance twoside'
#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -st_pc_factor_mat_solver_type superlu_dist'
#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -st_pc_factor_mat_solver_type superlu_dist'
#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -eps_error_relative ::ascii_info_detail -st_pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1'
#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -eps_error_relative ::ascii_info_detail -st_pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1 -eps_converged_reason -eps_conv_rel -eps_monitor_conv -eps_true_residual 1 -eps_balance oneside'
#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -eps_error_relative ::ascii_info_detail -st_pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1 -eps_converged_reason -eps_conv_rel -eps_monitor_conv -eps_true_residual 1'


#for j in $(seq 4 1 15)
for j in $(seq 1 1 1)
#for j in $(seq 0.67 0.003 0.99)
do

	#sed -i 's,^\(ricb[ ]*=\).*,\1'$j',g' bin/parameters.py

	#./bin/submatrices.py $ncpus >> out00
	#mpiexec -n $ncpus ./bin/assemble.py >> out0


	#for i in $(seq 0.05 0.05 1.0)
	for i in $(seq 1 1 1)
	do
		#sed -i 's,^\(itau[ ]*=\).*,\1'$i',g' bin/parameters.py
		#sed -i 's,^\(Ek[ ]*=\).*,\1'10**$i',g' bin/parameters.py
		#sed -i 's,^\(m[ ]*=\).*,\1'$i',g' bin/parameters.py
		#sed -i 's,^\(delta[ ]*=\).*,\1'$i',g' bin/parameters.py
		#sed -i 's,^\(ricb[ ]*=\).*,\1'$i',g' bin/parameters.py
	
		#echo $i

		./bin/submatrices.py $ncpus >> out00
		mpiexec -n $ncpus ./bin/assemble.py >> out0
		#mpiexec -n $ncpus ./bin/solve.py $opts >> out1
	
		#for k in $(seq 1 1 1)
		for k in $(seq 1 1 3 )
		do
	
			#if [ -f no_conv_solution ] && [ -f track_target ]; then
			#	echo 'No converged solution, stopping'
			#	break
			#fi
			#if [ -f big_error ] && [ -f track_target ]; then
			#	echo 'Error too big, stopping'
			#	break
			#fi
	
			rnd1=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
			#echo $rnd1
			rnd2=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
			#echo $rnd2 
			#sed -i 's,^\(Ek[ ]*=\).*,\1'10**$k',g' bin/parameters.py
			#sed -i 's,^\(itau0[ ]*=\).*,\1'$k',g' bin/parameters.py
			sed -i 's,^\(rnd1[ ]*=\).*,\1'$rnd1',g' bin/parameters.py
			sed -i 's,^\(rnd2[ ]*=\).*,\1'$rnd2',g' bin/parameters.py	
			#sed -i 's,^\(delta[ ]*=\).*,\1'$k',g' bin/parameters.py
			#sed -i 's,^\(N[ ]*=\).*,\1'$k',g' bin/parameters.py
			#sed -i 's,^\(Lambda[ ]*=\).*,\1'$k',g' bin/parameters.py
			#sed -i 's,^\(Le[ ]*=\).*,\1'10**$k',g' bin/parameters.py
			#sed -i 's,^\(m[ ]*=\).*,\1'$k',g' bin/parameters.py
			#sed -i 's,^\(rc[ ]*=\).*,\1'$k',g' bin/parameters.py
	
			#echo 'Solving for Pm = 10**'$k >> out1
			
			#rm *.dat out*
			
			#./bin/submatrices.py $ncpus >> out00
			#mpiexec -n $ncpus ./bin/assemble.py >> out0
			
			#python3 tools/just_induction.py
			#mpiexec -n $ncpus ./bin/solve_ind.py $opts >> out2
			
			mpiexec -n $ncpus ./bin/solve.py $opts >> out1
	
			#cp tools/underflow.py .
			#deactivate
			#source $HOME/venv_shtns2/bin/activate
			#export PYTHONPATH=$HOME/venv_shtns2/lib64/python-3.6/site-packages
			
			#mv real_magnetic_ind.field real_magnetic.field
			#mv imag_magnetic_ind.field imag_magnetic.field
	
			#rm *max.dat
			#python3 underflow.py
			
			#python3 bin/find_Rac_pbs.py >> out2
	
		done

	done

done

rm *.npz *.mtx

