#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l mem=160g
#PBS -N somename
#PBS -l walltime=4:00:00

ncpus=24

dir=somedir

cd $HOME/data/$dir

#./submatrices.py $ncpus >> out00
#mpiexec -n $ncpus ./assemble.py >> out0


# For forced problems use:

#export opts='-ksp_type preonly -pc_type lu'
 
#export opts='-ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor_true_residual -ksp_monitor -ksp_converged_reason'

#export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason'

export opts='-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_converged_reason -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1'


# For eigenvalue problems use:

#export opts='-st_type cayley -eps_error_relative ::ascii_info_detail'

#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'

#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -eps_balance oneside -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 300'

#export opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 1000 -mat_mumps_icntl_23 8000'

#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -st_pc_factor_mat_solver_type superlu_dist'

#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -st_pc_factor_mat_solver_type superlu_dist'

#export opts='-st_type sinvert -st_ksp_type preonly -st_pc_type lu -eps_error_relative ::ascii_info_detail -st_pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_iterrefine 1 -mat_superlu_dist_colperm PARMETIS -mat_superlu_dist_parsymbfact 1'


for i in $(seq 1 1 1)

do
	#sed -i 's,^\(itau[ ]*=\).*,\1'$i',g' parameters.py
	#sed -i 's,^\(Ek[ ]*=\).*,\1'10**$i',g' parameters.py
	#sed -i 's,^\(mu[ ]*=\).*,\1'$i',g' parameters.py
	#sed -i 's,^\(delta[ ]*=\).*,\1'$i',g' parameters.py
	#sed -i 's,^\(ricb[ ]*=\).*,\1'$i',g' parameters.py
	
	#echo $i

	./submatrices.py $ncpus >> out00
	#mpiexec -n $ncpus ./assemble.py >> out0
	#mpiexec -n $ncpus ./solve.py $opts >> out1

	for k in $(seq 0 0.002 0.008)
	#for k in $(seq 1 1 1)
	do
		#rnd1=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
		#echo $rnd1
		#rnd2=$(echo | awk -v seed=$RANDOM 'srand(seed) {print (2*rand()-1)}')
		#echo $rnd2 
		#sed -i 's,^\(Em[ ]*=\).*,\1'10**$k',g' parameters.py
		#sed -i 's,^\(itau[ ]*=\).*,\1'$k',g' parameters.py
		#sed -i 's,^\(rnd1[ ]*=\).*,\1'$rnd1',g' parameters.py
		#sed -i 's,^\(rnd2[ ]*=\).*,\1'$rnd2',g' parameters.py	
		sed -i 's,^\(delta[ ]*=\).*,\1'$k',g' parameters.py
		#sed -i 's,^\(ricb[ ]*=\).*,\1'$k',g' parameters.py
		#sed -i 's,^\(Le[ ]*=\).*,\1'10**$k',g' parameters.py

		#echo Solving...
		#./submatrices.py $ncpus >> out00
		mpiexec -n $ncpus ./assemble.py >> out0
		mpiexec -n $ncpus ./solve.py $opts >> out1
	done

done
rm *.npz

