#!/

ncpus=4

opts='-st_type sinvert -eps_error_relative ::ascii_info_detail -eps_balance twoside -pc_factor_mat_solver_type mumps -mat_mumps_icntl_14 10000'

./bin/submatrices.py $ncpus
mpiexec -n $ncpus ./bin/assemble.py
mpiexec -n $ncpus ./bin/solve.py $opts
./bin/spin_doctor.py $ncpus
