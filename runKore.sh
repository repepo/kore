#!/bin/bash

source $HOME/.kore_env.sh

ncpus=10

opts='-st_type sinvert -eps_error_relative ::ascii_info_detail'
#opts='-st_type sinvert -eps_error_relative -st_mat_mumps_icntl_14 1000 ::ascii_info_detail'

#opts='-ksp_type preonly -pc_type lu'

./bin/submatrices.py $ncpus
mpiexec -n $ncpus ./bin/assemble.py
mpiexec -n $ncpus ./bin/solve.py $opts
#./postprocess.py

RUN_FOLDER=../run
mkdir -p $RUN_FOLDER
cp -r ./bin/params_default.py $RUN_FOLDER
cp -r ./bin/parameters.py $RUN_FOLDER
mv *.field $RUN_FOLDER
mv *.npz $RUN_FOLDER
mv *.mtx $RUN_FOLDER
mv *.dat $RUN_FOLDER
mv *out* $RUN_FOLDER