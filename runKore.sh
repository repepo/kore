#!/bin/bash

ncpus=10

# Solver options are taken from petsc_opts in bin/params_default.py (override them in bin/parameters.py).
# Anything put in opts is added on the command line and takes precedence, e.g.:
#opts='-eps_balance twoside'
opts=''

./bin/submatrices.py $ncpus
mpiexec -n $ncpus ./bin/assemble.py
mpiexec -n $ncpus ./bin/solve.py $opts
./bin/spin_doctor.py $ncpus | tee out2
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