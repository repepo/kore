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

RUN_FOLDER=../run
mkdir -p $RUN_FOLDER
cp -r ./bin/params_default.py $RUN_FOLDER
cp -r ./bin/parameters.py $RUN_FOLDER
for f in *.npy *.field; do [ -e "$f" ] && mv "$f" $RUN_FOLDER; done   # the solutions (field_format: .npy or text)
mv *.npz $RUN_FOLDER   # A, B and submatrices.npz (the operators)
if ls *.mtx > /dev/null 2>&1; then mv *.mtx $RUN_FOLDER; fi   # only with write_mtx = 1
mv *.dat $RUN_FOLDER
mv *out* $RUN_FOLDER