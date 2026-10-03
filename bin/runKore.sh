#!/bin/bash
# Run from the top folder of kore (the one containing bin/):  ./bin/runKore.sh
# Solver options are taken from petsc_opts in bin/parameters.py; extra ones can be appended to solve_nopp.py.

ncpus=4

./bin/submatrices.py $ncpus
mpiexec -n $ncpus ./bin/assemble.py
mpiexec -n $ncpus ./bin/solve_nopp.py
./bin/spin_doctor.py $ncpus
