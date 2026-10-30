#!/bin/bash
# Run from the top folder of kore (the one containing bin/):  ./bin/runKore.sh
# FIX (2026-10-30): paths ./bin/ again, as this line says (they were ../bin/ since 2026-10-16)
# Solver options are taken from petsc_opts in bin/parameters.py; extra ones can be appended to solve_nopp.py.

ncpus=4

set -e

python3 ./bin/submatrices.py $ncpus
mpiexec -n $ncpus ./bin/assemble.py
mpiexec -n $ncpus ./bin/solve_nopp.py
python3 ./bin/spin_doctor.py $ncpus

# python3 ./bin/submatrices.py $ncpus
# mpiexec -n $ncpus ./bin/assemble.py
# mpiexec -n $ncpus ./bin/solve_nopp.py
# sed -i '' -E 's/^([[:space:]]*par\.magnetic[[:space:]]*=[[:space:]]*)[01]/\11/' bin/parameters.py
# python3 ./bin/submatrices.py $ncpus
# mpiexec -n $ncpus ./bin/assemble.py
# python3 ./bin/just_induction.py $ncpus
# mpiexec -n $ncpus ./bin/solve_ind.py
# python3 ./bin/spin_doctor.py $ncpus
