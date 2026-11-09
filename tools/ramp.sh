#!/bin/bash
# Parameter ramp: runs the problem of bin/parameters.py once for each value of one parameter. Run from the top folder of
# kore (the one that holds bin/):
#
#     tools/ramp.sh name value1 value2 ...     e.g.   tools/ramp.sh ViscosD 1e-4 7e-5 5e-5
#     tools/ramp.sh                            runs the case as it is, once
#
# NEW (2026-11-08, from conductive_IC): each value (any python expression) is written into bin/parameters.py as
# par.name = value by tools/setpar.py, in place of the statements that set par.name, or at the end of the file if none
# does (after set_scales and the other values, so that it holds). Every step runs submatrices.py, assemble.py, solve.py and spin_doctor.py, which
# appends the step's results to the .dat files (docs/output.md; tools/get_data.py reads them). With par.track_target = 1
# each step aims at the eigenvalue that the step before wrote to the file track_target (the first step at the one the
# file holds: par.track_target = 2, or one written by hand). The ramp stops when a step finds no eigenvalue, or a forced
# step gives NaN (solve.py then writes the file no_conv_solution).
# The environment variables ncpus (the processes of submatrices.py and spin_doctor.py, default par.ncpus), ranks (the MPI
# ranks of assemble.py and solve.py, default ncpus), opts (more solver options, docs/solver.md) and MPIEXEC (default
# mpiexec) can be set.

set -e
[ -f bin/parameters.py ] || { echo "tools/ramp.sh: run it from the top folder of kore (bin/parameters.py)"; exit 1; }
ncpus=${ncpus:-$(python3 -c "import sys; sys.path.insert(0, 'bin'); from parameters import par; print(par.ncpus)" | tail -n 1)}
ranks=${ranks:-$ncpus}
mpi=${MPIEXEC:-mpiexec}


name=$1
shift || true
values=("$@")
[ -z "$name" ] && values=("")   # no ramp: one run of the case as it is

for value in "${values[@]}"; do
    if [ -n "$name" ]; then
        python3 tools/setpar.py "$name" "$value"
        echo "---------------- par.$name = $value"
    fi
    python3 ./bin/submatrices.py $ncpus
    $mpi -n $ranks python3 ./bin/assemble.py
    rm -f no_conv_solution
    $mpi -n $ranks python3 ./bin/solve.py $opts
    if [ -f no_conv_solution ]; then
        echo "No solution (no eigenvalue converged, or NaN)${name:+ at par.$name = $value}: ramp stopped"
        exit 1
    fi
    python3 ./bin/spin_doctor.py $ncpus
done
