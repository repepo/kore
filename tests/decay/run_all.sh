#!/bin/bash
# Runs the checks of the runs without the flow, hydro = 0, and of the kinematic induction (README.md).
# usage: tests/decay/run_all.sh [WORKDIR]   (default: tests/decay/runs; needs numpy, scipy, mpi4py, petsc4py and
#        slepc4py with MUMPS; NCPUS ranks (default 2), MPIEXEC overrides 'mpiexec'; about 30 s on 2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/decay_check.py" "$W/decay" || fail=1
echo
[ $fail = 0 ] && echo "tests/decay: all checks passed" || echo "tests/decay: FAILED"
exit $fail
