#!/bin/bash
# Runs the checks of forcing 12, the steady rotation of the walls (README.md).
# usage: tests/couette/run_all.sh [WORKDIR]   (default: tests/couette/runs; needs numpy, scipy, mpi4py, petsc4py and
#        slepc4py with MUMPS; NCPUS ranks (default 2, at most 2 are used), MPIEXEC overrides 'mpiexec'; about 1.5 min
#        on 2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/couette_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/couette: all checks passed" || echo "tests/couette: FAILED"
exit $fail
