#!/bin/bash
# Runs the checks of the forced problems with a field (README.md).
# usage: tests/forced_field/run_all.sh [WORKDIR]   (default: tests/forced_field/runs; needs numpy, scipy, mpi4py, petsc4py
#        and slepc4py with MUMPS, mpmath for one check; NCPUS ranks (default 2), MPIEXEC overrides 'mpiexec'; about 3 min
#        on 2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/forced_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/forced_field: all checks passed" || echo "tests/forced_field: FAILED"
exit $fail
