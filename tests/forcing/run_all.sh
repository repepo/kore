#!/bin/bash
# Runs the checks of the boundary and body forcings (README.md).
# usage: tests/forcing/run_all.sh [WORKDIR]   (default: tests/forcing/runs; needs numpy, scipy, mpi4py, petsc4py and
#        slepc4py with MUMPS; NCPUS ranks (default 2), MPIEXEC overrides 'mpiexec')
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/rows_check.py" "$W/rows" || fail=1
echo
[ $fail = 0 ] && echo "tests/forcing: all checks passed" || echo "tests/forcing: FAILED"
exit $fail
