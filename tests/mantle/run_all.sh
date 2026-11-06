#!/bin/bash
# Runs the checks of the resolved conducting mantle (README.md).
# usage: tests/mantle/run_all.sh [WORKDIR]   (default: tests/mantle/runs; needs numpy, scipy, mpi4py, petsc4py and slepc4py
#        with MUMPS; NCPUS ranks (default 2), MPIEXEC overrides 'mpiexec'; about 1.5 min on 2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/mantle_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/mantle: all checks passed" || echo "tests/mantle: FAILED"
exit $fail
