#!/bin/bash
# Runs the checks of the thin conducting walls (README.md).
# usage: tests/twa/run_all.sh [WORKDIR]   (default: tests/twa/runs; needs numpy, scipy, mpi4py, petsc4py and slepc4py
#        with MUMPS; NCPUS ranks (default 2), MPIEXEC overrides 'mpiexec'; about 2 min on 2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/twa_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/twa: all checks passed" || echo "tests/twa: FAILED"
exit $fail
