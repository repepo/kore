#!/bin/bash
# Runs the checks of the push of B0's current sheet on a fluid slipping along a wall (README.md).
# usage: tests/sheet/run_all.sh [WORKDIR]   (default: tests/sheet/runs; needs numpy, scipy, mpi4py, petsc4py and slepc4py
#        with MUMPS; NCPUS caps the ranks of each run, at most 2 (default 2), MPIEXEC overrides 'mpiexec'; about 11 min on
#        2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/sheet_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/sheet: all checks passed" || echo "tests/sheet: FAILED"
exit $fail
