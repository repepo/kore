#!/bin/bash
# Runs the checks of the drag at the outer wall (README.md).
# usage: tests/drag/run_all.sh [WORKDIR]   (default: tests/drag/runs; needs numpy, scipy, mpi4py, petsc4py and slepc4py
#        with MUMPS; NCPUS ranks (default 2, at most 2 are used), MPIEXEC overrides 'mpiexec'; about 3 min on 2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/drag_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/drag: all checks passed" || echo "tests/drag: FAILED"
exit $fail
