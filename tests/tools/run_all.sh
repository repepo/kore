#!/bin/bash
# Runs the checks of the files of a run and of the run tools (README.md).
# usage: tests/tools/run_all.sh [WORKDIR]   (default: tests/tools/runs; needs numpy, scipy, mpi4py, petsc4py and slepc4py
#        with MUMPS, pandas for tools/get_data.py and matplotlib for the plotting tools; one rank, MPIEXEC overrides
#        'mpiexec'; about 2 min)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/tools_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/tools: all checks passed" || echo "tests/tools: FAILED"
exit $fail
