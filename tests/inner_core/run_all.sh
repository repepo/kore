#!/bin/bash
# Runs the checks of the inner core's field: B0 'dipole' and a fluid slipping along a conducting inner core (README.md).
# usage: tests/inner_core/run_all.sh [WORKDIR]   (default: tests/inner_core/runs; needs numpy, scipy, mpi4py, petsc4py and
#        slepc4py with MUMPS; NCPUS caps the ranks of each run, at most 2 (default 2), MPIEXEC overrides 'mpiexec';
#        about 1.5 min on 2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/core_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/inner_core: all checks passed" || echo "tests/inner_core: FAILED"
exit $fail
