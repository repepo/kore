#!/bin/bash
# Runs the checks of the composition (compositional = 1) and of the 'two zone' gradient (README.md).
# usage: tests/composition/run_all.sh [WORKDIR]   (default: tests/composition/runs; needs numpy, scipy, mpi4py, petsc4py
#        and slepc4py with MUMPS; NCPUS ranks per run (default 2, at most 2 are used), NPAR runs at a time (default half
#        the processors, at most 4), MPIEXEC overrides 'mpiexec'; about 3.5 min on 4 cores)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/composition_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/composition: all checks passed" || echo "tests/composition: FAILED"
exit $fail
