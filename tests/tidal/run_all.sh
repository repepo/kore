#!/bin/bash
# Builds the test runs of the tidal forcing (forcing = 14) and of the energy budget with wall work, and runs both checks.
# usage: tests/tidal/run_all.sh [WORKDIR]   (default: tests/tidal/runs; needs numpy, scipy, mpi4py, petsc4py and
#        slepc4py with MUMPS; NCPUS ranks (default 2), MPIEXEC overrides 'mpiexec'; about 2 minutes on 4 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/stokes_check.py" "$W/stokes" || fail=1
python3 "$T/budget_check.py" "$W/budget" || fail=1
echo
[ $fail = 0 ] && echo "tests/tidal: all checks passed" || echo "tests/tidal: FAILED"
exit $fail
