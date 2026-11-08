#!/bin/bash
# Runs the checks of forcings 1, 2 and 8, the tides and the libration of the conductive_IC branch (README.md).
# usage: tests/cic_forcings/run_all.sh [WORKDIR]   (default: tests/cic_forcings/runs; needs numpy, scipy, mpi4py, petsc4py
#        and slepc4py with MUMPS; NCPUS ranks (default 2, at most 2 are used), MPIEXEC overrides 'mpiexec'; about 2.5 min on
#        2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/cic_forcings_check.py" "$W" || fail=1
echo
[ $fail = 0 ] && echo "tests/cic_forcings: all checks passed" || echo "tests/cic_forcings: FAILED"
exit $fail
