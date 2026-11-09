#!/bin/bash
# Runs the checks of the resolved conducting mantle (README.md).
# usage: tests/mantle/run_all.sh [WORKDIR]   (default: tests/mantle/runs; needs numpy, scipy, mpi4py, petsc4py and slepc4py
#        with MUMPS, and sympy; NCPUS ranks (default 2), MPIEXEC overrides 'mpiexec'; about 4 min on 2 ranks)
# Exit code 1 if a check failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
python3 "$T/mantle_check.py" "$W" || fail=1
# NEW (2026-11-09): the conducting mantles on the exact spheroid (about 3 min for derive_spheroid, 2.5 min on 8 ranks for the runs)
python3 "$T/derive_spheroid.py" || fail=1
python3 "$T/spheroid_mantle_check.py" "$W/spheroid" "${NCPUS:-2}" || fail=1
echo
[ $fail = 0 ] && echo "tests/mantle: all checks passed" || echo "tests/mantle: FAILED"
exit $fail
