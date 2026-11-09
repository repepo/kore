#!/bin/bash
# Runs the checks of the drag at the outer wall (README.md).
# usage: tests/drag/run_all.sh [WORKDIR]   (default: tests/drag/runs; needs numpy, scipy, mpi4py, petsc4py and slepc4py
#        with MUMPS; NCPUS ranks (default 2, at most 2 are used), MPIEXEC overrides 'mpiexec'; about 3 min on 2 ranks)
# NEW (2026-11-09): drag_check.py and deformed_check.py (the deformed body, whose one-rank runs are built NPAR at a time:
# by default the processors, at most one per 2 GB of available memory) run side by side, each with its log in
# WORKDIR/logs, printed once both have ended (2 min for the first on 2 ranks, 1 min for the second on 4 processors).
# Exit code 1 if a check failed.
# Open MPI binds the ranks of every mpiexec to the same first cores: runs side by side would share them. With more ranks
# than cores, a rank that waits gives its core away instead of polling
export OMPI_MCA_hwloc_base_binding_policy=${OMPI_MCA_hwloc_base_binding_policy:-none}
export OMPI_MCA_mpi_yield_when_idle=${OMPI_MCA_mpi_yield_when_idle:-1}
# and with jobs side by side, one thread per process: numpy's BLAS would otherwise start a thread per core in each
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W/logs"; W=$(cd "$W" && pwd)
# on INT or TERM, every process the checks started (a kill of the checks alone would leave their runs going)
ptree() { local p; echo "$1"; for p in $(pgrep -P "$1"); do ptree "$p"; done; }
trap 'kill $(for j in $(jobs -p); do ptree $j; done) 2> /dev/null; exit 130' INT TERM
python3 "$T/drag_check.py" "$W" > "$W/logs/drag_check.log" 2>&1 & p1=$!
python3 "$T/deformed_check.py" "$W" > "$W/logs/deformed_check.log" 2>&1 & p2=$!
fail=0
wait $p1 || fail=1
wait $p2 || fail=1
cat "$W/logs/drag_check.log"; echo; cat "$W/logs/deformed_check.log"
echo
[ $fail = 0 ] && echo "tests/drag: all checks passed" || echo "tests/drag: FAILED"
exit $fail
