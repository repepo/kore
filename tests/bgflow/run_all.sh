#!/bin/bash
# Runs the checks of the background flow, bgflow = 1 (README.md).
# usage: tests/bgflow/run_all.sh [WORKDIR]   (default: tests/bgflow/runs; needs numpy, scipy, mpi4py, petsc4py and slepc4py
#        with MUMPS; NCPUS ranks (default 2, at most 2 are used), MPIEXEC overrides 'mpiexec'; about 8.5 min of runs on 2
#        ranks, which (NEW 2026-11-09) run side by side, NPAR at a time: by default the processors over the ranks of a run,
#        at most one run per 2 GB of available memory)
# Exit code 1 if a check failed.
# Open MPI binds the ranks of every mpiexec to the same first cores: runs side by side would share them. With more ranks
# than cores, a rank that waits gives its core away instead of polling
export OMPI_MCA_hwloc_base_binding_policy=${OMPI_MCA_hwloc_base_binding_policy:-none}
export OMPI_MCA_mpi_yield_when_idle=${OMPI_MCA_mpi_yield_when_idle:-1}
# and with jobs side by side, one thread per process: numpy's BLAS would otherwise start a thread per core in each
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"
fail=0
# NEW (2026-11-09): the check runs in the background, so that an INT or TERM of this script reaches it at once (it then
# stops its runs); bash would run the trap of a command in the foreground only once that command had ended
python3 "$T/bgflow_check.py" "$W" & p=$!
trap 'kill -TERM $p 2> /dev/null; exit 130' INT TERM
wait $p || fail=1
echo
[ $fail = 0 ] && echo "tests/bgflow: all checks passed" || echo "tests/bgflow: FAILED"
exit $fail
