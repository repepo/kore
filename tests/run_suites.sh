#!/bin/bash
# NEW (2026-11-09): runs test suites side by side, each in its own folder with its own log, the longest first, and says
# which passed.
# usage: tests/run_suites.sh [WORKDIR [SUITE ...]]
#   WORKDIR  each suite runs in WORKDIR/SUITE, its output in WORKDIR/logs/SUITE.log (default: tests/runs_suites)
#   SUITE    folders of tests/ (default: all of them), each run by its run_all.sh; 'ci' runs pytest in tests/ci (or,
#            without pytest, its test function), and
#            'diff_rot' and 'rsf_solve' the regression scripts tests/diff_rot/regress.sh and tests/rsf/regress_solve.sh,
#            which compare with OLD_KORE (a copy of the commit before the change: git archive <commit> | tar -x -C
#            $OLD_KORE) and are skipped without it.
# JOBS suites run at a time (default: all those given). The magnetic and centrifugal suites run their own checks side by
# side, MAG_NPAR and CF_NPAR at a time: by default half and a quarter of the processors when both are given, three
# quarters for one alone, and at most one check per 6 GB of available memory; the sheet, bgflow and heat_dr suites build
# their runs side by side, OTHER_NPAR at a time (default: a quarter of the processors, at most one per 6 GB). To split
# the suites between two machines, give each machine its list (docs/checking.md). Exit code 1 if a suite failed.
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs_suites}; [ $# -gt 0 ] && shift
mkdir -p "$W/logs"; W=$(cd "$W" && pwd)
# the suites, the longest first (mongo, 2026-11-08: magnetic 3.2 h before its checks ran side by side, centrifugal 23 min,
# the others below 7 min)
ALL="magnetic centrifugal sheet bgflow heat_dr composition rsf forced_field diff_rot cic_forcings tidal drag tools couette
     inner_core mantle twa rotdyn forcing decay rsf_solve ci"
ALL=$(echo $ALL)   # (one space between the names, which the tests " $ALL " below need)
SUITES=$(echo ${*:-$ALL})
for s in $SUITES; do
  case $s in
    */*)                   echo "tests/run_suites.sh: no suite '$s' (give the folder's name alone)"; exit 2;;
    ci|diff_rot|rsf_solve) ;;
    *)                     [ -f "$T/$s/run_all.sh" ] || { echo "tests/run_suites.sh: no suite '$s'"; exit 2; };;
  esac
done
ORDERED=$( (for s in $ALL; do case " $SUITES " in *" $s "*) echo $s;; esac; done
            for s in $SUITES; do case " $ALL " in *" $s "*) ;; *) echo $s;; esac; done) | awk '!seen[$0]++')
# the processors this process may use (sched_getaffinity: a taskset or a batch system can give fewer than the machine has)
NCPU=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 2)
MEMG=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo 2> /dev/null)   # (empty where unknown: no cap)
cap() { local n=$1; [ -n "$MEMG" ] && [ $((MEMG/6)) -lt "$n" ] && n=$((MEMG/6)); [ "$n" -ge 1 ] && echo $n || echo 1; }
both=0; case " $SUITES " in *" magnetic "*) case " $SUITES " in *" centrifugal "*) both=1;; esac;; esac
MAG_NPAR=${MAG_NPAR:-$(cap $([ $both = 1 ] && echo $((NCPU/2)) || echo $((3*NCPU/4))))}
CF_NPAR=${CF_NPAR:-$(cap $([ $both = 1 ] && echo $((NCPU/4)) || echo $((3*NCPU/4))))}
OTHER_NPAR=${OTHER_NPAR:-$(cap $((NCPU/4)))}
JOBS=${JOBS:-$(echo $ORDERED | wc -w)}
# Open MPI binds the ranks of every mpiexec to the same first cores: suites side by side would share them. With more
# ranks than cores, a rank that waits gives its core away instead of polling
export OMPI_MCA_hwloc_base_binding_policy=${OMPI_MCA_hwloc_base_binding_policy:-none}
export OMPI_MCA_mpi_yield_when_idle=${OMPI_MCA_mpi_yield_when_idle:-1}
# and with jobs side by side, one thread per process: numpy's BLAS would otherwise start a thread per core in each
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# on INT or TERM, every process the suites started (a kill of the jobs alone would leave their children running)
ptree() { local p; echo "$1"; for p in $(pgrep -P "$1"); do ptree "$p"; done; }
trap 'kill $(for j in $(jobs -p); do ptree $j; done) 2> /dev/null; exit 130' INT TERM

run() {   # run SUITE: its output in logs/SUITE.log, its exit code and seconds in logs/SUITE.rc
  local s=$1 t0=$SECONDS rc
  case $s in
    ci)          if python3 -c 'import pytest' 2> /dev/null; then (cd "$T/ci" && python3 -m pytest -v -p no:cacheprovider)
                 else   # (without pytest: the test function itself, in a fresh folder)
                   mkdir -p "$W/ci" && (cd "$T/ci" && python3 -c "import pathlib, tempfile, test_spinover as t
d = pathlib.Path(tempfile.mkdtemp(dir='$W/ci')); t.test_spinover(d); print('tests/ci: test_spinover passed (without pytest)')")
                 fi;;
    diff_rot)    bash "$T/diff_rot/regress.sh" "$OLD_KORE" "$W/$s";;
    rsf_solve)   bash "$T/rsf/regress_solve.sh" "$OLD_KORE" "$W/$s";;
    magnetic)    NPAR=$MAG_NPAR bash "$T/magnetic/run_all.sh" "$W/$s";;
    centrifugal) OLD_KORE= NPAR=$CF_NPAR bash "$T/centrifugal/run_all.sh" "$W/$s";;   # (regress.sh runs once: diff_rot)
    *)           NPAR=$OTHER_NPAR bash "$T/$s/run_all.sh" "$W/$s";;
  esac > "$W/logs/$s.log" 2>&1
  rc=$?
  echo "$rc $((SECONDS - t0))" > "$W/logs/$s.rc"
  printf '%-12s %-7s %6d s\n' "$s" "$([ $rc = 0 ] && echo ok || echo FAILED)" $((SECONDS - t0))
}

echo "tests/run_suites.sh: $(echo $ORDERED | wc -w) suites in $W, $JOBS at a time, on $NCPU processors" \
     "(magnetic $MAG_NPAR, centrifugal $CF_NPAR checks at a time, the others $OTHER_NPAR runs)"
rm -f "$W"/logs/*.rc
T0=$SECONDS
for s in $ORDERED; do
  case $s in diff_rot|rsf_solve) [ -z "$OLD_KORE" ] && { printf '%-12s skipped (no OLD_KORE)\n' $s; continue; };; esac
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
  run $s &
done
wait
fail=0; n=0
for s in $ORDERED; do   # (a suite that left no exit code failed too)
  case $s in diff_rot|rsf_solve) [ -z "$OLD_KORE" ] && continue;; esac
  n=$((n + 1)); rc=none; [ -f "$W/logs/$s.rc" ] && read rc t < "$W/logs/$s.rc"
  [ "$rc" = 0 ] || { fail=1; echo "FAILED: $s (exit $rc), its log: $W/logs/$s.log"; }
done
echo "tests/run_suites.sh: $n suites in $((SECONDS - T0)) s, $([ $fail = 0 ] && echo "all passed" || echo FAILED)"
exit $fail
