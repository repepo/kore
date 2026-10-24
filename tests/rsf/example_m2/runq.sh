#!/bin/bash
# runq.sh NAME [NPS]: a scan worker for the forcing-11 example. Takes frequency chunks (files of w values, kore units) from
# q/todo in name order until none is left, and solves each in its own directory s_<chunk> that shares base/'s matrices, like
# run_example.sh's scans. A chunk is claimed by an atomic mv, so several workers can share the queue; start one per free
# 16 cores. SOLVE overrides the solve command (tests).
set -e -o pipefail
source ~/z2/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/repepo/petsc/arch-linux-c-opt/bin:$PATH
E=$(cd "$(dirname "$0")" && pwd); cd $E
NAME=$1; NPS=${2:-16}; SOLVE=${SOLVE:-"mpiexec -n $NPS python bin/solve.py"}
mkdir -p q/todo q/taken q/done
while true; do
  c=$(ls q/todo | sort | head -1)
  [ -z "$c" ] && break
  mv q/todo/$c q/taken/$c 2>/dev/null || continue   # another worker took it first
  d=s_$c; rm -rf $d; mkdir -p $d; cp -r base/bin $d/bin; ln -s $E/base/A.npz $E/base/B.npz $E/base/B_forced.npz $d/
  cp q/taken/$c $d/freqs.txt; echo "par.rsf_frequencies = np.loadtxt('freqs.txt')" >> $d/bin/parameters.py
  echo "$(date) $NAME: $c ($(wc -l < $d/freqs.txt) frequencies)"
  ( set +e; cd $d; $SOLVE > log_scan 2>&1; echo "$d rc $? $NAME" > done )
  mv q/taken/$c q/done/$c
done
echo "$(date) $NAME: queue empty"
