#!/bin/bash
# run_example.sh [STEP]: forcing-11 example on cha2, Rossby m 2 window of the fs-dsdr full sphere (sigma 0.004 cut 0.708, Dey nu,
# JSOC rotation, N 896, lmax 33, symm -1). Run from its own directory E (~/lab/rsf11/example), which also holds mkrhs.py and
# mkfreqs.py. Steps (default all): build, eig (background), rhs, grid scans s1 s2 (background), wait eig, resonance scans s3 s4.
# Code: ~/lab/rsf11/kore_rcut/bin (kore_new + rsf_rcut). Env K (realisations per configuration, 16), NPS (ranks per scan, 16).
set -e -o pipefail
source ~/z2/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/repepo/petsc/arch-linux-c-opt/bin:$PATH   # the mpiexec of mpi4py's MPI on cha2
E=$(cd "$(dirname "$0")" && pwd); cd $E
K=${K:-16}; NPS=${NPS:-16}
KB=$HOME/lab/rsf11/kore_rcut/bin; TOOLS=$HOME/lab/fs_dsdr/tools; SHIM=$HOME/lab/rsf11/shim
STEP=${1:-all}; PIDS=()
( while true; do echo "$(date +%s) $(free -g | awk '/Mem:/ {print $3}')" >> mem.log; sleep 30; done ) & MEMPID=$!
trap "kill $MEMPID 2>/dev/null" EXIT
scandir() {   # scandir NAME FREQFILE: a solve.py frequency scan sharing base/'s matrices
  rm -rf $1; mkdir -p $1; cp -r base/bin $1/bin; ln -s $E/base/A.npz $E/base/B.npz $E/base/B_forced.npz $1/
  cp $2 $1/freqs.txt; echo "par.rsf_frequencies = np.loadtxt('freqs.txt')" >> $1/bin/parameters.py
}
if [ $STEP = all ] || [ $STEP = build ]; then
  echo "$(date) build"
  KORE_BIN=$KB python $TOOLS/mkgs.py base N=896 lmax=33 m=2 symm=-1 prof=gs sig=0.004 cut=0.708 NP=8 nev=80 F=-333.0 G=0 \
    "extra=par.forcing = 11; par.forcing_frequency = 0.0; par.rsf_nu_floor = 1e-3; par.rsf_nsamp = $K; par.rsf_seed = 0; par.rsf_theta = 'nu'; par.rsf_split = 0.70/0.985" > build.log 2>&1
  grep -A8 'Random Reynolds' base/log_asm > rsf_base.txt || true
fi
if [ $STEP = all ] || [ $STEP = eig ]; then
  echo "$(date) eigen solve (background)"
  rm -rf eig; mkdir eig; cp -r base/bin eig/bin; ln -s $E/base/A.npz $E/base/B.npz eig/
  echo "par.forcing = 0" >> eig/bin/parameters.py
  ( set +e; cd eig && PYTHONPATH=$SHIM:$PYTHONPATH mpiexec -n 8 python bin/solve.py > log_solve 2>&1 && python bin/spin_doctor.py 8 > log_sd 2>&1 \
      && python $TOOLS/anafs.py $E/eig > ana.txt 2> log_ana; echo "eig rc $?" > done ) & PIDS+=($!)
fi
if [ $STEP = all ] || [ $STEP = rhs ]; then
  echo "$(date) right-hand sides"
  python $E/mkrhs.py base $K > rhs.log 2>&1
  python $E/mkfreqs.py grid g1.txt g2.txt > freqs.log
fi
if [ $STEP = all ] || [ $STEP = grid ]; then
  echo "$(date) grid scans s1 s2"
  for s in 1 2; do scandir s$s g$s.txt; ( set +e; cd s$s; mpiexec -n $NPS python bin/solve.py > log_scan 2>&1; echo "s$s rc $?" > done ) & PIDS+=($!); done
fi
if [ $STEP = all ] || [ $STEP = res ]; then
  echo "$(date) waiting for the eigen solve"
  while [ ! -f eig/done ]; do sleep 60; done; cat eig/done
  python $E/mkfreqs.py res eig res.txt g1.txt g2.txt >> freqs.log
  awk 'NR%2==1' res.txt > r1.txt; awk 'NR%2==0' res.txt > r2.txt
  echo "$(date) resonance scans s3 s4"
  for s in 3 4; do scandir s$s r$((s-2)).txt; ( set +e; cd s$s; mpiexec -n $NPS python bin/solve.py > log_scan 2>&1; echo "s$s rc $?" > done ) & PIDS+=($!); done
fi
[ ${#PIDS[@]} -gt 0 ] && wait ${PIDS[@]}
echo "$(date) finished"
