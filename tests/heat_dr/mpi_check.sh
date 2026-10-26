#!/bin/bash
# ncpus > 1: assembles diff_rot = 1, thermal = 1 runs with mpiexec -n 1, 2, 3, 4 and compares A.npz, B.npz with the
# single-rank ones (sparsity pattern and values). assemble.py needs nb = (lmax - m + 1)/2 divisible by the number of
# ranks, so lmax = m - 1 + 24 (nb = 12). Builds the runs with mk_run.py (test defaults: ricb 0.35, m 1, conical -0.25).
# usage: tests/heat_dr/mpi_check.sh [WORKDIR]   (default: tests/heat_dr/runs; needs mpiexec and mpi4py)
set -e
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"
SUN="par.ricb=0; par.structure='8-term fit'; par.stratification='odd top hat'; par.diffusivity='step'; par.N=64; par.r_out=0.99; par.aux1=0.0; par.aux2=0.1; par.aux3=0.68; par.aux4=0.06; par.aux5=2.6; par.Beyonce=1e3; par.therm0=1.0; par.visc_pow=0.0; par.therm_pow=0.0; par.bco=1"
python3 "$T/mk_run.py" mpi_m1   'par.lmax=24; par.ncpus=4'
python3 "$T/mk_run.py" mpi_nodr 'par.lmax=24; par.ncpus=4; par.diff_rot=0'
python3 "$T/mk_run.py" mpi_m0   'par.m=0; par.lmax=23; par.ncpus=4'
python3 "$T/mk_run.py" mpi_m2a  'par.m=2; par.symm=-1; par.lmax=25; par.ncpus=4'
python3 "$T/mk_run.py" mpi_sun  "$SUN; par.lmax=24; par.ncpus=4"
python3 "$T/mk_run.py" mpi_sun0 "$SUN; par.m=0; par.symm=-1; par.lmax=23; par.ncpus=4"
python3 "$T/mk_run.py" mpi_bc1  'par.lmax=24; par.ncpus=4; par.bci_thermal=1; par.bco_thermal=1'
python3 "$T/mk_run.py" mpi_f10  'par.lmax=24; par.ncpus=4; par.forcing=10; par.forcing_frequency=0.7'
python3 "$T/mk_run.py" mpi_tw2  'par.lmax=24; par.ncpus=4; par.thermal_wind=2; par.Gaspard=3; par.Beyonce=90'
for d in mpi_m1 mpi_nodr mpi_m0 mpi_m2a mpi_sun mpi_sun0 mpi_bc1 mpi_f10 mpi_tw2; do
  for n in 1 2 3 4; do
    (cd $d && mpiexec -n $n python3 bin/assemble.py > log_asm_$n 2>&1 && for f in A B B_forced; do if [ -f $f.npz ]; then mv $f.npz ${f}_$n.npz; fi; done)
  done
  python3 - $d <<'EOF'
import sys, os, numpy as np, scipy.sparse as ss
d = sys.argv[1]
ld = lambda f: (lambda z: ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape']))(np.load(d + '/' + f))
out = []; bad = 0
for X in [X for X in ('A', 'B', 'B_forced') if os.path.isfile('%s/%s_1.npz' % (d, X))]:
    M1 = ld(X + '_1.npz')
    for n in (2, 3, 4):
        Mn = ld('%s_%d.npz' % (X, n))
        pat = abs((M1 != 0).astype(int) - (Mn != 0).astype(int)).sum()   # entries nonzero in one and not the other
        bad += pat > 0 or abs(M1 - Mn).max() > 0
        out += ['%s%d: pattern %d, |diff|max/|M|max %.1e' % (X, n, pat, abs(M1 - Mn).max() / abs(M1).max())]
print(d, 'size', M1.shape[0], 'nnz(A)', ld('A_1.npz').nnz, '|', '; '.join(out), '->', 'FAIL' if bad else 'ok')
EOF
done
echo "--- 5 ranks with nb = 12 must stop with a message (not hang):"
(cd mpi_m1 && timeout 60 mpiexec -n 5 python3 bin/assemble.py 2>&1 | grep -m1 'must be a multiple' || echo 'FAIL: no message (hang or crash)')
