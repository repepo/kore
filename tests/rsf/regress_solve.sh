#!/usr/bin/env bash
# regress_solve.sh OLD_KORE [WORKDIR]: an eigenvalue run (mk_run.py defaults: thermal, conical diff rot) and a forced run
# (forcing 10) built and solved with the bin/ of OLD_KORE (a checkout before forcing 11) and with this repo, one rank,
# one thread; A, B, B_forced, eigenvalues and fields must be bitwise equal (solve.py now loops over the columns of
# B_forced, assemble.py writes B for forcing 11). Needs petsc4py/slepc4py with MUMPS; MPIEXEC overrides 'mpiexec'.
set -e -o pipefail
OLD=$(cd "$1" && pwd); NEW=$(cd "$(dirname "$0")/../.." && pwd); W=${2:-$NEW/tests/rsf/runs_regress}
MPI=${MPIEXEC:-mpiexec}
export OMP_NUM_THREADS=1
mkdir -p "$W"; cd "$W"
names=(eig f10)
cfgs=("par.nev=10" "par.forcing=10; par.forcing_frequency=0.7")
fail=0
for i in "${!names[@]}"; do
  k=${names[$i]}
  for v in o n; do
    K=$([ $v = o ] && echo "$OLD" || echo "$NEW")
    python3 "$K/tests/heat_dr/mk_run.py" ${v}_$k "${cfgs[$i]}" > ${v}_$k.log 2>&1
    (cd ${v}_$k && $MPI -n 1 python3 bin/solve.py > log_solve 2>&1)
  done
  r=$(python3 - o_$k n_$k <<'PY'
import sys, os, filecmp, numpy as np
o, n = sys.argv[1], sys.argv[2]
bad = []
for M in ('A', 'B', 'B_forced'):
    if os.path.exists(o + '/%s.npz' % M) or os.path.exists(n + '/%s.npz' % M):
        a, b = np.load(o + '/%s.npz' % M), np.load(n + '/%s.npz' % M)
        if not all(np.array_equal(a[x], b[x]) for x in ('data', 'indices', 'indptr', 'shape')):
            bad.append(M)
outs = sorted(f for f in os.listdir(o) if f.endswith('.field') or f == 'eigenvalues0.dat')
if not outs:
    bad.append('no solution written')
for f in outs:
    if not filecmp.cmp(o + '/' + f, n + '/' + f, shallow=False):
        bad.append(f)
print('ok (%s)' % ', '.join(outs) if not bad else 'FAIL: ' + ', '.join(bad))
PY
)
  echo "$k: matrices and solutions bitwise equal: $r"; [ "${r:0:2}" = ok ] || fail=1
done
exit $fail
