#!/usr/bin/env bash
# regress_solve.sh OLD_KORE [WORKDIR]
#
# The companion of tests/diff_rot/regress.sh for solve.py: it builds and solves two runs with an older copy of kore
# (OLD_KORE, e.g. made with 'git archive <commit> | tar -x -C OLD_KORE') and with this one, and requires the matrices
# (A, B, B_forced), the eigenvalues and the fields to be bitwise equal. The two runs start from the defaults of
# tests/heat_dr/mk_run.py (thermal, a 'conical' rotation, N 48): an eigenvalue run (nev 10) and a forced run (forcing 10).
# Both solve on one rank with one thread, so that the results are reproducible to the last bit.
# It was written when forcing 11 arrived (solve.py then loops over the columns of B_forced, and assemble.py writes B for
# forcing 11), and works against any copy that has forcing 10. Needs petsc4py and slepc4py with MUMPS; MPIEXEC overrides
# 'mpiexec'. Each run prints 'ok' with the files compared, or 'FAIL' with those that differ; the exit status is 1 on a FAIL.
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
