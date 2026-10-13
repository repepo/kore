#!/usr/bin/env bash
# regress.sh OLD_KORE [WORKDIR]: builds 13 diff-rot configurations (tests/heat_dr/mk_run.py defaults: user-def profiles,
# ricb 0.35, m 1, N 48, lmax 20) with the bin/ of OLD_KORE (a checkout before the generic-Pn change) and with this repo,
# and checks that A and B are bitwise equal: the new code must not change existing runs (default amplitudes and switches).
set -e -o pipefail
OLD=$(cd "$1" && pwd); NEW=$(cd "$(dirname "$0")/../.." && pwd); W=${2:-$NEW/tests/diff_rot/runs_regress}
mkdir -p "$W"; cd "$W"
names=(con1 con2 y20t3 cyl2 sol1 ric0 td0 shl2 ywb1 ss2 g3 nodr m2s)
cfgs=(""
      "par.thermal_wind=2"
      "par.diff_rot_type='Y20'; par.thermal_wind=3; par.dSdtheta=1e-4"
      "par.diff_rot_type='cylindrical'; par.thermal_wind=2"
      "par.diff_rot_type='solar'; par.diff_rot_amplitude=1.0"
      "par.ricb=0; par.m=0; par.symm=-1; par.thermal_wind=2; par.lmax=21"
      "par.ThermaD=0; par.thermal_wind=1"
      "par.diff_rot_type='shellular'; par.thermal_wind=2"
      "par.diff_rot_type='Y20-wall-bounded'"
      "par.model_type='smoothed sun'; par.diff_rot_type='Y20'; par.thermal_wind=2; par.aux0=0.985; par.ricb=0.72; par.aux1=0.0; par.aux2=0.1; par.aux3=0.68; par.aux4=0.06; par.aux5=0.0"
      "par.Gaspard=3.; par.thermal_wind=2"
      "par.diff_rot=0"
      "par.m=2; par.symm=-1; par.lmax=21; par.thermal_wind=2")
fail=0
for i in "${!names[@]}"; do
  k=${names[$i]}
  python3 "$OLD/tests/heat_dr/mk_run.py" o_$k "${cfgs[$i]}" > o_$k.log 2>&1
  python3 "$NEW/tests/heat_dr/mk_run.py" n_$k "${cfgs[$i]}" > n_$k.log 2>&1
  r=$(python3 - o_$k n_$k <<'PY'
import sys, numpy as np
def load(f):
    z = np.load(f); return z
d = 0.0
for M in ('A', 'B'):
    a, b = load(sys.argv[1] + '/%s.npz' % M), load(sys.argv[2] + '/%s.npz' % M)
    same = all(np.array_equal(a[x], b[x]) for x in ('data', 'indices', 'indptr', 'shape'))
    d = max(d, 0.0 if same else 1.0)
print('ok' if d == 0 else 'FAIL')
PY
)
  echo "$k: A, B bitwise equal: $r"; [ "$r" = ok ] || fail=1
done
[ $fail = 0 ] && echo "regression ok" || { echo "regression FAIL"; exit 1; }
