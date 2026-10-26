#!/usr/bin/env bash
# regress.sh OLD_KORE [WORKDIR]: builds the diff-rot configurations below (tests/heat_dr/mk_run.py defaults: 'Wu 2005'
# structure, 'top hat' stratification, ricb 0.35, m 1, N 48, lmax 20) with the bin/ of OLD_KORE (an older checkout) and
# with this repo, and checks that A and B are equal: the new code must not change existing runs (default amplitudes and
# switches).
# Parameter names: each configuration is written in the names of three generations of checkouts, and the right one is
# used for each (each checkout's mk_run.py writes the defaults in its own names): with bin/background.py (cfgs: structure,
# stratification, diffusivity, r_out), with bin/diff_rot.py but not bin/background.py (cfgs_dr: model_type, aux0; the
# rotation as cfgs), and older ones (cfgs_old: diff_rot_amplitude, σ for 'shellular' in it). The 'user' configuration is
# built only when OLD_KORE has bin/diff_rot.py.
# Since the heat-equation couplings moved to the sparse recurrence algebra of bin/angular.py, the thermal diff-rot rows
# may differ from older checkouts in the last bits: a run that is not bitwise equal passes as 'round-off' if every entry
# differs by less than 1e-13 times the largest entry of its row (an entry stored in one matrix only counts with its value:
# where a coupling cancels exactly, one formula can give 0 and the other ~1e-17, e.g. P_n'(X) by recurrence or expanded).
set -e -o pipefail
OLD=$(cd "$1" && pwd); NEW=$(cd "$(dirname "$0")/../.." && pwd); W=${2:-$NEW/tests/diff_rot/runs_regress}
mkdir -p "$W"; cd "$W"
names=(con1 con2 y20t3 cyl2 sol1 ric0 td0 shl2 ywb1 ss2 g3 nodr m2s p4 usr)
cfgs=(""
      "par.thermal_wind=2"
      "par.diff_rot_type='Y20'; par.thermal_wind=3; par.dSdtheta=1e-4"
      "par.diff_rot_type='cylindrical'; par.thermal_wind=2"
      "par.diff_rot_type='solar'; par.diff_rot_ampl=1.0"
      "par.ricb=0; par.m=0; par.symm=-1; par.thermal_wind=2; par.lmax=21"
      "par.ThermaD=0; par.thermal_wind=1"
      "par.diff_rot_type='shellular'; par.diff_rot_ampl=1.0; par.diff_rot_param=-0.25; par.thermal_wind=2"
      "par.diff_rot_type='Y20-wall-bounded'"
      "par.structure='8-term fit'; par.stratification='odd top hat'; par.diffusivity='step'; par.diff_rot_type='Y20'; par.thermal_wind=2; par.r_out=0.985; par.ricb=0.72; par.aux1=0.0; par.aux2=0.1; par.aux3=0.68; par.aux4=0.06; par.aux5=0.0"
      "par.Gaspard=3.; par.thermal_wind=2"
      "par.diff_rot=0"
      "par.m=2; par.symm=-1; par.lmax=21; par.thermal_wind=2"
      "par.diff_rot_type='solar_P4'; par.diff_rot_ampl=1.3; par.thermal_wind=2"
      "par.diff_rot_type='user'; par.diff_rot_profiles={0: '(2/3)*r**2*exp(-r)', 2: '-(2/3)*r**2', 4: '0.1*tanh((x - p)/0.05)'}; par.diff_rot_param=0.7; par.thermal_wind=2")
# the same configurations in the names of checkouts without bin/background.py
cfgs_dr=("${cfgs[@]}")
cfgs_dr[9]="par.model_type='smoothed sun'; par.diff_rot_type='Y20'; par.thermal_wind=2; par.aux0=0.985; par.ricb=0.72; par.aux1=0.0; par.aux2=0.1; par.aux3=0.68; par.aux4=0.06; par.aux5=0.0"
# and in those of checkouts without bin/diff_rot.py ('' for those it cannot build)
cfgs_old=("${cfgs_dr[@]}")
cfgs_old[4]="par.diff_rot_type='solar'; par.diff_rot_amplitude=1.0"
cfgs_old[7]="par.diff_rot_type='shellular'; par.thermal_wind=2"
cfgs_old[13]="par.diff_rot_type='solar_P4'; par.diff_rot_amplitude=1.3; par.thermal_wind=2"
cfgs_old[14]=""
if [ -f "$OLD/bin/background.py" ]; then era=bg; elif [ -f "$OLD/bin/diff_rot.py" ]; then era=dr; else era=old; fi
fail=0
for i in "${!names[@]}"; do
  k=${names[$i]}
  case $era in
    bg) co="${cfgs[$i]}" ;;
    dr) co="${cfgs_dr[$i]}" ;;
    old)
      [ -z "${cfgs_old[$i]}" ] && [ $k != con1 ] && { echo "$k: not built (the older checkout lacks it)"; continue; }
      co="${cfgs_old[$i]}" ;;
  esac
  python3 "$OLD/tests/heat_dr/mk_run.py" o_$k "$co" > o_$k.log 2>&1
  python3 "$NEW/tests/heat_dr/mk_run.py" n_$k "${cfgs[$i]}" > n_$k.log 2>&1
  r=$(python3 - o_$k n_$k <<'PY'
import sys, numpy as np, scipy.sparse as ss
def load(f):
    z = np.load(f); return z
verdict, worst = 'ok', 0.0
for M in ('A', 'B'):
    a, b = load(sys.argv[1] + '/%s.npz' % M), load(sys.argv[2] + '/%s.npz' % M)
    if all(np.array_equal(a[x], b[x]) for x in ('data', 'indices', 'indptr', 'shape')):
        continue
    A0, A1 = [ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape']) for z in (a, b)]
    if A0.shape != A1.shape:
        verdict = 'FAIL'; break
    D = abs(A1 - A0).tocsr()   # an entry present in one matrix only counts with its full value
    rowmax = abs(A0).max(axis=1).toarray().ravel()
    rel = (D.multiply(1/np.where(rowmax > 0, rowmax, 1)[:, None])).max()
    worst = max(worst, rel)
    verdict = 'round-off' if worst < 1e-13 else 'FAIL'
print(verdict if verdict != 'round-off' else 'round-off (%.1e of the row scale)' % worst)
PY
)
  echo "$k: A, B bitwise equal: $r"; [ "${r%% *}" = ok ] || [ "${r%% *}" = round-off ] || fail=1
done
[ $fail = 0 ] && echo "regression ok" || { echo "regression FAIL"; exit 1; }
