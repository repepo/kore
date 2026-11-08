#!/usr/bin/env bash
# regress.sh OLD_KORE [WORKDIR]
#
# Did a change of the code alter the results of runs that already worked? This script answers that for the matrices: it
# builds the same small runs twice, with an older copy of kore and with this one, and compares their matrices A and B
# (A.npz, B.npz of assemble.py) entry by entry. Equal matrices mean equal eigenvalues and eigenvectors.
#
#   OLD_KORE  a directory with an older version of the repository, usually the last commit before the change:
#                 mkdir -p /tmp/kore_old && git archive <commit> | tar -x -C /tmp/kore_old
#   WORKDIR   where the runs are built (default: tests/diff_rot/runs_regress). Each run has two directories there,
#             o_NAME (built with OLD_KORE) and n_NAME (built with this repository), with their logs.
#
# Each run prints one line:
#   ok         A and B are bitwise equal
#   round-off  they differ only in the last bits: every entry by less than 1e-13 of the largest entry of its row
#   skipped    the older copy cannot build this run (it predates a feature the run uses); not counted as a failure
#   FAIL       anything else, including a run that does not build
# and the script ends with 'regression ok' (exit 0) or 'regression FAIL' (exit 1). A change that is meant to alter
# results should fail in the runs it concerns, and only there.
#
# The runs are small (N 48, lmax 20 or 21, one rank, a few seconds each) and start from the defaults of
# tests/heat_dr/mk_run.py: structure 'Wu 2005' with stratification 'top hat', thermal, ricb 0.35, m 1, a 'conical'
# differential rotation with thermal_wind 1. Each changes a few parameters (the lists below):
#   - 15 runs of the differential rotation and the heat equation: every rotation profile type, thermal_wind 0 to 3,
#     a full sphere, ThermaD = 0, a solar structure, Gaspard 3, m 2, and no rotation at all;
#   - 11 runs of the magnetic field: the poloidal types (axial, FDM, Luo_S2, confined, general, ferraro), a toroidal
#     field, a conducting inner core, a full sphere, the induction by the rotation, the wound field, diff_rot_type
#     'ferraro' and the couplings from mag_algebra.py.
#
# Older copies spell some parameters differently. Each rotation run is therefore also written in the names of the two
# older generations, and the older copy gets the spelling it reads: checkouts without bin/background.py use model_type and
# aux0, those without bin/diff_rot.py diff_rot_amplitude (with σ of 'shellular' as that amplitude). A magnetic run names
# the code it needs (a pattern in one file of bin/) and is skipped when the older copy does not have it.
#
# Why 'round-off' is accepted: the couplings of the heat equation moved to the recurrence algebra of bin/angular.py, so
# its rotation rows can differ in the last bits from checkouts older than that. An entry present in only one of the two
# matrices counts with its full value (where a coupling cancels exactly, one formula gives 0 and the other ~1e-17).
set -e -o pipefail
OLD=$(cd "$1" && pwd); NEW=$(cd "$(dirname "$0")/../.." && pwd); W=${2:-$NEW/tests/diff_rot/runs_regress}
mkdir -p "$W"; cd "$W"

# ------------------------------------------------------------------- the runs of the differential rotation (name, change)
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
# the same runs in the names of checkouts without bin/background.py
cfgs_dr=("${cfgs[@]}")
cfgs_dr[9]="par.model_type='smoothed sun'; par.diff_rot_type='Y20'; par.thermal_wind=2; par.aux0=0.985; par.ricb=0.72; par.aux1=0.0; par.aux2=0.1; par.aux3=0.68; par.aux4=0.06; par.aux5=0.0"
# and in those of checkouts without bin/diff_rot.py ('' for the run they cannot build)
cfgs_old=("${cfgs_dr[@]}")
cfgs_old[4]="par.diff_rot_type='solar'; par.diff_rot_amplitude=1.0"
cfgs_old[7]="par.diff_rot_type='shellular'; par.thermal_wind=2"
cfgs_old[13]="par.diff_rot_type='solar_P4'; par.diff_rot_amplitude=1.3; par.thermal_wind=2"
cfgs_old[14]=""

# -------------------------------------------------------- the runs of the magnetic field (name, change, code they need)
# (Le² = Em = 1e-3; the rotation of the defaults is kept only where the run is about it; no-slip walls where the toroidal
# field is not 0 at a wall, which stopped a slipping wall before phase 6c. The stress-free walls of m_gen (outer), m_fer
# and m_fdr (both) hold the push of the poloidal B0's current sheet there since phase 6c (2026-11-07, background.sheet_jump):
# against an older checkout these three FAIL, as intended: the stress rows of those walls gain entries on f, and A and B,
# divided by the norm of A, change by 1e-5 elsewhere. m_wnd FAILs too, by 8e-13 of the row scale (its eigenvalues by 4e-12):
# phase 6c cuts the round-off of the wound field's profiles at 1e-13)
MAG="par.magnetic=1; par.B0='axial'; par.cnorm='mag_energy'; par.Hendrik=1e-3; par.MagnetD=1e-3"
GEN="par.B0='general'; par.B0_profiles={1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}"
DRU="par.diff_rot_type='user'; par.diff_rot_ampl=0.2; par.diff_rot_profiles={0: '(2/3)*r**2 - (8/15)*r**4', 2: '-(2/3)*r**2 + (16/21)*r**4', 4: '-(8/35)*r**4'}"
mnames=(m_ax m_cic m_tor m_fs m_conf m_gen m_alg m_dr m_fer m_fdr m_wnd)
mcfgs=("$MAG; par.diff_rot=0"
       "$MAG; par.diff_rot=0; par.B0='FDM'; par.innercore='conducting, Chebys'; par.bci=1"
       "$MAG; par.diff_rot=0; par.B0_tor='Malkus'; par.B0_tor_l=2; par.m=2; par.symm=1; par.lmax=21; par.bci=1; par.bco=1"
       "$MAG; par.diff_rot=0; par.ricb=0; par.B0='Luo_S2'"
       "$MAG; par.diff_rot=0; par.B0='confined'; par.B0_l=2; par.B0_conf_r1=0.4; par.B0_conf_r2=0.8"
       "$MAG; par.diff_rot=0; $GEN; par.B0_tor='general'; par.B0_tor_profiles={2: 'r**2*(1 - r**2)'}; par.cnorm_tor='mag_energy'; par.bci=1"
       "$MAG; par.diff_rot=0; par.mag_couplings='algebra'; par.B0='Luo_S2'; par.innercore='conducting, Chebys'; par.bci=1"
       "$MAG"
       "$MAG; $DRU; par.B0='ferraro'; par.B0_ferraro_nmax=7"
       "$MAG; $GEN; par.diff_rot_type='ferraro'; par.diff_rot_F='psi + psi**2'; par.diff_rot_ampl=0.05; par.diff_rot_nmax=8"
       "$MAG; par.B0_wound=1")
mneeds=("bin/operators.py:def lorentz_rho:magnetic field with a variable density"   # (file:pattern:what it is)
        "bin/operators.py:def lorentz_rho:magnetic field with a variable density"
        "bin/background_field.py:B0_TOR_TYPES:toroidal background field"
        "bin/background_field.py:Luo_S2:magnetic field in a full sphere"
        "bin/mag_algebra.py:def block:B0 'confined', 'general' or 'ferraro'"
        "bin/mag_algebra.py:def block:B0 'confined', 'general' or 'ferraro'"
        "bin/mag_algebra.py:def block:mag_couplings 'algebra'"
        "bin/diff_rot.py:def induction_block:induction by the differential rotation"
        "bin/background_field.py:def _ferraro_psi:B0 'ferraro'"
        "bin/diff_rot.py:def _ferraro_shapes:diff_rot_type 'ferraro'"
        "bin/background_field.py:def wound_profiles:wound field, B0_wound")

# ---------------------------------------------------------------------------------------------------------- comparison
compare() {   # compare O N: the verdict for the A, B of the run directories O and N
python3 - "$1" "$2" <<'PY'
import sys, numpy as np, scipy.sparse as ss
verdict, worst = 'ok', 0.0
for M in ('A', 'B'):
    a, b = np.load(sys.argv[1] + '/%s.npz' % M), np.load(sys.argv[2] + '/%s.npz' % M)
    if all(np.array_equal(a[x], b[x]) for x in ('data', 'indices', 'indptr', 'shape')):
        continue
    A0, A1 = [ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape']) for z in (a, b)]
    if A0.shape != A1.shape:
        verdict = 'FAIL (A or B changed size)'; break
    D = abs(A1 - A0).tocsr()   # an entry present in one matrix only counts with its full value
    rowmax = abs(A0).max(axis=1).toarray().ravel()
    worst = max(worst, (D.multiply(1/np.where(rowmax > 0, rowmax, 1)[:, None])).max())
    verdict = 'round-off' if worst < 1e-13 else 'FAIL'
print(verdict if verdict == 'ok' or verdict.startswith('FAIL (') else '%s (%.1e of the row scale)' % (verdict, worst))
PY
}

fail=0; nok=0; nskip=0
run() {   # run NAME OLD_CHANGE NEW_CHANGE: builds both runs and prints the verdict
  local k=$1
  if ! python3 "$OLD/tests/heat_dr/mk_run.py" o_$k "$2" > o_$k.log 2>&1; then
    echo "$k: FAIL (the older copy could not build it: o_$k/log_sub, o_$k/log_asm)"; fail=1; return
  fi
  if ! python3 "$NEW/tests/heat_dr/mk_run.py" n_$k "$3" > n_$k.log 2>&1; then
    echo "$k: FAIL (this copy could not build it: n_$k/log_sub, n_$k/log_asm)"; fail=1; return
  fi
  local r; r=$(compare o_$k n_$k)
  echo "$k: $r"
  case $r in ok*|round-off*) nok=$((nok + 1)) ;; *) fail=1 ;; esac
}

if [ -f "$OLD/bin/background.py" ]; then era=bg; elif [ -f "$OLD/bin/diff_rot.py" ]; then era=dr; else era=old; fi
for i in "${!names[@]}"; do
  k=${names[$i]}
  case $era in
    bg) co="${cfgs[$i]}" ;;
    dr) co="${cfgs_dr[$i]}" ;;
    old)
      if [ -z "${cfgs_old[$i]}" ] && [ $k != con1 ]; then echo "$k: skipped (the older copy predates it)"; nskip=$((nskip + 1)); continue; fi
      co="${cfgs_old[$i]}" ;;
  esac
  run $k "$co" "${cfgs[$i]}"
done
for i in "${!mnames[@]}"; do
  k=${mnames[$i]}; f=${mneeds[$i]%%:*}; rest=${mneeds[$i]#*:}; pat=${rest%%:*}; what=${rest#*:}
  if ! grep -q "$pat" "$OLD/$f" 2>/dev/null; then echo "$k: skipped (the older copy has no $what)"; nskip=$((nskip + 1)); continue; fi
  run $k "${mcfgs[$i]}" "${mcfgs[$i]}"
done
if [ $fail = 0 ]; then echo "regression ok ($nok compared, $nskip skipped)"; else echo "regression FAIL"; exit 1; fi
