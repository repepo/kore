#!/bin/bash
# Checks of the centrifugal deformation (centrifugal = 1; docs/centrifugal-equations.md, Checks). NEW (2026-11-05).
# usage: tests/centrifugal/run_all.sh [WORKDIR]   (default tests/centrifugal/runs; about 5 minutes)
# Needs numpy, scipy, sympy and mpi4py; the budgets also petsc4py and slepc4py (SKIP_SOLVE=1 skips them). With OLD_KORE set to
# a copy of the commit before the change (git archive <commit> | tar -x -C $OLD_KORE), also tests/diff_rot/regress.sh.
# Each check prints its verdict; the script ends with 'centrifugal checks ok' (exit 0) or lists the ones that failed (exit 1).
set -o pipefail
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"
failed=()
check() { local name=$1; shift; if ! "$@"; then failed+=("$name"); fi; }

echo "--- derive.py: the closed forms of bin/centrifugal.py from X = α + ξ, w = J F⁻¹ u (sympy)"
check derive bash -c "python3 '$T/derive.py' | tail -1"
echo "--- algebra_check.py: the angular couplings (angular.py's algebra) against a direct projection of the closed forms"
check algebra bash -c "python3 '$T/algebra_check.py' | tail -1"
echo "--- clairaut_check.py: the 'clairaut' flattening (Maclaurin 5/4, polytrope 7.5/π², Wu 2005 against Clairaut from ρ, the Sun)"
check clairaut python3 "$T/clairaut_check.py"
echo "--- eig_check.py: f = 0, the spin-over 1 + f, the relabelling test, homogeneous 'clairaut', the monopole, the ideal fluid"
check eig python3 "$T/eig_check.py"

echo "--- ru_insane.py: the metric blocks against (r, θ) quadratures, Hermitian and positive, in shells and full spheres"
WU="par.structure = 'Wu 2005'; par.r_out = 0.99; par.thermal = 1; par.stratification = 'top hat'; par.aux1 = 0.5; par.aux2 = 0.03; par.aux3 = 0.85; par.aux4 = 0.03; par.aux5 = 2.5; par.Beyonce = 10."
CL="par.centrifugal_shape = 'clairaut'; par.centrifugal_f = None; par.centrifugal_q = 0.05"
US="par.centrifugal_shape = 'user'; par.centrifugal_f = None; par.centrifugal_profile = 'p*(1 + r**2 - 0.5*r**4)'; par.centrifugal_param = 0.02"
BT="par.thermal = 1; par.stratification = 'linear'"
VI="par.ViscosD = 1e-3; par.ThermaD = 1e-3"
DR="par.diff_rot = 1; par.diff_rot_type = 'conical'; par.diff_rot_ampl = -0.25; par.thermal_wind = 1"
names=(ru_wu_m1 ru_wu_m0 ru_wu_dr ru_bq_m2 ru_bq_fs1 ru_bq_fs2 ru_bq_td0)
cfgs=("$WU; $VI; $CL; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 24"
      "$WU; $VI; $CL; par.ricb = 0.35; par.m = 0; par.symm = -1; par.N = 96; par.lmax = 25"
      "$WU; $VI; $CL; $DR; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 24"
      "$BT; $VI; $US; par.ricb = 0.35; par.m = 2; par.symm = -1; par.N = 48; par.lmax = 21"
      "$BT; $VI; $US; par.N = 48"
      "$BT; $VI; $US; par.m = 2; par.symm = 1; par.N = 48; par.lmax = 21"
      "$BT; par.ViscosD = 1e-3; par.ThermaD = 0; $US; par.symm = 1; par.N = 48")
for i in "${!names[@]}"; do
  d=${names[$i]}
  python3 "$T/mk_run.py" $d "${cfgs[$i]}" > /dev/null || { failed+=("$d"); continue; }
  check $d bash -c "cd $d && python3 bin/ru_insane.py | grep 'centrifugal\|PASS\|FAIL' | sed 's/^/$d: /'"
done

if [ -z "$SKIP_SOLVE" ]; then
  echo "--- budget_check.py: spin_doctor's budgets with the metric weights of KE, the Coriolis power and TE (solve.py, 2 ranks)"
  BUD="$WU; $VI; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 40; par.ncpus = 2; par.nev = 4; par.itau = 0.3"
  names=(bud_wu_sphere bud_wu_cl bud_wu_dr bud_bq_fs)
  cfgs=("$BUD; par.centrifugal = 0"
        "$BUD; $CL"
        "$BUD; $CL; $DR"
        "$BT; $VI; $US; par.m = 2; par.symm = 1; par.N = 64; par.lmax = 41; par.ncpus = 2; par.nev = 4; par.itau = 0.5")
  for i in "${!names[@]}"; do
    d=${names[$i]}
    python3 "$T/mk_run.py" $d "${cfgs[$i]}" solve > /dev/null || { failed+=("$d"); continue; }
    check $d bash -c "cd $d && python3 '$T/budget_check.py' | sed 's/^/$d: /'"
  done
fi

if [ -n "$OLD_KORE" ]; then
  echo "--- regress.sh: centrifugal = 0 leaves the matrices of the 26 regression runs unchanged"
  check regress bash -c "bash '$T/../diff_rot/regress.sh' '$OLD_KORE' '$W/regress' | tail -1"
fi

if [ ${#failed[@]} -eq 0 ]; then echo "centrifugal checks ok"; else echo "centrifugal checks FAILED: ${failed[*]}"; exit 1; fi
