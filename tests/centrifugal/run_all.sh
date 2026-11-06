#!/bin/bash
# Checks of the centrifugal deformation (centrifugal = 1; docs/centrifugal-equations.md, Checks). NEW (2026-11-05).
# usage: tests/centrifugal/run_all.sh [WORKDIR]   (default tests/centrifugal/runs; about 17 minutes on 16 cores)
# Needs numpy, scipy, sympy and mpi4py; the budgets also petsc4py and slepc4py (SKIP_SOLVE=1 skips them). With OLD_KORE set to
# a copy of the commit before the change (git archive <commit> | tar -x -C $OLD_KORE), also tests/diff_rot/regress.sh.
# Each check prints its verdict; the script ends with 'centrifugal checks ok' (exit 0) or lists the ones that failed (exit 1).
set -o pipefail
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"
failed=()
check() { local name=$1 t0=$SECONDS; shift; if ! "$@"; then failed+=("$name"); fi; echo "    ($name: $((SECONDS - t0)) s)"; }

echo "--- derive.py: the closed forms of bin/centrifugal.py from X = α + ξ, w = J F⁻¹ u (sympy)"
check derive bash -c "python3 '$T/derive.py' | tail -1"
echo "--- algebra_check.py: the angular couplings (angular.py's algebra) against a direct projection of the closed forms"
check algebra bash -c "python3 '$T/algebra_check.py' | tail -1"
echo "--- derive_mag.py: the magnetic terms of the deformed body from b = F b̃/J (sympy), and kore's couplings against them"
check derive_mag bash -c "python3 '$T/derive_mag.py' | tail -1"
# NEW (2026-11-06, phase 2)
echo "--- derive_diff.py: the viscous force, the diffusion, the stress-free and fixed-flux walls and the differential rotation"
echo "    of the deformed body (sympy), and kore's couplings and wall rows against them"
check derive_diff bash -c "python3 '$T/derive_diff.py' | tail -1"
echo "--- bc_check.py: the insulating-wall rows on the deformed wall against the exact matching of a potential field"
check bc bash -c "python3 '$T/bc_check.py' | tail -1"
echo "--- clairaut_check.py: the 'clairaut' flattening (Maclaurin 5/4, polytrope 7.5/π², Wu 2005 against Clairaut from ρ, the Sun)"
check clairaut python3 "$T/clairaut_check.py"
echo "--- eig_check.py: f = 0, the spin-over 1 + f, the relabelling test, homogeneous 'clairaut', the monopole, the ideal fluid,"
echo "    f = 0 with a field, the relabelling test with the axial and the Malkus field"
check eig python3 "$T/eig_check.py"
# NEW (2026-11-06, phase 2)
echo "--- spheroid_modes.py: the inertial modes of a spheroid, exact (polynomial method), against kore's dω/df (73 modes)"
check spheroid python3 "$T/spheroid_modes.py"
echo "--- shape_check.py: damped modes when the walls are deformed (no-slip, stress-free, heat), exactly at Ek 1e-3"
check shape python3 "$T/shape_check.py" $(( $(nproc) < 11 ? $(nproc) : 11 ))
# NEW (2026-11-06, the exact spheroid)
echo "--- spheroid_check.py: the exact spheroid ('spheroid'): its terms from the map, kore's couplings and wall rows against"
echo "    their projections, the inviscid, viscous and heat modes against exact solutions, the rigid rotation; and (step 2)"
echo "    with a field the insulating walls on exact potential fields, the free decay against an exact solution, the inner"
echo "    cores, and the first-order code parting as f²"
check spheroid_exact python3 "$T/spheroid_check.py"

echo "--- ru_insane.py: the metric blocks against (r, θ) quadratures, Hermitian and positive, in shells and full spheres"
WU="par.structure = 'Wu 2005'; par.r_out = 0.99; par.thermal = 1; par.stratification = 'top hat'; par.aux1 = 0.5; par.aux2 = 0.03; par.aux3 = 0.85; par.aux4 = 0.03; par.aux5 = 2.5; par.Beyonce = 10."
CL="par.centrifugal_shape = 'clairaut'; par.centrifugal_f = None; par.centrifugal_q = 0.05"
US="par.centrifugal_shape = 'user'; par.centrifugal_f = None; par.centrifugal_profile = 'p*(1 + r**2 - 0.5*r**4)'; par.centrifugal_param = 0.02"
BT="par.thermal = 1; par.stratification = 'linear'"
VI="par.ViscosD = 1e-3; par.ThermaD = 1e-3"
DR="par.diff_rot = 1; par.diff_rot_type = 'conical'; par.diff_rot_ampl = -0.25; par.thermal_wind = 1"
SP="par.centrifugal_shape = 'spheroid'; par.centrifugal_f = 0.3"     # NEW (2026-11-06): the exact spheroid
names=(ru_wu_m1 ru_wu_m0 ru_wu_dr ru_bq_m2 ru_bq_fs1 ru_bq_fs2 ru_bq_td0 ru_sph_m1 ru_sph_fs2)
cfgs=("$WU; $VI; $CL; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 24"
      "$WU; $VI; $CL; par.ricb = 0.35; par.m = 0; par.symm = -1; par.N = 96; par.lmax = 25"
      "$WU; $VI; $CL; $DR; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 24"
      "$BT; $VI; $US; par.ricb = 0.35; par.m = 2; par.symm = -1; par.N = 48; par.lmax = 21"
      "$BT; $VI; $US; par.N = 48; par.rpower_v = 4"
      "$BT; $VI; $US; par.m = 2; par.symm = 1; par.N = 48; par.lmax = 21; par.rpower_v = 4"
      "$BT; par.ViscosD = 1e-3; par.ThermaD = 0; $US; par.symm = 1; par.N = 48; par.rpower_v = 4"
      "$BT; $VI; $SP; par.ricb = 0.35; par.symm = 1; par.N = 48; par.lmax = 40"
      "$BT; $VI; $SP; par.m = 2; par.symm = 1; par.N = 40; par.lmax = 41; par.rpower_v = 4")
for i in "${!names[@]}"; do
  d=${names[$i]}
  python3 "$T/mk_run.py" $d "${cfgs[$i]}" > /dev/null || { failed+=("$d"); continue; }
  check $d bash -c "cd $d && python3 bin/ru_insane.py | grep 'centrifugal\|PASS\|FAIL' | sed 's/^/$d: /'"
done

if [ -z "$SKIP_SOLVE" ]; then
  echo "--- budget_check.py: spin_doctor's budgets with the metric weights of KE, the Coriolis power and TE (solve.py, 2 ranks)"
  BUD="$WU; $VI; par.ricb = 0.35; par.symm = 1; par.N = 96; par.lmax = 40; par.ncpus = 2; par.nev = 4; par.itau = 0.3"
  SPB="$BT; par.ViscosD = 1e-2; par.ThermaD = 1e-2; par.Beyonce = 2.; $SP; par.ncpus = 2; par.nev = 4; par.itau = 0.5"   # NEW (2026-11-06)
  names=(bud_wu_sphere bud_wu_cl bud_wu_dr bud_bq_fs bud_sph_ns bud_sph_sf bud_sph_fs)
  cfgs=("$BUD; par.centrifugal = 0"
        "$BUD; $CL"
        "$BUD; $CL; $DR"
        "$BT; $VI; $US; par.m = 2; par.symm = 1; par.N = 64; par.lmax = 41; par.ncpus = 2; par.nev = 4; par.itau = 0.5; par.rpower_v = 4"
        "$SPB; par.ricb = 0.35; par.symm = 1; par.N = 48; par.lmax = 40"
        "$SPB; par.ricb = 0.35; par.bco = 0; par.bci = 0; par.bco_thermal = 1; par.m = 2; par.symm = -1; par.N = 48; par.lmax = 41"
        "$SPB; par.m = 2; par.symm = 1; par.N = 40; par.lmax = 41; par.rpower_v = 4")
  for i in "${!names[@]}"; do
    d=${names[$i]}
    python3 "$T/mk_run.py" $d "${cfgs[$i]}" solve > /dev/null || { failed+=("$d"); continue; }
    check $d bash -c "cd $d && python3 '$T/budget_check.py' | sed 's/^/$d: /'"
  done

  # NEW (2026-11-05, phase 1b): with a field, at f and f/2 (@F@): residσ, residω, resid𝐛 to round-off, resid𝐄 of second order
  echo "--- budget_check.py with a field: residσ, residω and resid𝐛 to round-off, resid𝐄 of second order in f (solve.py, 2 ranks)"
  MAG="par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 'mag_energy'; par.Ek = 1e-3; par.Em = 1e-3; par.Le = np.sqrt(1e-3); par.set_scales(); par.Gaspard = 1.; par.Beyonce = 1.; par.ricb = 0.35; par.N = 96; par.lmax = 24; par.ncpus = 2; par.nev = 4; par.itau = 1.0; par.prescale = None"
  UP="par.centrifugal_shape = 'user'; par.centrifugal_f = None; par.centrifugal_profile = 'p*(1 + r**2)/2'; par.centrifugal_param = @F@"
  GEN="par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.05*r**2', 2: '-0.1*r**2', 4: '-0.03*r**4'}; par.lmax = 32; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; par.B0_tor = 'general'; par.B0_tor_profiles = {2: 'r**2*(1 - r**2)'}; par.cnorm_tor = 'mag_energy'"
  names=(bm_uniform bm_cic bm_fs bm_tor bm_gen)
  cfgs=("$MAG; par.centrifugal_f = @F@"
        "$MAG; $UP; par.innercore = 'conducting, Chebys'"
        "$MAG; $UP; par.ricb = 0; par.B0 = 'Luo_S2'; par.symm = 1; par.itau = 0.6; par.rpower_v = 4"
        "$MAG; $UP; par.B0 = None; par.B0_tor = 'Malkus'; par.cnorm_tor = 'mag_energy'"
        "$MAG; $UP; $GEN")
  for i in "${!names[@]}"; do
    d=${names[$i]}; built=1
    for F in 0.005 0.01; do
      python3 "$T/mk_run.py" ${d}_$F "${cfgs[$i]//@F@/$F}" solve > /dev/null || { failed+=("${d}_$F"); built=0; }
    done
    [ $built = 1 ] && check $d bash -c "cd ${d}_0.01 && python3 '$T/budget_check.py' 1e-8 ../${d}_0.005 | sed 's/^/$d: /'"
  done
  # NEW (2026-11-06, step 2): the exact spheroid with a field (f = 0.3, -0.2 for the conducting inner core and the m = 2 case):
  # every budget to round-off, the energy budget resid𝐄 too (an insulating and a conducting inner core, a full sphere with
  # Luo_S2, the Malkus field, general poloidal and toroidal fields)
  echo "--- budget_check.py in the exact spheroid with a field: residσ, residω, resid𝐛 and resid𝐄 to round-off (solve.py, 2 ranks)"
  SPM="$MAG; $SP; par.lmax = 40"
  GEN0="par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; par.B0_tor = 'general'; par.B0_tor_profiles = {2: 'r**2*(1 - r**2)'}; par.cnorm_tor = 'mag_energy'"
  names=(bms_ins bms_cic bms_fs bms_tor bms_gen)
  cfgs=("$SPM"
        "$SPM; par.innercore = 'conducting, Chebys'; par.sigma_i2o = 0.3; par.centrifugal_f = -0.2"
        "$SPM; par.ricb = 0; par.B0 = 'Luo_S2'; par.symm = 1; par.itau = 0.6; par.rpower_v = 4; par.lmax = 48"
        "$SPM; par.B0 = None; par.B0_tor = 'Malkus'; par.cnorm_tor = 'mag_energy'"
        "$SPM; $GEN0; par.centrifugal_f = -0.2; par.m = 2; par.symm = -1; par.lmax = 41")
  for i in "${!names[@]}"; do
    d=${names[$i]}
    python3 "$T/mk_run.py" $d "${cfgs[$i]}" solve > /dev/null || { failed+=("$d"); continue; }
    check $d bash -c "cd $d && python3 '$T/budget_check.py' 1e-9 exact | sed 's/^/$d: /'"
  done
  # the anelastic 'Wu 2005' with a field needs N 128 for its budgets (the sphere's are 2e-6 at N 96): closure only
  d=bm_wu
  python3 "$T/mk_run.py" $d "$MAG; $WU; par.thermal = 0; par.Beyonce = 1.; $CL; par.N = 128" solve > /dev/null && \
    check $d bash -c "cd $d && python3 '$T/budget_check.py' 1e-7 | sed 's/^/$d: /'" || failed+=("$d")
fi

if [ -n "$OLD_KORE" ]; then
  echo "--- regress.sh: centrifugal = 0 leaves the matrices of the 26 regression runs unchanged"
  check regress bash -c "bash '$T/../diff_rot/regress.sh' '$OLD_KORE' '$W/regress' | tail -1"
fi

if [ ${#failed[@]} -eq 0 ]; then echo "centrifugal checks ok ($SECONDS s)"; else echo "centrifugal checks FAILED: ${failed[*]} ($SECONDS s)"; exit 1; fi
