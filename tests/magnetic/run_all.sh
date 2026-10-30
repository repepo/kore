#!/bin/bash
# Checks of the magnetic field with a variable density (docs/magnetic.md). usage: tests/magnetic/run_all.sh [WORKDIR]
# (default tests/magnetic/runs; needs numpy, scipy, sympy, mpi4py, petsc4py, slepc4py; ~40 minutes on 2 to 4 ranks)
set -e -o pipefail
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"
echo "--- derive_check.py: the rows from the definitions (sympy) against conductive_IC's operators and the closed forms (SKIP_DERIVE=1 skips it, ~40 min)"
if [ -z "$SKIP_DERIVE" ]; then python3 "$T/derive_check.py" | tail -2; fi
echo "--- ru_insane.py: operator symmetries and the Lorentz/induction energy identity (B0 'axial', no-slip, insulating walls)"
python3 "$T/mk_run.py" ru_uni    "par.structure = 'Boussinesq'"
python3 "$T/mk_run.py" ru_wu_m1  ""
python3 "$T/mk_run.py" ru_wu_m0  "par.m = 0; par.symm = 1; par.lmax = 23"
python3 "$T/mk_run.py" ru_wu_m2  "par.m = 2; par.symm = 1; par.lmax = 25"
python3 "$T/mk_run.py" ru_wu_cic "par.B0 = 'FDM'; par.innercore = 'conducting, Chebys'"
for d in ru_uni ru_wu_m1 ru_wu_m0 ru_wu_m2 ru_wu_cic; do (cd $d && python3 bin/ru_insane.py | grep "magnetic\|PASS\|FAIL" | sed "s/^/$d: /"); done
echo "--- row_check.py: every magnetic row block against derive.py ('Wu 2005' needs N = 192 for round-off: the pole of (ln rho0)' sits at r = 1/r_out)"
python3 "$T/mk_run.py" rc_ax_m1  "par.N = 192"
python3 "$T/mk_run.py" rc_fdm_m0 "par.N = 192; par.B0 = 'FDM'; par.m = 0; par.symm = 1; par.lmax = 23; par.tol_B0 = 1e-15"   # the FDM series uncut: 1e-9 with tol_B0 = 1e-8
for d in rc_ax_m1 rc_fdm_m0; do (cd $d && python3 "$T/row_check.py" | tail -1 | sed "s/^/$d: /"); done
echo "--- budget_check.py: spin_doctor's budgets with variable rho0 (FDM, conducting inner core, Ek 1e-3, N 192)"
python3 "$T/mk_run.py" bud_m1 "par.N = 192; par.lmax = 48; par.B0 = 'FDM'; par.cnorm = 'Schmitt2012'; par.innercore = 'conducting, Chebys'" solve
python3 "$T/mk_run.py" bud_m0 "par.N = 192; par.lmax = 47; par.B0 = 'FDM'; par.cnorm = 'Schmitt2012'; par.innercore = 'conducting, Chebys'; par.m = 0; par.symm = 1; par.itau = 0.3" solve
for d in bud_m1 bud_m0; do (cd $d && python3 "$T/budget_check.py" | tail -1 | sed "s/^/$d: /"); done
echo "--- toroidal B0 (B0_tor 'Malkus' or 'wall', degree 1 or 2, alone or with 'axial'): row checks, budgets ('Wu 2005', N 192) and the Malkus (1967) theory"
python3 "$T/mk_run.py" rc_mal_m1  "par.N = 192; par.B0 = None; par.B0_tor = 'Malkus'"
python3 "$T/mk_run.py" rc_wall_m0 "par.N = 192; par.B0 = None; par.B0_tor = 'wall'; par.m = 0; par.symm = 1; par.lmax = 23"
python3 "$T/mk_run.py" rc2_mal_m1  "par.N = 192; par.B0 = None; par.B0_tor = 'Malkus'; par.B0_tor_l = 2"
python3 "$T/mk_run.py" rc2_comb_m2 "par.N = 192; par.B0_tor = 'Malkus'; par.B0_tor_l = 2; par.m = 2; par.symm = 1; par.lmax = 25"
for d in rc_mal_m1 rc_wall_m0 rc2_mal_m1 rc2_comb_m2; do (cd $d && python3 "$T/row_check.py" | tail -1 | sed "s/^/$d: /"); done
python3 "$T/mk_run.py" tb_mal_m1  "par.N = 192; par.lmax = 48; par.B0 = None; par.B0_tor = 'Malkus'; par.cnorm_tor = 'mag_energy'; par.innercore = 'conducting, Chebys'" solve
python3 "$T/mk_run.py" tb_wall_m0 "par.N = 192; par.lmax = 47; par.B0 = None; par.B0_tor = 'wall'; par.cnorm_tor = 'mag_energy'; par.m = 0; par.symm = 1; par.itau = 0.3" solve
python3 "$T/mk_run.py" tb2_wall_m1 "par.N = 192; par.lmax = 48; par.B0 = None; par.B0_tor = 'wall'; par.B0_tor_l = 2; par.cnorm_tor = 'mag_energy'; par.innercore = 'conducting, Chebys'" solve
for d in tb_mal_m1 tb_wall_m0 tb2_wall_m1; do (cd $d && python3 "$T/budget_check.py" | tail -1 | sed "s/^/$d: /"); done
python3 "$T/malkus_check.py"
echo "--- NEW (2026-10-28): full sphere (ricb = 0) and the degree-2 poloidal B0 (Luo_S2, FDM with B0_l = 2): row checks and budgets"
FS="par.ricb = 0; par.innercore = 'insulator'"
python3 "$T/mk_run.py" rcf_ax_m1   "par.N = 288; $FS"   # (the pole of (ln rho0)' at r = 1/r_out is nearer to the full-sphere domain [-1, 1])
python3 "$T/mk_run.py" rcf_s2_m0   "par.N = 288; $FS; par.B0 = 'Luo_S2'; par.m = 0; par.symm = -1; par.lmax = 23"
python3 "$T/mk_run.py" rcf_mal_m2  "par.N = 288; $FS; par.B0 = None; par.B0_tor = 'Malkus'; par.m = 2; par.symm = 1; par.lmax = 25"
python3 "$T/mk_run.py" rcf_inv_s2  "par.N = 96; $FS; par.structure = 'Boussinesq'; par.Ek = 0; par.ViscosD = 0; par.bci = 0; par.bco = 0; par.B0 = 'Luo_S2'; par.m = 1; par.symm = 1; par.lmax = 24"
python3 "$T/mk_run.py" rc_s2_m1    "par.N = 192; par.B0 = 'Luo_S2'; par.tol_B0 = 1e-15"   # (2e-8 with the default tol_B0: the B0_ic r^-3 term)
python3 "$T/mk_run.py" rc_f2_m2    "par.N = 192; par.B0 = 'FDM'; par.B0_l = 2; par.m = 2; par.symm = -1; par.lmax = 25; par.tol_B0 = 1e-15"
for d in rcf_ax_m1 rcf_s2_m0 rcf_mal_m2 rcf_inv_s2 rc_s2_m1 rc_f2_m2; do (cd $d && python3 "$T/row_check.py" | tail -1 | sed "s/^/$d: /"); done
python3 "$T/mk_run.py" bf_s2_m1 "par.N = 192; par.lmax = 48; $FS; par.B0 = 'Luo_S2'" solve
python3 "$T/mk_run.py" bf_ax_m0 "par.N = 96; par.lmax = 47; $FS; par.structure = 'Boussinesq'; par.m = 0; par.symm = 1; par.itau = 0.3" solve   # ('Wu 2005': 6e-7 at N 192, the Hartmann layer of a uniform field)
python3 "$T/mk_run.py" b_s2_cic "par.N = 192; par.lmax = 48; par.B0 = 'Luo_S2'; par.innercore = 'conducting, Chebys'" solve
for d in bf_s2_m1 bf_ax_m0 b_s2_cic; do (cd $d && python3 "$T/budget_check.py" | tail -1 | sed "s/^/$d: /"); done
echo "--- NEW (2026-10-30): the magnetic field with a differential rotation, the confined, general and ferraro fields, the coupling algebra"
if [ -z "$SKIP_DERIVE" ]; then
  python3 "$T/dr_induction_check.py" --quick | tail -1   # ∇×(U0×b) of diff_rot.py against derive.py (sympy)
  python3 "$T/algebra_check.py" --quick | tail -1        # mag_algebra.py against derive.py, degrees 1 and 3
fi
python3 "$T/rigid_check.py" | tail -7                    # ΔΩ = c is a rigid rotation: λ = λ(Gaspard 1 + c) - i m c exactly
DR="par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_profiles = {0: '0.05*r**2', 2: '-0.1*r**2', 4: '-0.03*r**4'}; par.lmax = 32"
python3 "$T/mk_run.py" bdr_tor  "par.N = 96; $DR; par.B0 = None; par.B0_tor = 'Malkus'" solve
python3 "$T/mk_run.py" bdr_pol  "par.N = 192; $DR; par.B0 = 'axial'" solve
python3 "$T/mk_run.py" bdr_fs   "par.N = 192; $DR; $FS; par.B0 = 'Luo_S2'; par.symm = 1; par.itau = 0.6" solve
python3 "$T/mk_run.py" bconf    "par.N = 128; par.structure = 'Boussinesq'; par.B0 = 'confined'; par.B0_l = 2; par.B0_conf_r1 = 0.4; par.B0_conf_r2 = 0.8; par.symm = 1; par.itau = 0.6; par.lmax = 32" solve
python3 "$T/mk_run.py" bgen     "par.N = 192; $DR; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; par.B0_tor = 'general'; par.B0_tor_profiles = {2: 'r**2*(1 - r**2)'}; par.cnorm_tor = 'mag_energy'" solve
python3 "$T/mk_run.py" bfer     "par.N = 192; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_ampl = 0.2; par.diff_rot_profiles = {0: '(2/3)*r**2 - (8/15)*r**4', 2: '-(2/3)*r**2 + (16/21)*r**4', 4: '-(8/35)*r**4'}; par.B0 = 'ferraro'; par.B0_ferraro_G = 'w'; par.B0_ferraro_nmax = 7; par.lmax = 32" solve
python3 "$T/mk_run.py" bfdr     "par.N = 192; par.diff_rot = 1; par.diff_rot_type = 'ferraro'; par.diff_rot_F = 'psi + psi**2'; par.diff_rot_ampl = 0.05; par.diff_rot_nmax = 8; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2)'}; par.lmax = 32" solve   # ΔΩ = F(ψ) from the field
python3 "$T/mk_run.py" bfer_tor "par.N = 192; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_ampl = 0.2; par.diff_rot_profiles = {0: '(2/3)*r**2 - (8/15)*r**4', 2: '-(2/3)*r**2 + (16/21)*r**4', 4: '-(8/35)*r**4'}; par.B0 = 'ferraro'; par.B0_ferraro_G = 'w'; par.B0_ferraro_nmax = 7; par.B0_tor = 'Malkus'; par.B0_tor_l = 2; par.cnorm_tor = 'mag_energy'; par.lmax = 32" solve   # with a toroidal part (degree 2: the same symmetry)
for d in bdr_tor bdr_pol bdr_fs bconf bgen bfer bfdr bfer_tor; do (cd $d && python3 "$T/budget_check.py" | tail -1 | sed "s/^/$d: /"); done
echo "--- continuity.py: 'Wu 2005' with r_out = sqrt(eps) -> 'Boussinesq' as eps -> 0"
python3 "$T/continuity.py" 1e-4 1e-3 1e-2 1e-1
