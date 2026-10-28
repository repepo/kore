#!/bin/bash
# Checks of the magnetic field with a variable density (docs/magnetic.md). usage: tests/magnetic/run_all.sh [WORKDIR]
# (default tests/magnetic/runs; needs numpy, scipy, sympy, mpi4py, petsc4py, slepc4py; ~40 minutes on 2 to 4 ranks)
set -e -o pipefail
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"
echo "--- derive_check.py: the rows from the definitions (sympy) against conductive_IC's operators and the closed forms (SKIP_DERIVE=1 skips it, ~40 min)"
if [ -z "$SKIP_DERIVE" ]; then python3 "$T/derive_check.py" | tail -2; fi
echo "--- ru_insane.py: operator symmetries and the Lorentz/induction energy identity (B0 'axial', no-slip, insulating walls)"
python3 "$T/mk_run.py" ru_uni    "par.structure = 'uniform'"
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
echo "--- continuity.py: 'Wu 2005' with r_out = sqrt(eps) -> 'uniform' as eps -> 0"
python3 "$T/continuity.py" 1e-4 1e-3 1e-2 1e-1
