#!/bin/bash
# Rebuilds the test runs of the differential-rotation heat-equation terms and runs every check (docs/heat-equation.md).
# usage: tests/heat_dr/run_all.sh [WORKDIR]   (default: tests/heat_dr/runs; needs numpy, scipy, mpi4py; several minutes)
set -e
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; mkdir -p "$W"; cd "$W"
SUN="par.ricb=0; par.model_type='smoothed sun'; par.N=120; par.lmax=30; par.m=1; par.symm=1; par.aux0=0.99; par.aux1=0.0; par.aux2=0.1; par.aux3=0.68; par.aux4=0.06; par.aux5=2.6; par.Beyonce=1e3; par.ViscosD=1e-3; par.ThermaD=1e-3; par.visc0=1.0; par.therm0=1.0; par.visc_pow=0.0; par.therm_pow=0.0; par.diff_rot_type='conical'; par.diff_rot_amplitude=-0.25; par.bco=1; par.bco_thermal=0"
echo "--- building the runs (conical rotation and user-def profiles unless stated; mk_run.py has the defaults; wall-bounded and shellular exercise f', f'')"
python3 "$T/mk_run.py" s96      'par.N=96'
python3 "$T/mk_run.py" nodr96   'par.N=96; par.diff_rot=0'
python3 "$T/mk_run.py" s160     'par.N=160'
python3 "$T/mk_run.py" tw0      'par.N=96; par.thermal_wind=0'
python3 "$T/mk_run.py" m0       'par.N=96; par.m=0; par.lmax=21'
python3 "$T/mk_run.py" nodr_m0  'par.N=96; par.m=0; par.lmax=21; par.diff_rot=0'
python3 "$T/mk_run.py" m2b      'par.N=96; par.m=2; par.symm=-1; par.lmax=21'
python3 "$T/mk_run.py" nodr_m2b 'par.N=96; par.m=2; par.symm=-1; par.lmax=21; par.diff_rot=0'
python3 "$T/mk_run.py" y20wb      "par.N=96; par.diff_rot_type='Y20-wall-bounded'"
python3 "$T/mk_run.py" nodr_y20wb "par.N=96; par.diff_rot_type='Y20-wall-bounded'; par.diff_rot=0"
python3 "$T/mk_run.py" shell      "par.N=96; par.diff_rot_type='shellular'"
python3 "$T/mk_run.py" nodr_shell "par.N=96; par.diff_rot_type='shellular'; par.diff_rot=0"
python3 "$T/mk_run.py" sun      "$SUN; par.diff_rot=1"
python3 "$T/mk_run.py" nodr_sun "$SUN; par.diff_rot=0"
python3 "$T/mk_run.py" sun_a      "$SUN; par.symm=-1; par.diff_rot=1"
python3 "$T/mk_run.py" nodr_sun_a "$SUN; par.symm=-1; par.diff_rot=0"
python3 "$T/mk_run.py" rbase    'par.N=64; par.lmax=25; par.m=2; par.diff_rot=0; par.ViscosD=1e-3/(1+0.1); par.ThermaD=1e-3/(1+0.1); par.Beyonce=10/(1+0.1)**2'
python3 "$T/mk_run.py" rfix     "par.N=64; par.lmax=25; par.m=2; par.diff_rot=1; par.diff_rot_type='rigid'; par.diff_rot_amplitude=0.1"
echo "--- heat_check: operator blocks vs the (r, theta) quadrature of p0 s* [i m dOmega s + u.grad S_tw]"
for d in s96 s160 tw0 m0 m2b y20wb shell sun sun_a; do (cd $d && python3 "$T/heat_check.py" 3 | tail -1 | sed "s/^/$d: /"); done
echo "--- asm_check: heat rows of the assembled A(diff_rot=1) - A(diff_rot=0) vs the same quadrature"
(cd s96 && python3 "$T/asm_check.py" ../nodr96/A.npz 3 | tail -1 | sed "s/^/s96: /")
for d in m0 m2b y20wb shell sun sun_a; do (cd $d && python3 "$T/asm_check.py" ../nodr_$d/A.npz 3 | tail -1 | sed "s/^/$d: /"); done
echo "--- rigid-offset identity with thermal = 1: lambda(dO) = (1+dO) lambda(0; E/(1+dO), kappa/(1+dO), Beyonce/(1+dO)^2) - i m dO"
(cd rbase && python3 "$T/csolve.py" 0 1.0 16 > csolve.out)
(cd rfix  && python3 "$T/csolve.py" 0 0.9 16 > csolve.out)
python3 "$T/rigid_thermal.py" rbase rfix 0.1 2 | tail -1
echo "--- smoothed sun, ricb = 0: 12 modes near -0.05 + 0.8i, momentum and entropy balances (terms.py)"
(cd sun && python3 "$T/csolve.py" -0.05 0.8 12 > csolve.out && python3 "$T/terms.py" > terms.out && grep -A 13 "^entropy" terms.out)
