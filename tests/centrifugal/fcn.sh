#!/bin/bash
# The free core nutation with a field (centrifugal = 1; docs/centrifugal-equations.md, Checks). NEW (2026-11-05, phase 1b).
# The spin-over of a fixed mantle at Ek 1e-7, Em 1e-5 and Le 3e-3, with the axial field and a conducting inner core: at the
# flattenings 0, f/2 and f = 2.5e-3 (N 160, lmax 120), at 0 and f without the field, and at 0 and f with N 200, lmax 150.
# In the first-order equations Im λ moves by f (1 + f/3). The check asks for that up to the part the viscosity and the field
# add (1e-4 of f here; below 3e-4), for the f² coefficient 1/3 (to 1e-2), and that the field and the resolution change the
# shift by less than 1e-4 and 2e-5 of f.
# usage: tests/centrifugal/fcn.sh [WORKDIR] [NP]   (default tests/centrifugal/runs and 15 ranks; NP must be 1, 3, 5 or 15,
# since lmax - m + 1, 120 and 150, must be a multiple of 2 NP; about two minutes and 5 GB on 15 ranks). Needs petsc4py and
# slepc4py. MUMPS gets 500 % more workspace than it estimates (st_mat_mumps_icntl_14; with 50 it stops with INFOG(1)=-9).
# Not part of run_all.sh. The budgets of these runs need more resolution (residσ 5e-4 at N 160, 1e-5 at N 200, also at f = 0).
set -o pipefail
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; NP=${2:-15}; mkdir -p "$W"; cd "$W"
FCN="par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 'mag_energy'; par.innercore = 'conducting, Chebys'; par.ricb = 0.35; par.Ek = 1e-7; par.Em = 1e-5; par.Le = 3e-3; par.set_scales(); par.Gaspard = 1.; par.Beyonce = 1.; par.nev = 4; par.prescale = None; par.ncpus = $NP; par.petsc_opts['st_mat_mumps_icntl_14'] = 500"
LO="par.N = 160; par.lmax = 120"; HI="par.N = 200; par.lmax = 150"; HYD="par.magnetic = 0"
names=(fcn_0 fcn_half fcn_f fcn_hyd_0 fcn_hyd_f fcn_hi_0 fcn_hi_f)
cfgs=("$FCN; $LO; par.centrifugal_f = 0.0; par.itau = 1.0"
      "$FCN; $LO; par.centrifugal_f = 0.00125; par.itau = 1.00125"
      "$FCN; $LO; par.centrifugal_f = 0.0025; par.itau = 1.0025"
      "$FCN; $LO; $HYD; par.centrifugal_f = 0.0; par.itau = 1.0"
      "$FCN; $LO; $HYD; par.centrifugal_f = 0.0025; par.itau = 1.0025"
      "$FCN; $HI; par.centrifugal_f = 0.0; par.itau = 1.0"
      "$FCN; $HI; par.centrifugal_f = 0.0025; par.itau = 1.0025")
for i in "${!names[@]}"; do
  python3 "$T/mk_run.py" ${names[$i]} "${cfgs[$i]}" solve || { echo "fcn.sh: FAILED (${names[$i]})"; exit 1; }
done
python3 - "${names[@]}" <<'EOF'
import sys
import numpy as np
f = 0.0025
def spin_over(d):
    '''λ of the spin-over: the solution whose kinetic energy is the most toroidal (flow.dat columns 1 and 3)'''
    ev, fl = np.loadtxt(d + '/eigenvalues.dat', ndmin=2), np.loadtxt(d + '/flow.dat', ndmin=2)
    return complex(*ev[np.argmax(fl[:, 2]/fl[:, 0])])
lam = {d: spin_over(d) for d in sys.argv[1:]}
for d, l in lam.items():
    print('  %-10s λ = %.9e %+.9ei' % (d, l.real, l.imag))
shift = lambda d0, d1, ff: (lam[d1].imag - lam[d0].imag)/ff      # (Im λ(f) - Im λ(0))/f
s, s_half = shift('fcn_0', 'fcn_f', f), shift('fcn_0', 'fcn_half', f/2)
checks = [('shift/f - 1 - f/3 (the part of viscosity and field)', s - 1 - f/3, 3e-4),
          ('the coefficient of f² - 1/3', (s - s_half)/(f/2) - 1/3, 1e-2),
          ('the field: shift/f with it - without', s - shift('fcn_hyd_0', 'fcn_hyd_f', f), 1e-4),
          ('the resolution: shift/f at N 200 - at N 160', shift('fcn_hi_0', 'fcn_hi_f', f) - s, 2e-5)]
bad = []
for name, v, tol in checks:
    print('  %-52s %+.2e (below %g)' % (name, v, tol))
    if not abs(v) < tol:
        bad.append(name)
print('fcn.sh: ' + ('ok' if not bad else 'FAIL ' + '; '.join(bad)))
sys.exit(1 if bad else 0)
EOF
