#!/bin/bash
# The free core nutation with a field (centrifugal = 1; docs/centrifugal-equations.md, Checks). NEW (2026-11-05, phase 1b).
# The spin-over of a fixed mantle at Ek 1e-7, Em 1e-5 and Le 3e-3, with the axial field and a conducting inner core: at the
# flattenings f = 2.5e-3, -f and f/2 (N 160, lmax 120), at ±f without the field, and at ±f with N 200, lmax 150.
# FIX (2026-11-06, phase 2): the shift is measured between deformed bodies, b = (Im λ(f) - Im λ(-f))/(2f). A run at
# f = 0 keeps the rows of the sphere (the toroidal flow in C^(2), centrifugal.deformed), whose error of discretization
# at N 160 differs from that of the deformed runs (C^(3)) by 1.5e-6 in λ: 6e-4 of f in a shift taken from it. In the
# first-order equations, without diffusion, Im λ = 1 + f + f²/3. The Ekman layers add to b a part of order Ek^½, now
# that the viscous force is that of the deformed body: Greenspan's boundary-layer theory puts it at Ek^½ Im dλ1/df =
# 4.6e-4 (dλ1/df = 0.021 + 1.447i for the spin-over of a spheroid, spheroid_modes.py --ekman), which kore approaches
# only slowly (as Ek^0.1, the critical latitudes): 5.8e-4 here, 6.1e-4 with the field. The check asks for b - 1 within a
# factor 2 of Greenspan's value, for the f² coefficient 1/3 (to 1e-2), and that the field and the resolution change b by
# less than 1e-4 and 2e-5.
# usage: tests/centrifugal/fcn.sh [WORKDIR] [NP]   (default tests/centrifugal/runs and 15 ranks; NP must be 1, 3, 5 or 15,
# since lmax - m + 1, 120 and 150, must be a multiple of 2 NP; about 5 minutes and 15 GB on 15 ranks). Needs petsc4py and
# slepc4py. MUMPS gets 1000 % more workspace than it estimates (st_mat_mumps_icntl_14: with 50 it stops with INFOG(1)=-9,
# and with 500 a deformed run at N 200 stopped once with INFOG(1)=-20, its reception buffer too small, where the same run
# had passed before). FIX (2026-11-06, phase 2): 1000, was 500.
# Not part of run_all.sh. The budgets of these runs need more resolution (residσ 5e-4 at N 160, 1e-5 at N 200, also at f = 0).
set -o pipefail
T=$(cd "$(dirname "$0")" && pwd)
W=${1:-$T/runs}; NP=${2:-15}; mkdir -p "$W"; cd "$W"
FCN="par.magnetic = 1; par.B0 = 'axial'; par.cnorm = 'mag_energy'; par.innercore = 'conducting, Chebys'; par.ricb = 0.35; par.Ek = 1e-7; par.Em = 1e-5; par.Le = 3e-3; par.set_scales(); par.Gaspard = 1.; par.Beyonce = 1.; par.nev = 4; par.prescale = None; par.ncpus = $NP; par.petsc_opts['st_mat_mumps_icntl_14'] = 1000"
LO="par.N = 160; par.lmax = 120"; HI="par.N = 200; par.lmax = 150"; HYD="par.magnetic = 0"
names=(fcn_p fcn_m fcn_half fcn_hyd_p fcn_hyd_m fcn_hi_p fcn_hi_m)
cfgs=("$FCN; $LO; par.centrifugal_f = 0.0025; par.itau = 1.0025"
      "$FCN; $LO; par.centrifugal_f = -0.0025; par.itau = 0.9975"
      "$FCN; $LO; par.centrifugal_f = 0.00125; par.itau = 1.00125"
      "$FCN; $LO; $HYD; par.centrifugal_f = 0.0025; par.itau = 1.0025"
      "$FCN; $LO; $HYD; par.centrifugal_f = -0.0025; par.itau = 0.9975"
      "$FCN; $HI; par.centrifugal_f = 0.0025; par.itau = 1.0025"
      "$FCN; $HI; par.centrifugal_f = -0.0025; par.itau = 0.9975")
for i in "${!names[@]}"; do
  python3 "$T/mk_run.py" ${names[$i]} "${cfgs[$i]}" solve || { echo "fcn.sh: FAILED (${names[$i]})"; exit 1; }
done
python3 - "${names[@]}" <<'EOF'
import sys
import numpy as np
f, EK = 0.0025, 1e-7
GREENSPAN = np.sqrt(EK)*1.4465     # Ek^½ Im dλ1/df of the spin-over of a spheroid (spheroid_modes.py --ekman)
def spin_over(d):
    '''λ of the spin-over: the solution whose kinetic energy is the most toroidal (flow.dat columns 1 and 3)'''
    ev, fl = np.loadtxt(d + '/eigenvalues.dat', ndmin=2), np.loadtxt(d + '/flow.dat', ndmin=2)
    return complex(*ev[np.argmax(fl[:, 2]/fl[:, 0])])
lam = {d: spin_over(d) for d in sys.argv[1:]}
for d, l in lam.items():
    print('  %-10s λ = %.9e %+.9ei' % (d, l.real, l.imag))
b = lambda p, m: (lam[p].imag - lam[m].imag)/(2*f)             # the shift of first order, between ±f
s = b('fcn_p', 'fcn_m')
c2 = 4*((lam['fcn_p'].imag + lam['fcn_m'].imag)/2 - lam['fcn_half'].imag + s*f/2)/(3*f*f)   # the f² coefficient, from f, -f, f/2
checks = [('b - 1, the part of the Ekman layers and the field', s - 1, 'Greenspan %.1e, within a factor 2' % GREENSPAN,
           0.5 < (s - 1)/GREENSPAN < 2),
          ('the coefficient of f² - 1/3', c2 - 1/3, 'below 1e-2', abs(c2 - 1/3) < 1e-2),
          ('the field: b with it - without', s - b('fcn_hyd_p', 'fcn_hyd_m'), 'below 1e-4',
           abs(s - b('fcn_hyd_p', 'fcn_hyd_m')) < 1e-4),
          ('the resolution: b at N 200 - at N 160', b('fcn_hi_p', 'fcn_hi_m') - s, 'below 2e-5',
           abs(b('fcn_hi_p', 'fcn_hi_m') - s) < 2e-5)]
bad = []
for name, v, crit, ok in checks:
    print('  %-52s %+.2e (%s)' % (name, v, crit))
    if not ok:
        bad.append(name)
print('fcn.sh: ' + ('ok' if not bad else 'FAIL ' + '; '.join(bad)))
sys.exit(1 if bad else 0)
EOF
