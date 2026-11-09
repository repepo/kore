#!/usr/bin/env python3
'''
usage: budget_check.py [DIR]      runs the cases below in DIR (default budget_runs) and checks spin_doctor's budgets

The kinetic energy budget with the work of the wall stresses (spin_doctor, flow.dat columns 21-27), for forced boundary
flows (forcing 14) and eigenproblems, with a variable density, buoyancy, a differential rotation and the three kinds of
velocity wall condition. For every solution:
  - residσ and residω below TOL_BUDGET (TOL_CASE for a case whose solve is noisier): the budget closes, the pressure
    work Wp included (it is the only term that balances the power the walls put in, so a wrong wall pressure shows up
    here);
  - residD (Dkin = Wv - Dvisc) below TOL_D: the dissipation from the strain rate on an (r, θ) grid against the power of
    the viscous force from the operator of spin_doctor's flow_worker and the wall traction;
  - Dvisc > 0;
and by kind of wall:
  - no-slip and stress-free walls of eigenproblems: uᵣ = 0 at the walls, so Wp = 0, and Wv = 0 (𝐮 = 0, or no tangential
    stress and uᵣ = 0);
  - bco = 2 (zero radial stress, uᵣ ≠ 0): Wp ≠ 0;
  - forcing 14: the walls put in power, Wp + Wv > 0, equal to Dvisc - Wthm - Wdr.
Last, the run must stop at startup (background.check_forcing), with its message, when forcing 14 is given a stress-free
wall.
'Wu 2005' (r_out 0.99) at Ek 1e-3 needs N 192 in a shell and N 384 in a full sphere for the wall pressure (residσ 5e-2 at
N 96 in a shell, 0.4 at N 192 in a full sphere): see docs/tidal-forcing.md.
'''
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL_BUDGET = 1e-7
# NEW (2026-11-09): residσ of f14_m2_wu_conical goes from 3e-10 to 1.2e-7 between repeated runs of the same code (MUMPS on
# 2 ranks is not reproducible), Dkin the same to 11 digits; a missing or wrong term leaves 1e-2 or more
TOL_CASE = {'f14_m2_wu_conical': 1e-6}
TOL_D = 1e-10
TOL_ZERO = 1e-12     # |Wp|, |Wv| relative to |Dkin| where they vanish

WU = "par.structure = 'Wu 2005'; par.thermal = 1; par.stratification = 'top hat'"
BOUS_THM = "par.thermal = 1; par.stratification = 'top hat'"
F14 = "par.forcing = 14; par.forcing_frequency = 1.0; par.forcing_amplitude_cmb = 1.0; par.forcing_amplitude_icb = 0.5"

CASES = [   # (name, parameter changes, kind of walls: 'fixed' (uᵣ = 0), 'bc2' or 'forced')
    ('eig_noslip_wu',     WU, 'fixed'),
    ('eig_stressfree_wu', WU + "; par.bci = 0; par.bco = 0; par.N = 192", 'fixed'),
    ('eig_bc2_wu',        WU + "; par.bco = 2; par.N = 192", 'bc2'),
    ('f14_m2_thermal',    BOUS_THM + "; " + F14, 'forced'),
    ('f14_m1_diffrot',    BOUS_THM + "; " + F14 + "; par.m = 1; par.symm = -1; par.diff_rot = 1; "
                          "par.diff_rot_type = 'Y20-wall-bounded'; par.diff_rot_ampl = 0.1", 'forced'),
    ('f14_m2_wu_conical', WU + "; " + F14 + "; par.N = 192; g = 0.25; par.diff_rot = 1; par.thermal_wind = 1", 'forced'),
    ('f14_sphere_wu',     WU + "; " + F14 + "; par.N = 384; g = 0.25; par.ricb = 0; par.forcing_amplitude_icb = 0.0", 'forced'),
]


def check(d, kind, tol=TOL_BUDGET):
    f = np.loadtxt(os.path.join(d, 'flow.dat'), ndmin=2)
    t = np.loadtxt(os.path.join(d, 'thermal.dat'), ndmin=2) if os.path.isfile(os.path.join(d, 'thermal.dat')) else None
    fails = []
    for i, row in enumerate(f):
        Dkin, resid2, resid1, Wdr = row[3], row[15], row[16], row[17]
        Wp_icb, Wp_cmb, Wp_im, Wv_icb, Wv_cmb, Dvisc, residD = row[20:27]
        Wthm = t[i, 4] if t is not None else 0.
        Wp, Wv = Wp_icb + Wp_cmb, Wv_icb + Wv_cmb
        print('  %d: residσ %.1e  residω %.1e  residD %.1e  Wp % .3e  Wv % .3e  Dvisc %.3e  Dkin % .3e'
              % (i, resid1, resid2, residD, Wp, Wv, Dvisc, Dkin))
        if not resid1 < tol:        fails.append('%d: residσ %.1e' % (i, resid1))
        if not resid2 < tol:        fails.append('%d: residω %.1e' % (i, resid2))
        if not residD < TOL_D:      fails.append('%d: residD %.1e' % (i, residD))
        if not Dvisc > 0:           fails.append('%d: Dvisc %.3e' % (i, Dvisc))
        if kind == 'fixed' and max(abs(Wp_icb), abs(Wp_cmb), abs(Wp_im), abs(Wv_icb), abs(Wv_cmb)) > TOL_ZERO*abs(Dkin):
            fails.append('%d: wall powers not zero' % i)
        if kind == 'bc2' and not abs(Wp_cmb) > 1e-8:
            fails.append('%d: Wp_cmb = %.1e with bco = 2' % i)
        if kind == 'forced':
            if not Wp + Wv > 0:
                fails.append('%d: power input %.3e' % (i, Wp + Wv))
            if not abs(Wp + Wv - (Dvisc - Wthm - Wdr)) < tol*max(abs(Wp + Wv), Dvisc):
                fails.append('%d: Wp + Wv != Dvisc - Wthm - Wdr' % i)
    return fails


def main(top):
    sys.path.insert(0, HERE)
    import mk_run
    os.makedirs(top, exist_ok=True)
    bad = {}
    for name, changes, kind in CASES:
        d = os.path.join(top, name)
        mk_run.build(d, changes)
        print(name + ':')
        fails = check(d, kind, TOL_CASE.get(name, TOL_BUDGET))
        if fails:
            bad[name] = fails
            print('  FAIL: ' + '; '.join(fails))
    # wrong walls for forcing 14: the run must stop with a message, not hang (NEW 2026-11-09: at startup, in submatrices.py
    # (background.check_forcing), before any operator is built; assemble.py prints the same message if a run gets there)
    try:
        mk_run.build(os.path.join(top, 'bad_walls'), F14 + "; par.bco = 0")
        bad['bad_walls'] = ['the run did not stop']
    except SystemExit as e:
        if 'forcing 14 needs no-slip walls' not in str(e) or 'bin/submatrices.py failed' not in str(e):
            bad['bad_walls'] = [str(e)[-300:]]
    print('bad_walls:', 'FAIL' if 'bad_walls' in bad else 'the run stopped at startup with its message')
    print('budget_check:', 'all passed' if not bad else 'FAILED in ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'budget_runs'))
