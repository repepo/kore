#!/usr/bin/env python3
'''
usage: stops_check.py [WORKDIR]   (default: tests/forcing/runs)

NEW (2026-11-09, phase 11): the settings of the forced runs that background.check_forcing stops at startup, each with its
message, and the nearby settings that run. Each case runs submatrices.py only (where the check stops a run; assemble.py,
solve.py and spin_doctor.py stop on the same check), on 1 rank, N 24.
  - an inviscid fluid (ViscosD = 0) in a shell, whatever the forcing (its wall rows would leave A - λB singular);
  - the settings of forcings 6, 7, 9, 10, 11, 13 and 14 (assemble.py tested them only as it wrote the right-hand side,
    after every operator was built, and 6, 7, 9 and 10 then wrote none, so that solve.py solved whatever B_forced.npz an
    earlier run had left);
  - forcing 7 with m = 0 and an inviscid fluid (its flow is 0);
  - a forced run of an entropy without diffusion (thermal = 1, ThermaD = 0), or of a composition without it, where the
    fluid does not carry it round (background.check_transport): w + m Ω = 0 everywhere (frequency 0 with m = 0 or with
    no rotation, diff_rot_ampl = 0, bgflow_Ro = 0; w = -m Ω of a 'rigid' ΔΩ, also as a frequency of the scan of forcing
    11), w + m Ω of both signs in the fluid ('Y20' at frequency 0), or a background flow with a meridional circulation;
  - any forcing at frequency 0 with a wall of zero radial stress (bco or bci = 2), as forcing 12 (forcings 11 and 13);
  - forcing 9 with the drag kappa_cmb where it moves the outer wall (forcing_amplitude_cmb != 0).
The cases that run: bci in a full sphere and bco with an inviscid fluid (forcings 7 with m = 2, 9, 14; 1 and 8), forcing 13 at
frequency 0 between stress-free walls with the drag, forcing 9 with the drag at an outer wall that stays in place, and
with bco = 2 where it does not move the outer wall, a forced run without diffusion where the fluid carries the entropy
round at a rate of one sign (a 'rigid' ΔΩ at frequency 0, 'conical', a background flow along the azimuth), forcing 11
in a full sphere, with a cut, with rsf_taper_in = 0 in a shell (where it is not used) and with rsf_theta 'mlt'.
tests/forcing/rows_check.py solves several of them.
And three settings outside the forcings that the same audit relaxed (NEW 2026-11-09): a thin layer and the resolved mantle
around an inviscid fluid at bco = 2, and a uniform differential rotation with B0 under ferraro_policy 'refuse' (whose
Ferraro check was round-off over round-off); a viscous fluid at bco = 2 under a thin layer still stops.

Exit code 1 if a case does not stop with its message, or stops when it should run.
'''
import os, sys, shutil, subprocess

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, no-slip walls)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
LMAX_LINE = 'par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)'
assert LMAX_LINE in mk_run.T

def stops(name, extra):
    '''(return code, output) of submatrices.py for the run W/stop/name: N 24, lmax 13 (m 0, 2) or 12 (m 1, 3), 1 rank'''
    d = os.path.join(W, 'stop', name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    t = mk_run.T.replace(LMAX_LINE, 'par.lmax = 13 - par.m % 2')
    open(d + '/bin/parameters.py', 'w').write(t.format(ncpus=1, extra=('par.N = 24; ' + extra).replace('; ', '\n')))
    p = subprocess.run([sys.executable, 'bin/submatrices.py', '1'], cwd=d, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr

INV = 'par.ViscosD = 0.0'
TH = "par.thermal = 1; par.stratification = 'linear'"
STEADY = 'par.forcing_frequency = 0.0'
F9, F13 = 'par.forcing = 9; par.m = 2', 'par.forcing = 13; par.m = 0; ' + TH + '; par.forcing_amplitude_icb = 1.0'
BG = "par.bgflow = 1; par.bgflow_Ro = 0.1; par.bgflow_profiles = {'T1': 'r**2', 'P2': '(r - 0.35)**2*(1 - r)**2'}"
DR = "par.diff_rot = 1; par.diff_rot_type = 'rigid'; par.diff_rot_ampl = 0.1"
MAG = "par.magnetic = 1; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.B0 = 'axial'"
F11 = 'par.forcing = 11; par.m = 1; par.symm = -1'
CASES = [   # (name, settings, a part of the message, or None: it runs)
    ('inviscid_shell', F9 + '; ' + INV, 'an inviscid fluid (ViscosD = 0) needs ricb = 0'),
    ('inviscid_shell_f8', 'par.forcing = 8; par.m = 0; ' + INV, 'an inviscid fluid (ViscosD = 0) needs ricb = 0'),
    ('f6_m0', 'par.forcing = 6; par.m = 0', 'forcing 6 (the radial flow of Buffett 2010 at the inner wall) needs m = 1'),
    ('f6_sphere', 'par.forcing = 6; par.m = 1; par.symm = -1; par.ricb = 0', 'an inner core (ricb > 0)'),
    ('f7_m0_slip', 'par.forcing = 7; par.m = 0; par.bco = 0', 'forcing 7 (the libration of the walls) needs'),
    ('f7_m2_slip_ic', 'par.forcing = 7; par.m = 2; par.bci = 0', 'forcing 7 (the libration of the walls) needs'),
    ('f7_m1', 'par.forcing = 7; par.m = 1; par.symm = -1', 'forcing 7 (the libration of the walls) needs'),
    ('f7_m0_inviscid', 'par.forcing = 7; par.m = 0; par.ricb = 0; ' + INV, 'its flow is 0'),
    ('f9_slip_ic', F9 + '; par.bci = 0', 'forcing 9 (a radial flow of degree 2 at the walls) needs'),
    ('f9_m1', 'par.forcing = 9; par.m = 1; par.symm = -1', 'forcing 9 (a radial flow of degree 2 at the walls) needs'),
    ('f9_bc2', F9 + '; par.bco = 2', 'forcing 9 moves the outer wall radially: it needs bco = 0 or 1'),
    ('f10_m0', 'par.forcing = 10; par.m = 0', 'needs symm = 1 and m >= 1'),
    ('f10_bc2', 'par.forcing = 10; par.m = 2; par.bco = 2', 'forcing 10 moves the outer wall radially'),
    ('f13_m2', F13 + '; par.m = 2', 'forcing 13 needs m = 0'),
    ('f13_steady_free', F13 + '; ' + STEADY + '; par.bci = 0; par.bco = 0',
     'needs a no-slip wall (bci = 1 or bco = 1) or the drag kappa_cmb'),
    ('f13_steady_bc2', F13 + '; ' + STEADY + '; par.bci = 2; par.bco = 2',
     'needs a no-slip wall (bci = 1 or bco = 1) or the drag kappa_cmb'),
    ('f14_m3', 'par.forcing = 14; par.m = 3', 'forcing 14 needs (m, symm) = (0, 1), (1, -1) or (2, 1)'),
    ('f14_slip', 'par.forcing = 14; par.m = 2; par.bco = 0', 'forcing 14 needs no-slip walls'),
    ('steady_entropy', F9 + '; ' + TH + '; ' + STEADY + '; par.ThermaD = 0.0',
     'forcing 9 at frequency 0 with thermal = 1 needs ThermaD > 0'),
    ('steady_composition', F9 + "; par.compositional = 1; par.comp_stratification = 'linear'; " + STEADY + '; par.CompoD = 0.0',
     'forcing 9 at frequency 0 with compositional = 1 needs CompoD > 0'),
    ('steady_scan', 'par.forcing = 11; par.m = 1; par.symm = -1; ' + TH + '; par.forcing_frequency = 0.5; '
                    'par.rsf_frequencies = [0.5, 0.0]; par.ThermaD = 0.0', '(a frequency of its scan)'),
    ('steady_m0_bgflow', 'par.forcing = 12; par.m = 0; par.forcing_amplitude_cmb = 0.01; ' + TH + '; ' + STEADY +
                         '; par.ThermaD = 0.0; ' + BG, 'forcing 12 at frequency 0 with thermal = 1 needs ThermaD > 0'),
    ('drag_f9', F9 + '; par.bco = 0; par.kappa_cmb = 0.05', 'the wall moves radially (forcing_amplitude_cmb != 0)'),
    # (NEW 2026-11-09) where the fluid does not carry the scalar round (check_transport), and steady runs with a wall of zero
    # radial stress
    ('steady_meridional', F9 + '; ' + TH + '; ' + STEADY + "; par.ThermaD = 0.0; par.bgflow = 1; par.bgflow_Ro = 0.1; "
                          "par.bgflow_profiles = {'P2': '(r - 0.35)**2*(1 - r)**2'}", 'the meridional circulation'),
    ('steady_ro0', F9 + '; ' + TH + '; ' + STEADY + "; par.ThermaD = 0.0; par.bgflow = 1; par.bgflow_Ro = 0.0; "
                   "par.bgflow_profiles = {'T1': 'r**2'}", 'and w + m Ω = 0 everywhere'),
    ('steady_rigid0', F9 + '; ' + TH + '; ' + STEADY + '; par.ThermaD = 0.0; ' + DR + '; par.diff_rot_ampl = 0.0',
     'and w + m Ω = 0 everywhere'),
    ('doppler_rigid', F9 + '; ' + TH + '; par.forcing_frequency = -0.2; par.ThermaD = 0.0; ' + DR,
     'forcing 9 at frequency -0.2 with thermal = 1 needs ThermaD > 0'),
    ('steady_y20', F9 + '; ' + TH + '; ' + STEADY + "; par.ThermaD = 0.0; par.diff_rot = 1; par.diff_rot_type = 'Y20'; "
                   "par.diff_rot_ampl = 0.1", 'vanishes inside the fluid'),
    ('scan_doppler', F11 + '; ' + TH + '; par.forcing_frequency = 0.3; par.rsf_frequencies = [0.3, -0.1]; '
                     'par.ThermaD = 0.0; ' + DR, 'at frequency -0.1 (a frequency of its scan) with thermal = 1'),
    ('steady_cmp_ro0', F9 + "; par.compositional = 1; par.comp_stratification = 'linear'; " + STEADY + "; par.CompoD = 0.0; "
                       "par.bgflow = 1; par.bgflow_profiles = {'T1': 'r**2'}", 'with compositional = 1 needs CompoD > 0'),
    ('f13_steady_bci2', F13 + '; ' + STEADY + '; par.bci = 2', 'forcing 13 at frequency 0 needs walls that stay in place'),
    ('f11_steady_bco2', F11 + '; ' + STEADY + '; par.bco = 2', 'forcing 11 at frequency 0 needs walls that stay in place'),
    ('runs_steady_conical', F9 + '; ' + TH + '; ' + STEADY + "; par.ThermaD = 0.0; par.diff_rot = 1; "
                            "par.diff_rot_type = 'conical'; par.diff_rot_ampl = -0.25", None),
    ('runs_steady_azimuthal', F9 + '; ' + TH + '; ' + STEADY + "; par.ThermaD = 0.0; par.bgflow = 1; par.bgflow_Ro = 0.1; "
                              "par.bgflow_profiles = {'T1': 'r**2'}", None),
    ('runs_f9_bc2_inner', F9 + '; par.bco = 2; par.forcing_amplitude_cmb = 0.0; par.forcing_amplitude_icb = 1.0', None),
    # (NEW 2026-11-09) forcing 11's own conditions, which only reynolds_stress.Builder tested, in assemble.py
    ('f11_inviscid', F11 + '; par.ricb = 0; ' + INV, 'forcing = 11 needs ViscosD > 0'),
    ('f11_taper', F11 + '; par.rsf_taper = 0.0', 'rsf_taper (and rsf_taper_in with ricb = 0) must be > 0'),
    ('f11_taper_in', F11 + '; par.ricb = 0; par.rsf_taper_in = 0.0', 'rsf_taper (and rsf_taper_in with ricb = 0) must be > 0'),
    ('f11_rcut', F11 + '; par.rsf_rcut = 0.1', 'rsf_rcut must lie in [ricb, rcmb - rsf_rcut_width)'),
    ('f11_rcut_width', F11 + '; par.rsf_rcut = 0.5; par.rsf_rcut_width = 0.0', 'rsf_rcut_width must be > 0'),
    ('f11_theta', F11 + "; par.rsf_theta = 'eddy'", "rsf_theta must be 'nu', 'mlt' or a number"),
    ('f11_mlt', F11 + "; par.rsf_theta = 'mlt'", "rsf_theta = 'mlt' needs rsf_lum"),
    ('f11_mlt_visc_pow', F11 + "; par.rsf_theta = 'mlt'; par.rsf_lum = 1.0; par.visc_pow = 1.0", 'which needs visc_pow = 0'),
    ('f11_sphere_rpower', F11 + '; par.ricb = 0; par.rpower_u = 3', 'needs rpower_u >= 5 and rpower_v >= 3'),
    ('runs_f9_sphere_bci0', F9 + '; par.ricb = 0; par.bci = 0', None),
    ('runs_f14_sphere_bci0', 'par.forcing = 14; par.m = 2; par.ricb = 0; par.bci = 0', None),
    ('runs_f7_m2_inviscid', 'par.forcing = 7; par.m = 2; par.ricb = 0; par.bco = 0; ' + INV, None),
    ('runs_f9_inviscid', F9 + '; par.ricb = 0; par.bco = 0; ' + INV, None),
    ('runs_f14_inviscid', 'par.forcing = 14; par.m = 2; par.ricb = 0; par.bco = 0; ' + INV, None),
    ('runs_f13_steady_drag', F13 + '; ' + STEADY + '; par.bci = 0; par.bco = 0; par.kappa_cmb = 0.05', None),
    ('runs_f9_drag', F9 + '; par.bco = 0; par.kappa_cmb = 0.05; par.forcing_amplitude_cmb = 0.0; '
                     'par.forcing_amplitude_icb = 1.0', None),
    ('runs_steady_diffrot', F9 + '; ' + TH + '; ' + STEADY + '; par.ThermaD = 0.0; ' + DR, None),
    ('runs_f11_sphere', F11 + '; par.ricb = 0', None),
    ('runs_f11_cut', F11 + '; par.rsf_rcut = 0.5; par.rsf_taper_in = 0.0', None),
    ('runs_f11_mlt', F11 + "; par.rsf_theta = 'mlt'; par.rsf_lum = 1.0", None),
    ('runs_f1_inviscid_bc2', 'par.forcing = 1; par.m = 1; par.symm = -1; par.ricb = 0; par.bco = 2; ' + INV, None),
    ('runs_f8_inviscid_bc2', 'par.forcing = 8; par.m = 0; par.ricb = 0; par.forcing_amplitude_icb = 0.0; par.bco = 2; ' + INV, None),
    # (NEW 2026-11-09) three settings outside the forcings that the same audit relaxed: a thin layer and the resolved mantle
    # around an inviscid fluid at bco = 2 (its one wall row is no penetration, whatever bco: A, B, the solution and
    # spin_doctor's files are those of bco = 0, bitwise), and a uniform ΔΩ, which winds up no field, under 'refuse'
    ('runs_twa_inviscid_bc2', MAG + "; par.mantle = 'TWA'; par.c_cmb = 0.01; par.c1_cmb = 0.01; par.ricb = 0; par.bco = 2; "
                              + INV, None),
    ('runs_man_inviscid_bc2', MAG + "; par.mantle = 'resolved'; par.ricb = 0; par.bco = 2; " + INV, None),
    ('runs_rigid_dr_refuse', MAG + "; par.ferraro_policy = 'refuse'; " + DR, None),
    ('twa_bc2', MAG + "; par.mantle = 'TWA'; par.c_cmb = 0.01; par.c1_cmb = 0.01; par.bco = 2",
     "a thin conducting layer ('TWA') needs bco, bci = 0 or 1 at its wall"),
]

fail = 0
for name, extra, msg in CASES:
    rc, out = stops(name, extra)
    ok = (rc == 0) if msg is None else (rc != 0 and msg in out)
    fail |= not ok
    last = (out.strip().splitlines() or ['(no output)'])[-1]
    print('%-22s %s  (%s)' % (name, 'ok' if ok else 'FAIL', 'runs' if msg is None and rc == 0 else last[:100]))
print('stops_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
