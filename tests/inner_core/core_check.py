#!/usr/bin/env python3
'''
usage: core_check.py [WORKDIR]   (default: tests/inner_core/runs)

The field of a dipole at the centre (B0 = 'dipole') and a fluid slipping along a conducting inner core (bci = 0 with
innercore 'conducting, Chebys' or 'conducting, Bessel': utils.cic_slips, the u×B0 of the slip in the core's electric-field
row, assemble.bc_b_slip), against the conductive_IC branch and between the two cores. Exit code 1 if a check fails.

1. B0 'dipole' against the eigenvalues of conductive_IC 2d84351 for the same runs ('Boussinesq', Ek 1e-3, Em = Le = 1e-2,
   cnorm 1, N 48, lmax 20 or 19, 2 ranks): insulating no-slip walls, cnorm 'mag_energy', the antisymmetric modes (and
   resid𝐋), a thin conducting layer at the outer wall (B0_cmb 'wall'), a resolved mantle, a thin layer on the inner core,
   the rotating solids with a conducting inner core: the first two eigenvalues to 1e-10 (the r^-2 series of h is cut at
   tol_B0, which leaves ~4e-12). The libration of the mantle (forcing 7) with 'conducting, Bessel': the kinetic energy to
   1e-9 (flow.dat has 10 digits).
2. A fluid slipping along a conducting inner core, against conductive_IC: 'axial' with the antisymmetric modes of m 1, and
   with m 0 and resid𝐋 (Ek = Em = Le² = 1e-3, one rank), the dipole (Em = Le = 1e-2), 'G21 dipole' with mu_i2o 2 and
   sigma_i2o 0.3, 'Luo_S2' with both walls stress-free and sigma_i2o 0.5: the first two eigenvalues to 1e-10, and resid𝐄
   to 1e-8 (resid𝐛 cannot see the term; resid𝐄 is 7e-5 to 4e-2 without it). Without a reference: resid𝐄 to 1e-8 with
   'Wu 2005' (N 96, N_cic 48, with buoyancy: the budget with a variable density; the slip's (ln ρ0)' term needs a radial
   flow at the wall, forcing 6 in 3.), and a rigid rotation offset (diff_rot_type 'rigid', 0.3, both walls stress-free)
   against the frame turning with the solids: the eigenvalues of the modes found in both to 1e-12 (at least nev of them),
   resid𝐄 the same to 1e-11. just_induction.py and solve_ind.py give the field, in the fluid and in the core, of a mode whose Lorentz
   force vanishes (Hendrik 1e-30), from its flow and eigenvalue (1e-10).
3. Forced runs with a slipping inner wall: the libration of the mantle (forcing 7, m 0) with 'axial' and with the dipole,
   both cores against conductive_IC's kinetic energy (1e-9) and against each other (the flow and the field, 1e-11);
   forcing 10 with 'G21 dipole', sigma_i2o 0.3 and mu_i2o 1.5 (where 1/sigma_i2o in place of sigma_i2o, or a missing
   mu_i2o, would show), forcing 6 (the inner wall moving, the core at rest) and forcing 10 with a differential rotation
   (N 64), Bessel against Chebys (1e-11); resid𝐄 to 1e-9 in all, and with forcing 6 under forced_walls 'deforming' (with
   'axial' and with the dipole, which the deforming core holds as a x + b x³), and with 'Wu 2005' (the slip's
   (ln ρ0)' term: 2e-3 without it).
4. Stops (submatrices.py): B0 'dipole' in a full sphere, 'general' and 'user' profiles that are not finite at r = 0 in a
   full sphere (the where() form of the message runs), a zero toroidal field normalised by its energy and a zero
   'ferraro' field (its own cnorm) and a NaN one, a conducting inner core with bci = 2, centrifugal = 1 with a fluid slipping along a
   conducting inner core, the libration of an inner core behind a stress-free wall along a conducting core, and a
   toroidal B0 that is not 0 at a slipping wall with an insulator beyond (its current sheet there, background.sheet_jump):
   an inviscid full sphere, also with a sheet of 1e-6 under a strong 'axial' field (b0_at_wall's cut would miss it) and
   with one that an irregular profile leaves at the wall (the message says why), and rotdyn = 1 (its own stop). The same
   sheet at a viscous slipping wall runs, its push in the stress rows (phase 6c, tests/sheet): stress-free outer and inner
   walls (bc 0 and 2), a thin layer that does not conduct at either wall; and so do no-slip walls, a conducting core, a
   conducting layer at either wall, the resolved mantle and B0_tor 'wall'.
'''
import os, sys, shutil, subprocess
import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, NCPUS ranks)

W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
LMAX_LINE = 'par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)'
assert LMAX_LINE in mk_run.T


def build(name, extra, lmax, ncpus=2):
    '''mk_run.build in W/name with that lmax on ncpus ranks (fewer if NCPUS is smaller), as the conductive_IC runs'''
    d = os.path.join(W, name)
    T0, n0 = mk_run.T, mk_run.NCPUS
    mk_run.T, mk_run.NCPUS = T0.replace(LMAX_LINE, 'par.lmax = %d' % lmax), min(n0, ncpus)
    try:
        mk_run.build(d, extra)
    finally:
        mk_run.T, mk_run.NCPUS = T0, n0
    return d

def eig(d):
    e = np.loadtxt(os.path.join(d, 'eigenvalues0.dat')).reshape(-1, 2)
    return e[:, 0] + 1j*e[:, 1]

def col(d, name, c):
    '''column c (from 1) of flow.dat or magnetic.dat'''
    return np.loadtxt(os.path.join(d, name), ndmin=2)[:, c - 1]

def top(x):
    x = np.asarray(x, dtype=float)
    return np.nan if np.all(np.isnan(x)) else np.nanmax(x)

def closest(lam, ref):
    '''the largest relative distance from a value of ref to the nearest of lam'''
    return max(np.min(abs(lam - z))/abs(z) for z in ref)

def field(d, kind):
    return np.loadtxt(os.path.join(d, 'real_%s.field' % kind)) + 1j*np.loadtxt(os.path.join(d, 'imag_%s.field' % kind))

def apart(d1, d2):
    '''the largest relative difference of the flow and of the field of two forced runs'''
    return max(np.max(abs(field(d1, k) - field(d2, k)))/np.max(abs(field(d1, k))) for k in ('flow', 'magnetic'))

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-20s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))


# 1. B0 'dipole' against conductive_IC 2d84351 (Em = Le = 1e-2: Hendrik 1e-4, MagnetD 1e-2; cnorm 1)
DIP = ("par.N = 48; par.magnetic = 1; par.B0 = 'dipole'; par.cnorm = 1; par.Hendrik = 1e-4; par.MagnetD = 1e-2; "
       "par.m = 1; par.symm = 1")
dipole = [
    ('dip_insulating', 20, "",
     [-2.422030520446791235e-01+1.218609167082208566e+00j, -3.170077416191694475e-01+9.003926962446170412e-01j]),
    ('dip_mag_energy', 20, "par.cnorm = 'mag_energy'",
     [-2.188284397961023364e-01+1.168056074674111766e+00j, -2.819312238003884308e-01+8.678664536708633470e-01j]),
    ('dip_antisym', 20, "par.symm = -1",
     [-1.026322242802502477e-01+1.009493835011314955e+00j, -3.417536795398849470e-01+1.061112708404809313e+00j]),
    ('dip_twa_mantle', 20, "par.mantle = 'TWA'; par.c_cmb = 0.01; par.c1_cmb = 0.01; par.muw_cmb = np.inf",
     [-2.424041160541642037e-01+1.218549057058673890e+00j, -3.171134255545542402e-01+9.003857170778680752e-01j]),
    ('dip_resolved_mantle', 20, "par.mantle = 'resolved'; par.r_surface = 1.5; par.N_mantle = 24; par.sigma_mantle = 0.1",
     [-2.426557072079254529e-01+1.218558993043874272e+00j, -3.171263190208299387e-01+9.003814242446328553e-01j]),
    ('dip_twa_core', 20, "par.innercore = 'TWA'; par.c_icb = 0.01; par.c1_icb = 0.01",
     [-2.434696469437559463e-01+1.218566383497649008e+00j, -3.177192243876923450e-01+9.000773961947721169e-01j]),
    ('dip_rotdyn', 19, "par.m = 0; par.rotdyn = 1; par.innercore = 'conducting, Chebys'; par.N_cic = 32",
     [-2.976314189773586194e-01+1.041944921162055726e+00j, -1.642808989606774628e-01+1.345469510481451358e+00j]),
]
for name, lmax, extra, ref in dipole:
    d = build(name, DIP + ('; ' + extra if extra else ''), lmax)
    err = closest(eig(d), np.array(ref))
    text = 'eigenvalues %.1e from conductive_IC' % err
    ok = err < 1e-10
    if name == 'dip_antisym':   # resid𝐋 closes with no term for the source of the dipole (rotdyn.wall_torques)
        rL = top(col(d, 'flow.dat', 39))
        ok, text = ok and rL < 1e-10, text + ', resid𝐋 %.1e' % rL
    report(name, ok, text)

F7 = "par.m = 0; par.forcing = 7; par.forcing_frequency = 0.7; par.forcing_amplitude_cmb = 1.0"
d = build('dip_libration', DIP + "; par.innercore = 'conducting, Bessel'; " + F7, 19)
err = abs(col(d, 'flow.dat', 1)[0]/2.795315559790952781e-02 - 1)
report('dip_libration', err < 1e-9, 'kinetic energy %.1e from conductive_IC' % err)


# 2. a fluid slipping along a conducting inner core (bci = 0), against conductive_IC
AX = "par.N = 48; par.magnetic = 1; par.cnorm = 1; par.Hendrik = 1e-3; par.MagnetD = 1e-3"   # Em = Le² = 1e-3
G21 = ("par.N = 48; par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3; "
       "par.B0 = 'G21 dipole'; par.m = 1; par.symm = 1; par.N_cic = 24")
CORE = "par.innercore = 'conducting, Chebys'; par.bci = 0"
slip = [
    ('slip_axial_antisym', 20, 1, AX + "; par.B0 = 'axial'; par.m = 1; par.symm = -1",
     [-1.096524225739990560e-01+1.017814396616456563e+00j, -3.343127154474607177e-01+1.076395403812852480e+00j]),
    ('slip_axial_m0', 21, 1, AX + "; par.B0 = 'axial'; par.m = 0; par.symm = 1",
     [-2.466757577476881291e-01+1.015039547225914385e+00j, -3.127709012469342409e-01+7.997599553589318511e-01j]),
    ('slip_dipole', 20, 2, DIP + "; par.N_cic = 32",
     [-3.156140709631607977e-01+8.993039682745509467e-01j, -2.554804350123606183e-01+1.225825309549912756e+00j]),
    ('slip_mu_sigma', 20, 2, G21 + "; par.sigma_i2o = 0.3; par.mu_i2o = 2.0",
     [-2.067503451335563625e-01+1.171802788550381891e+00j, -2.673002492147625309e-01+8.750343421402363431e-01j]),
    ('slip_both_walls', 20, 2, G21.replace("'G21 dipole'", "'Luo_S2'; par.B0_l = 2") + "; par.bco = 0; par.sigma_i2o = 0.5",
     [-1.384517157210344018e-01+8.817768532312846297e-01j, -1.116491620275695124e-01+1.173823315462261263e+00j]),
]
for name, lmax, n, extra, ref in slip:
    d = build(name, extra + '; ' + CORE, lmax, n)
    err = closest(eig(d), np.array(ref))
    rE, rb = top(col(d, 'magnetic.dat', 11)), top(col(d, 'magnetic.dat', 4))
    ok, text = err < 1e-10 and rE < 1e-8, 'eigenvalues %.1e from conductive_IC; resid𝐄 %.1e, resid𝐛 %.1e' % (err, rE, rb)
    if name == 'slip_axial_m0':
        rL = top(col(d, 'flow.dat', 39))
        ok, text = ok and rL < 1e-10, text + ', resid𝐋 %.1e' % rL
    report(name, ok, text)

WU = ("par.thermal = 1; par.structure = 'Wu 2005'; par.stratification = 'top hat'; par.r_out = 0.9; par.rpower_u = 5; "
      "par.rhopower_u = 1; par.rpower_v = 3; par.rhopower_v = 1; par.rpower_pp = 4; par.rhopower_pp = 4; par.N = 96; "
      "par.N_cic = 48")   # (N_cic 24 leaves 2e-8 in the most damped mode)
d = build('slip_wu', G21 + '; ' + CORE + '; ' + WU, 20, 1)
rE = top(col(d, 'magnetic.dat', 11))
report('slip_wu', rE < 1e-8, "resid𝐄 %.1e with 'Wu 2005'" % rE)

OFF = G21 + '; ' + CORE + '; par.bco = 0; par.nev = 4'
do = build('slip_offset', OFF + "; par.diff_rot = 1; par.diff_rot_type = 'rigid'; par.diff_rot_ampl = 0.3; par.itau = 0.7",
           20, 1)
df = build('slip_frame', OFF + '; par.Gaspard = 1.3', 20, 1)
lo, lf = eig(do), eig(df) - 0.3j            # the frame's eigenvalues, seen from the frame of the offset run
# solve.py writes every converged pair, often more than nev and not as many in both runs: compare the modes found in both
pairs = [(i, int(np.argmin(abs(lf - z)))) for i, z in enumerate(lo)]
pairs = [(i, j) for i, j in pairs if abs(lf[j] - lo[i]) < 1e-8*abs(lo[i])]
i, j = np.array([p[0] for p in pairs], dtype=int), np.array([p[1] for p in pairs], dtype=int)
err = np.max(abs(lf[j] - lo[i])/abs(lo[i])) if pairs else np.inf
dE = np.max(abs(col(do, 'magnetic.dat', 11)[i] - col(df, 'magnetic.dat', 11)[j])) if pairs else np.inf   # (1e-12 to 1e-8)
report('slip_offset', len(pairs) >= 4 and err < 1e-12 and dE < 1e-11, '%d modes: eigenvalues %.1e from the frame turning '
       'with the solids, resid𝐄 %.1e apart' % (len(pairs), err, dE))

# kinematic induction: just_induction.py and solve_ind.py give the field of a mode whose Lorentz force vanishes, from its flow
d = build('slip_kinematic', AX + "; par.B0 = 'axial'; par.Hendrik = 1e-30; par.MagnetD = 1e-2; par.m = 1; par.symm = 1; "
                            "par.rtau = 0.0; par.itau = 0.5; par.nev = 2; par.N_cic = 24; " + CORE, 20, 1)
beig, ceig, z = field(d, 'magnetic'), field(d, 'magnetic_ic'), eig(d)[0]
py = sys.executable
for cmd in ([py, 'bin/just_induction.py', repr(complex(z))], mk_run.MPIEXEC.split() + ['-n', '1', py, 'bin/solve_ind.py']):
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    if p.returncode:
        sys.exit('%s: %s failed\n%s' % (d, cmd[-1], (p.stdout + p.stderr)[-3000:]))
eb_ = abs(field(d, 'magnetic')[:, 0] - beig[:, 0]).max()/abs(beig[:, 0]).max()
ec_ = abs(field(d, 'magnetic_ic')[:, 0] - ceig[:, 0]).max()/abs(ceig[:, 0]).max()
report('slip_kinematic', eb_ < 1e-10 and ec_ < 1e-10, 'field %.1e in the fluid, %.1e in the core' % (eb_, ec_))


# 3. forced runs with a slipping inner wall: Chebys and Bessel, against conductive_IC (forcing 7) and each other
CHEB, BES = "par.innercore = 'conducting, Chebys'", "par.innercore = 'conducting, Bessel'"
forced = [
    ('f7_axial', 21, 1, AX + "; par.B0 = 'axial'; par.bci = 0; par.m = 0; par.forcing = 7; par.forcing_frequency = 0.8; "
                        "par.forcing_amplitude_cmb = 1.0", 3.598608513602839787e-02),
    ('f7_dipole', 19, 2, DIP + "; par.N_cic = 32; par.bci = 0; " + F7, 2.795721976390480590e-02),
    ('f10_mu_sigma', 21, 2, G21 + "; par.N_cic = 32; par.bci = 0; par.m = 2; par.forcing = 10; par.forcing_frequency = 0.37; "
                            "par.sigma_i2o = 0.3; par.mu_i2o = 1.5", None),
    ('f6_axial', 20, 2, AX + "; par.B0 = 'axial'; par.N_cic = 32; par.bci = 0; par.m = 1; par.symm = -1; par.forcing = 6; "
                        "par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 0.0; par.forcing_amplitude_icb = 1.0", None),
    ('f10_diffrot', 21, 2, AX + "; par.B0 = 'axial'; par.N = 64; par.N_cic = 40; par.bci = 0; par.m = 2; par.forcing = 10; "
                           "par.forcing_frequency = 0.37; par.diff_rot = 1; par.diff_rot_type = 'shellular_boussinesq'; "
                           "par.diff_rot_ampl = 0.4", None),
]
for name, lmax, n, extra, ke in forced:
    dc = build(name + '_cheb', extra + '; ' + CHEB, lmax, n)
    db = build(name + '_bes', extra + '; ' + BES, lmax, n)
    gap = apart(dc, db)
    rE = np.max([top(col(dc, 'magnetic.dat', 11)), top(col(db, 'magnetic.dat', 11))])   # (NaN if either is)
    ok, text = gap < 1e-11 and rE < 1e-9, 'Bessel %.1e from Chebys; resid𝐄 %.1e' % (gap, rE)
    if ke is not None:
        err = max(abs(col(dd, 'flow.dat', 1)[0]/ke - 1) for dd in (dc, db))
        ok, text = ok and err < 1e-9, text + ', kinetic energy %.1e from conductive_IC' % err
    report(name, ok, text)
F6 = "par.m = 1; par.symm = -1; par.forcing = 6; par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 0.0; " \
     "par.forcing_amplitude_icb = 1.0"
for name, extra, n, what in [
        ('f6_deforming', forced[3][3] + '; ' + CHEB + "; par.forced_walls = 'deforming'", 2, 'the core deforming'),
        ('f6_dipole_deforming', DIP + "; par.N_cic = 32; par.bci = 0; " + F6 + '; ' + CHEB + "; par.forced_walls = 'deforming'",
         1, 'the dipole in the core deforming'),
        ('f6_wu', AX + "; par.B0 = 'axial'; par.bci = 0; " + F6 + '; ' + CHEB + '; ' + WU, 1, "'Wu 2005'")]:
    d = build(name, extra, 20, n)
    rE = top(col(d, 'magnetic.dat', 11))
    report(name, rE < 1e-9, 'resid𝐄 %.1e with %s' % (rE, what))


# 4. stops
def stops(name, extra, words):
    '''whether submatrices.py stops with a message that contains words (a string, or a tuple of strings that must all be
    there); words None: whether it goes on'''
    d = os.path.join(W, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    open(d + '/bin/parameters.py', 'w').write(mk_run.T.format(ncpus=1, extra=extra.replace('; ', '\n')))
    p = subprocess.run([sys.executable, 'bin/submatrices.py', '1'], cwd=d, capture_output=True, text=True)
    if words is None:   # it must not stop
        return p.returncode == 0
    return p.returncode != 0 and all(w in p.stdout + p.stderr for w in ((words,) if isinstance(words, str) else words))

SMALL = "par.N = 24; par.magnetic = 1; par.cnorm = 1; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.m = 1"
cases = [
    ('stop_dipole_sphere', SMALL + "; par.B0 = 'dipole'; par.ricb = 0", "B0 'dipole' is singular at r = 0"),
    ('stop_general_sphere', SMALL + "; par.B0 = 'general'; par.B0_profiles = {1: '0.5/r**2'}; par.ricb = 0",
     'is not finite at some of the points where it is interpolated'),
    ('stop_user_sphere', SMALL + "; par.B0 = None; par.B0_tor = 'user'; par.ricb = 0; par.B0_tor_param = 4.4934; "
                         "par.B0_tor_profile = 'sin(p*r)/(p*r)**2 - cos(p*r)/(p*r)'",   # j_1: regular, but 0/0 at r = 0
     ('is not finite at some of the points where it is interpolated', 'or where(r == 0, 0, ...))')),
    ('runs_user_where', SMALL + "; par.B0 = None; par.B0_tor = 'user'; par.ricb = 0; par.B0_tor_param = 4.4934; "
                        "par.B0_tor_profile = 'where(r == 0, 0, sin(p*r)/(p*r)**2 - cos(p*r)/(p*r))'", None),   # the advice
    ('stop_cnorm_tor', SMALL + "; par.B0 = None; par.B0_tor = 'user'; par.B0_tor_profile = '0*r'; par.cnorm_tor = 'mag_energy'",
     'the field is zero, or not finite'),
    ('stop_ferraro_zero', SMALL + "; par.symm = -1; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_ampl = 0.2; "
                          "par.diff_rot_profiles = {0: '(2/3)*r**2 - (8/15)*r**4', 2: '-(2/3)*r**2 + (16/21)*r**4', "
                          "4: '-(8/35)*r**4'}; par.B0 = 'ferraro'; par.B0_ferraro_G = '0*w'; par.B0_ferraro_nmax = 3; "
                          "par.cnorm = 'mag_energy'",   # its cnorm is ferraro_cnorm: inf
     "for B0 'ferraro' (G = '0*w'): the field is zero, or not finite"),
    ('stop_ferraro_nan', SMALL + "; par.symm = -1; par.diff_rot = 1; par.diff_rot_type = 'user'; par.diff_rot_ampl = 0.2; "
                         "par.diff_rot_profiles = {0: '(2/3)*r**2 - (8/15)*r**4', 2: '-(2/3)*r**2 + (16/21)*r**4', "
                         "4: '-(8/35)*r**4'}; par.B0 = 'ferraro'; par.B0_ferraro_G = 'sqrt(-w)'; par.B0_ferraro_nmax = 3",
     "B0 'ferraro' (G = 'sqrt(-w)') is not finite over the fluid"),   # with cnorm 1: its profiles, not its cnorm
    ('stop_bci2', SMALL + "; par.B0 = 'axial'; par.innercore = 'conducting, Chebys'; par.bci = 2", 'needs bci = 0 or 1'),
    ('stop_centrifugal', SMALL + "; par.B0 = 'axial'; " + CORE + "; par.centrifugal = 1; par.centrifugal_shape = 'uniform'; "
                         "par.centrifugal_f = 0.05", 'needs a no-slip inner wall (bci = 1)'),
    ('stop_core_libration', SMALL + "; par.B0 = 'axial'; " + CORE + "; par.m = 0; par.forcing = 7; "
                            "par.forcing_frequency = 0.8; par.forcing_amplitude_icb = 0.5",
     'the u×B0 of the slip relative to the core alone, which is not implemented'),
]
TOR = SMALL + "; par.B0 = None; par.B0_tor = 'Malkus'"   # k = r: not 0 at either wall
INV = "; par.ricb = 0; par.ViscosD = 0; par.m = 2"            # an inviscid full sphere
cases += [   # the current sheet of a toroidal B0 at a slipping wall with an insulator beyond (background.sheet_jump)
    ('stop_sheet_inviscid', TOR + INV, ('B0 ends at the outer wall in a current sheet (the toroidal B0 is not 0 there '
                                        '(degree 1: |k| 1.0e+00, 1.0e+00 at most in the fluid)), with an insulator beyond, '
                                        'and the fluid is inviscid (ViscosD = 0)')),
    # a sheet of 4e-6 of the toroidal field (|k| = 1e-6 at the wall) under an 'axial' field of 100 (h' = 50): b0_at_wall's
    # cut (1e-6 of the largest of h, h' and k) would not see it
    ('stop_sheet_small', SMALL + INV + "; par.B0 = 'axial'; par.cnorm = 100; par.B0_tor = 'user'; par.B0_tor_l = 2; "
                         "par.cnorm_tor = 1; par.B0_tor_profile = 'r**2*(1 - r**2) + 1e-6*r**2'",   # even in r: smooth at 0
     ('the toroidal B0 is not 0 there (degree 2: |k| 1.0e-06, 2.5e-01 at most', 'it does so only to 4e-06 of the field')),
    ('stop_sheet_rotdyn', TOR + '; par.bco = 0; par.m = 0; par.rotdyn = 1',   # check_rotdyn's stop, not the sheet's remedies
     'rotdyn = 1 needs a toroidal B0 that vanishes at both walls'),
    # a profile meant to vanish at the wall that is not smooth at the centre of a full sphere (r|r| at |r|): its
    # interpolant misses the zero at the wall, and the message says why
    ('stop_sheet_irregular', TOR + INV + "; par.B0_tor = 'user'; par.B0_tor_profile = 'r*(1 - r)**2'",
     'or the profile may not be smooth at r = 0'),
    ('runs_sheet_outer', TOR + '; par.bco = 0', None),   # a viscous slipping wall takes the push (phase 6c)
    ('runs_sheet_inner', TOR + '; par.bci = 0', None),
    ('runs_sheet_bco2', TOR + '; par.bco = 2', None),
    ('runs_sheet_bci2', TOR + '; par.bci = 2', None),
    ('runs_sheet_bare_layer', TOR + "; par.bco = 0; par.mantle = 'TWA'; par.c_cmb = 0.01; par.c1_cmb = 0.0", None),
    ('runs_sheet_bare_layer_ic', TOR + "; par.bci = 0; par.innercore = 'TWA'; par.c_icb = 0.01; par.c1_icb = 0.0", None),
    ('runs_sheet_noslip', TOR, None),   # no-slip walls take the push
    ('runs_sheet_core', TOR + '; ' + CORE + '; par.N_cic = 16', None),   # the conductor beyond holds the sheet
    ('runs_sheet_layer', TOR + "; par.bco = 0; par.mantle = 'TWA'; par.c_cmb = 0.01; par.c1_cmb = 1e-3", None),
    ('runs_sheet_layer_ic', TOR + "; par.bci = 0; par.innercore = 'TWA'; par.c_icb = 0.01; par.c1_icb = 1e-3", None),
    ('runs_sheet_mantle', TOR + "; par.bco = 0; par.mantle = 'resolved'", None),
    ('runs_sheet_wall', TOR + "; par.bci = 0; par.bco = 0; par.B0_tor = 'wall'", None),
]
for name, extra, words in cases:
    report(name, stops(name, extra, words), 'stops with its message' if words else 'goes on')

print('core_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
