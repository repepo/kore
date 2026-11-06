#!/usr/bin/env python3
'''
usage: rotdyn_check.py [WORKDIR]   (default: tests/rotdyn/runs)

The rotating solids (rotdyn = 1: bin/rotdyn.py, the unknowns omega_M, omega_IC and phi) and the angular momentum of the
fluid (spin_doctor's resid𝐋), against exact results, against the conductive_IC branch and against the conservation of
angular momentum and energy. Exit code 1 if a check fails.

1. Decoupled solids (vtrq = 0, no field): the solids feel only each other, and two modes are the roots of
   lambda² + c lambda + K (1/I_M + 1/I_IC) = 0 (c = Gaspard/OmgtauIC, K = gTorque Gaspard²); the others are the modes of
   fixed walls (rotdyn = 0); 1e-12.
2. Heavy solids (MoIZ_M = MoIZ_IC = 1e12): the eigenvalues of fixed walls (1e-10).
3. Against conductive_IC 2d84351 for the same runs ('Boussinesq', N 48, lmax 19, 2 ranks): without a field (two sets of
   solids), thin walls with a dipole, a conducting inner core and a resolved mantle with a field of degree 2, an axial
   field with an exponential resolved mantle and a thin wall on the inner core. The first two eigenvalues (1e-10), and for
   the first the rotation of the solids and the torques on the mantle relative to omega_M (rotdyn.dat, 1e-8).
4. Conservation ('Wu 2005', r_out 0.9, N 64). Every mode but the rigid rotation of everything (lambda = 0) carries no
   total angular momentum, L + I_M omega_M + I_IC omega_IC = 0, and closes resid𝐋; without a field it closes the energy
   budget of the fluid and the solids
       2 sigma (KE + I_M |omega_M|² + I_IC |omega_IC|² + K |phi|²) = (the power of the viscous force) + W,
       W = 2 Re(omega_M* G_M + omega_IC* G_IC) - 2 K c |phi|²   (G the viscous torques, flow.dat and rotdyn.dat),
   with a field, a conducting inner core and a resolved mantle it closes resid𝐄; 1e-8.
5. The equatorial resid𝐋 (m = 1, symm = -1, rotdyn = 0), lambda L = (the torques on the fluid) + i Gaspard L: 'Wu 2005'
   without and with a field (a conducting inner core and a resolved mantle), thin walls with a dipole, a toroidal B0 that
   does not vanish at the walls, a full sphere with a field; 1e-8.
6. Stops: rotdyn = 1 with m = 1, with a stress-free wall, with a differential rotation, with a vanishing moment of inertia,
   with vtrq = 2 and with a toroidal B0 that does not vanish at the walls stops in submatrices.py with its message.
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


def build(name, extra, lmax):
    '''mk_run.build in W/name with that lmax on 2 ranks (1 if NCPUS = 1), as the conductive_IC runs'''
    d = os.path.join(W, name)
    T0, n0 = mk_run.T, mk_run.NCPUS
    mk_run.T, mk_run.NCPUS = T0.replace(LMAX_LINE, 'par.lmax = %d' % lmax), min(n0, 2)
    try:
        mk_run.build(d, extra)
    finally:
        mk_run.T, mk_run.NCPUS = T0, n0
    return d

def eig(d):
    e = np.loadtxt(os.path.join(d, 'eigenvalues0.dat')).reshape(-1, 2)
    return e[:, 0] + 1j*e[:, 1]

def table(d, name):
    return np.loadtxt(os.path.join(d, name), ndmin=2)

def solids(d):
    '''rotdyn.dat as complex columns: omega_M, omega_IC, phi, K phi, and the viscous and magnetic torques on the mantle and
    on the inner core (vtorq, mtorq, vtorq_ic, mtorq_ic)'''
    a = table(d, 'rotdyn.dat')
    return a[:, 0::2] + 1j*a[:, 1::2]

def closest(lam, ref):
    '''the largest relative distance from a value of ref to the nearest of lam'''
    return max(np.min(abs(lam - z))/abs(z) for z in ref)

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-16s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))


SOL = ("par.N = 48; par.m = 0; par.symm = 1; par.rotdyn = 1; par.gTorque = 1e-3; par.MoIZ_M = 1.0; par.MoIZ_IC = 0.05; "
       "par.OmgtauIC = 50.0")
MAG = "par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3"
TARGET = "par.rtau = -0.01; par.itau = 0.14"

# 1. decoupled solids, and 2. heavy solids: the fixed walls' modes
fixed = eig(build('fixed', "par.N = 48; par.m = 0; par.symm = 1; %s; par.nev = 3" % TARGET, 19))
lam = eig(build('decoupled', "%s; par.vtrq = 0; %s; par.nev = 6" % (SOL, TARGET), 19))
K, c, I = 1e-3, 1/50.0, 1/1.0 + 1/0.05
pair = np.roots([1.0, c, K*I])
e1, e2 = closest(lam, pair), closest(lam, fixed)
report('decoupled', e1 < 1e-12 and e2 < 1e-12, 'the solids %.1e from the roots, the fluid %.1e from fixed walls' % (e1, e2))
lam = eig(build('heavy', "%s; par.MoIZ_M = 1e12; par.MoIZ_IC = 1e12; %s; par.nev = 7" % (SOL, TARGET), 19))
err = closest(lam, fixed)
report('heavy', err < 1e-10, 'eigenvalues %.1e from fixed walls' % err)

# 3. against conductive_IC 2d84351 (Ek 1e-3, and Em 1e-3, Le² 1e-3, cnorm 'mag_energy' with a field): the first two
#    eigenvalues, and for the first omega_IC, phi and the viscous and magnetic torques on the mantle relative to omega_M
cic = [
    ('cic_hydro', "par.rtau = -0.04; par.itau = 0.13; par.nev = 6",
     [-3.788583808108039047e-02+1.349815659476278074e-01j, -1.029752311320160929e-01+0.000000000000000000e+00j],
     [(-28.62665940992845+2.308900416552629j), (45.391584144763556+213.47205956146246j),
      (-0.08327742222587321-0.0784904936137785j), 0j]),
    ('cic_earth', "par.gTorque = 5e-8; par.MoIZ_M = 11.16; par.MoIZ_IC = 9.2e-3; par.OmgtauIC = 2.3e3; par.itau = 0.5; "
                  "par.nev = 5",
     [-3.799896343006742283e-01+6.202225389499711472e-01j, -3.101345418025624467e-01+7.688687301684058450e-01j],
     [(-118.93302552372998+8.738569951514988j), (96.34456196563082+134.41139971577402j),
      (-4.240689136023997+6.921676814111331j), 0j]),
    ('cic_twa', MAG + "; par.B0 = 'G21 dipole'; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; "
                      "par.innercore = 'TWA'; par.c_icb = 0.0; par.c1_icb = 0.05; par.itau = 0.2; par.nev = 6",
     [-4.087234939849557214e-02+1.342106124347179907e-01j, -1.042561910907709305e-01+2.928498065329512845e-02j],
     [(-28.1689745035294+1.7912315107915047j), (46.03321235943439+210.1782616106384j),
      (-0.08422283018613747-0.07811053236736804j), (-0.002682731571778136+0.002142883191420834j)]),
    ('cic_s2', MAG + "; par.B0 = 'Luo_S2'; par.B0_l = 2; par.innercore = 'conducting, Chebys'; par.N_cic = 24; "
                     "par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 24; par.sigma_mantle = 0.5; "
                     "par.itau = 0.2; par.nev = 6",
     [-4.381668930140437956e-02+1.405750203432453160e-01j, -1.106292989375452090e-01+4.810650578854308579e-02j],
     [(-31.580984028498005-3.102498167581812j), (16.717231631937768+228.93708169601499j),
      (-0.06105147040265271-0.08941955493338231j), (0.0005175494691492492+0.0010574935805339586j)]),
    ('cic_axial', MAG + "; par.B0 = 'axial'; par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 32; "
                        "par.sigma_mantle = '0.5*exp(-p*(r - 1))'; par.sigma_mantle_param = 5.0; par.innercore = 'TWA'; "
                        "par.c_icb = 0.0; par.c1_icb = 0.05; par.itau = 0.15; par.rtau = -0.02; par.nev = 6",
     [-3.878340932687934089e-02+1.349213298303400255e-01j, -3.152940674192462184e-02+3.649941929967093213e-04j],
     [(-28.19657337936293+2.961984909863171j), (51.089481164335844+209.28446805158612j),
      (-0.08288347426623066-0.08229144146405068j), (-0.006989416225018771+0.007928303242731779j)]),
]
for name, extra, ref, rat in cic:
    d = build(name, SOL + '; ' + extra, 19)
    lam = eig(d)
    err = closest(lam, np.array(ref))
    s = solids(d)[np.argmin(abs(lam - ref[0]))]
    got = np.array([s[1], s[2], s[4], s[5]])/s[0]
    er2 = max(abs(got - rat))/max(abs(np.array(rat)))
    report(name, err < 1e-10 and er2 < 1e-8, 'eigenvalues %.1e, rotdyn.dat %.1e from conductive_IC' % (err, er2))

# 4. conservation, 'Wu 2005': the total angular momentum, resid𝐋, and the energy budget (resid𝐄 with a field)
WU = "par.structure = 'Wu 2005'; par.r_out = 0.9; par.N = 64"
WU_MAG = (MAG + "; par.B0 = 'G21 dipole'; par.innercore = 'conducting, Chebys'; par.N_cic = 24; par.mantle = 'resolved'; "
          "par.r_surface = 1.3; par.N_mantle = 24; par.sigma_mantle = '0.5*exp(-p*(r - 1))'; par.sigma_mantle_param = 5.0")
for name, extra in (('wu_hydro', "par.rtau = -0.04; par.itau = 0.13; par.nev = 6"),
                    ('wu_field', WU_MAG + "; par.itau = 0.2; par.nev = 6")):
    d = build(name, SOL + '; ' + WU + '; ' + extra, 19)
    lam, f, s = eig(d), table(d, 'flow.dat'), solids(d)
    keep = abs(lam) > 1e-8                    # not the rigid rotation of everything
    L = f[:, 28] + 1j*f[:, 29]
    parts = np.array([L, 1.0*s[:, 0], 0.05*s[:, 1]])
    Ltot = (abs(parts.sum(0))/abs(parts).max(0))[keep].max()
    rL = f[keep, 38].max()
    if 'magnetic' in extra:
        rE = table(d, 'magnetic.dat')[keep, 10].max()
        what = 'resid𝐄'
    else:
        E = f[:, 0] + 1.0*abs(s[:, 0])**2 + 0.05*abs(s[:, 1])**2 + 1e-3*abs(s[:, 2])**2
        Wsol = 2*np.real(np.conj(s[:, 0])*s[:, 4] + np.conj(s[:, 1])*s[:, 6]) - 2*1e-3*(1/50.0)*abs(s[:, 2])**2
        terms = np.array([2*lam.real*E, f[:, 3], Wsol])
        rE = (abs(terms[0] - terms[1] - terms[2])/abs(terms).max(0))[keep].max()
        what = 'energy'
    report(name, Ltot < 1e-8 and rL < 1e-8 and rE < 1e-8,
           'total angular momentum %.1e, resid𝐋 %.1e, %s %.1e (%d modes)' % (Ltot, rL, what, rE, keep.sum()))

# 5. the equatorial resid𝐋 (m = 1, symm = -1, fixed walls)
EQ = "par.N = 48; par.m = 1; par.symm = -1; par.itau = 1.0; par.nev = 6"
for name, extra in (('eq_wu', WU),
                    ('eq_wu_field', WU + '; ' + WU_MAG),
                    ('eq_twa', MAG + "; par.B0 = 'G21 dipole'; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; "
                                     "par.innercore = 'TWA'; par.c_icb = 0.0; par.c1_icb = 0.05"),
                    ('eq_toroidal', WU + '; ' + MAG + "; par.B0 = None; par.B0_tor = 'Malkus'; "
                                                      "par.innercore = 'conducting, Chebys'; par.N_cic = 24"),
                    ('eq_sphere', WU + '; ' + MAG + "; par.ricb = 0.0; par.B0 = 'G21 dipole'")):
    d = build(name, EQ + '; ' + extra, 20)
    rL = table(d, 'flow.dat')[:, 38].max()
    report(name, rL < 1e-8, 'resid𝐋 %.1e' % rL)

# 6. the stops of background.check_rotdyn
def stops(name, extra, words):
    '''whether submatrices.py stops (background.check) with a message that contains words'''
    d = os.path.join(W, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    open(d + '/bin/parameters.py', 'w').write(mk_run.T.format(ncpus=1, extra=(SOL + '; ' + extra).replace('; ', '\n')))
    p = subprocess.run([sys.executable, 'bin/submatrices.py', '1'], cwd=d, capture_output=True, text=True)
    return p.returncode != 0 and words in p.stdout + p.stderr

cases = [('m = 1', "par.m = 1", 'rotdyn = 1 needs the flow'),
         ('stress-free wall', "par.bco = 0", 'rotdyn = 1 needs a viscous fluid with no-slip walls'),
         ('diff_rot', "par.diff_rot = 1", 'rotdyn = 1 with a differential rotation'),
         ('MoIZ_IC = 0', "par.MoIZ_IC = 0.0", 'MoIZ_IC = 0.0'),
         ('vtrq = 2', "par.vtrq = 2", 'vtrq = 2'),
         ('B0_tor Malkus', MAG + "; par.B0 = None; par.B0_tor = 'Malkus'", 'rotdyn = 1 needs a toroidal B0 that vanishes')]
bad = [name for k, (name, extra, words) in enumerate(cases) if not stops('stop_%d' % k, extra, words)]
report('stops', not bad, '%d of %d stop with their message%s' % (len(cases) - len(bad), len(cases),
                                                                   '; not: ' + ', '.join(bad) if bad else ''))

print('rotdyn_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
