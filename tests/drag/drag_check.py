#!/usr/bin/env python3
'''
usage: drag_check.py [WORKDIR]   (default: tests/drag/runs)

The drag at a stress-free outer wall (kappa_cmb, topo_drag = 'rekier': bin/topo.py) and the power it dissipates
(spin_doctor's Dtopo and δσtopo), against the conductive_IC branch, against exact limits and against the budgets of energy
and angular momentum. Exit code 1 if a check fails.

1. Against conductive_IC 2d84351 for the same runs ('Boussinesq', N 48, 2 ranks): a constant drag (kappa_cmb) on the
   spin-over mode in a shell and in a full sphere; the drag of the topography on the spin-over mode, on an m = 2 mode
   (topo_omega set) and in a full sphere; the rotating solids with a stress-free outer wall, without a drag, with each drag,
   with thin walls and a dipole, and with a conducting inner core, a resolved mantle and a field of degree 2; the drag of the
   topography with a thin wall and a dipole. The first two eigenvalues (1e-10), and with the solids, for the first, their
   rotation and the torques on the mantle relative to omega_M (rotdyn.dat, 1e-8).
2. Conservation ('Wu 2005', r_out 0.9, N 64) with the rotating solids, kappa_cmb without a field, the drag of the
   topography with a dipole, a conducting inner core and a resolved mantle: every mode but the rigid rotation of everything
   carries no total angular momentum, closes resid𝐋, and closes the energy budget (computed from flow.dat and rotdyn.dat
   without a field, resid𝐄 with); 1e-8.
3. Equivalences. The test hook topo_K = '(0.01 + 0*x, 0.01 + 0*x)' (a constant tensor through the couplings of the
   topography's drag) gives the eigenvalues of kappa_cmb = 0.01 (1e-12). A rigid differential rotation (diff_rot_type
   'rigid', diff_rot_ampl a = 0.3) with kappa_cmb = 0.01 gives the eigenvalues of a frame turning at Gaspard (1 + a) with
   kappa_cmb/(1 + a), minus i m Gaspard a (1e-12): the drag acts on the slip relative to the mantle, in units of the frame's
   rotation rate.
4. Limits. kappa_cmb = 1e8 gives the eigenvalues of a no-slip wall, without and with the rotating solids (and their
   rotdyn.dat), 1e-8. A weak drag (kappa_cmb = 1e-6) dissipates kappa_cmb times the Dtopo/κ of the stress-free run, without
   and with the solids (1e-3). With the mantle at rest it changes the damping rate of the spin-over mode by kappa_cmb times
   its δσtopo/κ (5e-3: δσtopo leaves out the change of the other losses that the drag causes, small here); with the solids
   the drag also turns the mantle, and the change is only printed (about half of it).
5. The frequencies the drag of the topography uses (params.dat): N/Omega = topo_N when set, sqrt(Beyonce g dS/dr)/Gaspard of
   the stratification otherwise ('linear', Beyonce 10: sqrt(10)), w0/Omega = topo_omega when set, |itau|/Gaspard otherwise,
   |forcing_frequency|/Gaspard in a forced run (forcing 6); 1e-9. The same run with a time unit twice as long (Gaspard 2,
   Beyonce 40, ViscosD, ThermaD and itau doubled) has its eigenvalues doubled (1e-10) and the same N/Omega and w0/Omega. The
   forced run closes Dtopo = -Wv_cmb (1e-8), and so do (NEW 2026-11-09) two forced runs with kappa_cmb that stopped before:
   forcing 9 with the outer wall in place (forcing_amplitude_cmb = 0) and forcing 13, steady, between stress-free walls.
6. The power of the drag: what the viscous stress at the outer wall takes from the fluid goes to the drag and to the
   turning mantle, Dtopo = -Wv_cmb - 2 Re(omega_M* G_M) (G_M the viscous torque on the mantle), in every run with a drag but
   the stiff ones of 4, whose slip is round-off; 1e-8.
7. Stops: the settings check_drag does not cover stop in submatrices.py with their message.
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

def ratios(d, lam, z):
    '''omega_IC, phi and the viscous and magnetic torques on the mantle relative to omega_M, for the eigenvalue nearest z'''
    s = solids(d)[np.argmin(abs(lam - z))]
    return np.array([s[1], s[2], s[4], s[5]])/s[0]

def wall_power(d):
    '''the largest relative residual of Dtopo = -Wv_cmb - 2 Re(omega_M* G_M) (flow.dat columns 40, 25, 31-32, rotdyn.dat)
    over the modes but the rigid rotation of everything (lambda = 0)'''
    f, lam = table(d, 'flow.dat'), eig(d)
    G = f[:, 30] + 1j*f[:, 31]
    wM = solids(d)[:, 0] if os.path.exists(os.path.join(d, 'rotdyn.dat')) else 0*G
    terms = np.array([f[:, 39], f[:, 24], 2*np.real(np.conj(wM)*G)])
    keep = abs(lam) > 1e-8
    return (abs(terms.sum(0))/abs(terms).max(0))[keep].max()

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-16s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))


N48 = "par.N = 48; par.nev = 6"
SO = N48 + "; par.m = 1; par.symm = -1; par.bco = 0; par.itau = 0.98"        # the spin-over mode, stress-free outer wall
REK = "par.topo_drag = 'rekier'; par.topo_hrms = 0.02; par.topo_lambda0 = 0.431; par.topo_H = 0.8"
SOL = (N48 + "; par.m = 0; par.symm = 1; par.bco = 0; par.rotdyn = 1; par.gTorque = 1e-3; par.MoIZ_M = 1.0; "
       "par.MoIZ_IC = 0.05; par.OmgtauIC = 50.0")
SOL_T = "par.rtau = -0.04; par.itau = 0.13"
MAG = "par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3"
TWA = "par.B0 = 'G21 dipole'; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02"
drag_runs = []   # the runs with a drag, for 6.

# 1. against conductive_IC 2d84351 (Ek 1e-3, and Em 1e-3, Le² 1e-3, cnorm 'mag_energy' with a field): the first two
#    eigenvalues (lambda = 0 left out), and with the solids, for the first, omega_IC, phi and the viscous and magnetic torques
#    on the mantle relative to omega_M
cic = [
    ('k_so', SO + "; par.kappa_cmb = 0.01", 20,
     [-4.006463497989679e-02+1.0010892570603775j, -2.4656606140584303e-01+1.0282107282582311j], None),
    ('k_sphere', N48 + "; par.ricb = 0.0; par.m = 1; par.symm = -1; par.bco = 0; par.kappa_cmb = 0.01; par.itau = 0.97", 20,
     [-3.786892079590201e-02+1.0003542717286722j, -2.2283402967059351e-01+1.0193847915899865j], None),
    ('t_so', SO + "; " + REK + "; par.topo_N = 0.5", 20,
     [-3.817621640452094e-02+1.0009718869454056j, -2.41036936554251e-01+1.030331662880938j], None),
    ('t_m2', N48 + "; par.m = 2; par.symm = 1; par.bco = 0; par.topo_drag = 'rekier'; par.topo_hrms = 0.02; "
                   "par.topo_lambda0 = 0.3; par.topo_H = 0.9; par.topo_N = 0.0; par.topo_omega = 0.7; par.itau = 0.5", 21,
     [-3.054975993075409e-01+6.06901751527106e-01j, -2.3003167706095093e-01+7.80072693386833e-01j], None),
    ('t_sphere', N48 + "; par.ricb = 0.0; par.m = 1; par.symm = 1; par.bco = 0; " + REK + "; par.topo_N = 1.5; "
                       "par.itau = 0.6", 20,
     [-2.077483496652575e-01+5.85173034455509e-01j, -1.4244067178671813e-01+7.560518306897646e-01j], None),
    ('r_free', SOL + "; " + SOL_T, 19,
     [-3.7289948261550354e-02+1.3457785369112066e-01j, -4.14016534031745e-02+0j],
     [-16.466457427990573-7.345245328762405j, -37.289948261550684+134.57785369112062j, 0j, 0j]),
    ('r_k', SOL + "; par.kappa_cmb = 0.01; " + SOL_T, 19,
     [-3.839552806206431e-02+1.3489437357015638e-01j, -5.590908122703851e-02+0j],
     [-20.16898835315716-0.5742326076932577j, 16.830637458282844+154.6349142474966j,
      -0.055226165520349216-0.019740540677337572j, 0j]),
    ('r_t', SOL + "; " + REK + "; par.topo_N = 0.3; par.topo_omega = 0.2; " + SOL_T, 19,
     [-3.784932620063232e-02+1.3468893539517535e-01j, -5.5721937052805884e-02+0j],
     [-17.59701523573015-5.0234948474035095j, -18.671204662133007+140.54817199914874j,
      -0.019178121538498554-0.0058592366039718615j, 0j]),
    ('r_twa_k', SOL + "; par.kappa_cmb = 0.01; " + MAG + "; " + TWA + "; par.innercore = 'TWA'; par.c_icb = 0.0; "
                      "par.c1_icb = 0.05; par.itau = 0.2", 19,
     [-4.161618750252146e-02+1.3339945891139166e-01j, -9.599853653603262e-03+0j],
     [-19.726919101970953-0.8163041607199383j, 18.570249571689804+152.36569376763026j,
      -0.05628128796961209-0.020905682639148087j, -0.0039051491045993363+0.0019394477829125579j]),
    ('r_res_k', SOL + "; par.kappa_cmb = 0.01; " + MAG + "; par.B0 = 'Luo_S2'; par.B0_l = 2; "
                      "par.innercore = 'conducting, Chebys'; par.N_cic = 24; par.mantle = 'resolved'; par.r_surface = 1.3; "
                      "par.N_mantle = 24; par.sigma_mantle = 0.5; par.itau = 0.2", 19,
     [-4.475639221945612e-02+1.3860468742127746e-01j, -9.398831042855804e-02+5.21086708352016e-02j],
     [-22.765782981171263-2.158001911522717j, 14.590589495270049+168.8584495975739j,
      -0.05952980000850956-0.03125903380055932j, 0.00018281829378123896+0.0010052716242524588j]),
    ('m_twa_t', SO + "; " + REK + "; par.topo_N = 0.5; " + MAG + "; " + TWA, 20,
     [-4.179835933013642e-02+1.0038632343861809j, -2.445615665992149e-01+1.049371387875261j], None),
]
dirs = {}
for name, extra, lmax, ref, rat in cic:
    d = dirs[name] = build(name, extra, lmax)
    lam = eig(d)
    err = closest(lam, np.array(ref))
    text = 'eigenvalues %.1e' % err
    ok = err < 1e-10
    if rat is not None:
        er2 = max(abs(ratios(d, lam, ref[0]) - rat))/max(abs(np.array(rat)))
        ok &= er2 < 1e-8
        text += ', rotdyn.dat %.1e' % er2
    report(name, ok, text + ' from conductive_IC')
    if name != 'r_free':
        drag_runs.append(name)

# 2. conservation, 'Wu 2005', with the rotating solids: the total angular momentum, resid𝐋, and the energy budget
WU = "par.structure = 'Wu 2005'; par.r_out = 0.9; par.N = 64"
WU_MAG = (MAG + "; par.B0 = 'G21 dipole'; par.innercore = 'conducting, Chebys'; par.N_cic = 24; par.mantle = 'resolved'; "
          "par.r_surface = 1.3; par.N_mantle = 24; par.sigma_mantle = '0.5*exp(-p*(r - 1))'; par.sigma_mantle_param = 5.0")
for name, extra in (('wu_k', "par.kappa_cmb = 0.01; " + SOL_T),
                    ('wu_t_field', REK + "; par.topo_N = 0.3; " + WU_MAG + "; par.itau = 0.2")):
    d = dirs[name] = build(name, SOL + '; ' + WU + '; ' + extra, 19)
    drag_runs.append(name)
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

# 3. equivalences: the test hook topo_K with a constant tensor is kappa_cmb; a rigid differential rotation is a faster frame
d = dirs['k_hook'] = build('k_hook', SO + "; par.topo_drag = 'rekier'; par.topo_K = '(0.01 + 0*x, 0.01 + 0*x)'", 20)
drag_runs.append('k_hook')
err = closest(eig(d), eig(dirs['k_so'])[:2])
report('k_hook', err < 1e-12, 'eigenvalues %.1e from kappa_cmb = 0.01' % err)
d = dirs['k_rigid'] = build('k_rigid', SO + "; par.kappa_cmb = 0.01; par.diff_rot = 1; par.diff_rot_type = 'rigid'; "
                                            "par.diff_rot_ampl = 0.3", 20)
drag_runs.append('k_rigid')
lam = eig(build('k_rigid_frame', SO + "; par.Gaspard = 1.3; par.kappa_cmb = 0.01/1.3; par.itau = 1.28", 20)) - 0.3j
err = closest(eig(d), lam[:2])
report('k_rigid', err < 1e-12, 'eigenvalues %.1e from those of a frame turning 1.3 times faster' % err)

# 4. limits: a stiff drag is a no-slip wall; a weak drag dissipates kappa_cmb Dtopo/κ of the free run and, with the mantle
#    at rest, adds kappa_cmb δσtopo/κ to its damping rate
for name, a, b, lmax, z in (('k_stiff', SO + "; par.kappa_cmb = 1e8", SO + "; par.bco = 1", 20, None),
                            ('r_stiff', SOL + "; par.kappa_cmb = 1e8; " + SOL_T, SOL + "; par.bco = 1; " + SOL_T, 19, 1)):
    d1 = dirs[name] = build(name, a, lmax)
    d0 = build(name + '_noslip', b, lmax)
    l0, l1 = eig(d0), eig(d1)
    l0 = l0[abs(l0) > 1e-8]
    err = closest(l1, l0[:3])
    text = 'eigenvalues %.1e' % err
    ok = err < 1e-8
    if z is not None:
        r0 = ratios(d0, eig(d0), l0[0])
        er2 = max(abs(ratios(d1, l1, l0[0]) - r0))/max(abs(r0))
        ok &= er2 < 1e-8
        text += ', rotdyn.dat %.1e' % er2
    report(name, ok, text + ' from a no-slip wall')

dirs['k_free'] = build('k_free', SO, 20)
for name, free, extra, lmax in (('k_weak', 'k_free', SO + "; par.kappa_cmb = 1e-6", 20),
                                ('r_weak', 'r_free', SOL + "; par.kappa_cmb = 1e-6; " + SOL_T, 19)):
    d = dirs[name] = build(name, extra, lmax)
    drag_runs.append(name)
    l0, f0 = eig(dirs[free]), table(dirs[free], 'flow.dat')
    l1, f1 = eig(d), table(d, 'flow.dat')
    i0 = 0
    i1 = int(np.argmin(abs(l1 - l0[i0])))
    eD = abs(f1[i1, 39] - 1e-6*f0[i0, 39])/abs(1e-6*f0[i0, 39])
    es = abs((l1[i1].real - l0[i0].real) - 1e-6*f0[i0, 40])/abs(1e-6*f0[i0, 40])
    if name == 'k_weak':
        report(name, eD < 1e-3 and es < 5e-3, 'Dtopo %.1e, the change of the damping rate %.1e from kappa_cmb times the '
                                              'stress-free run\'s' % (eD, es))
    else:
        ratio = (l1[i1].real - l0[i0].real)/(1e-6*f0[i0, 40])
        report(name, eD < 1e-3, 'Dtopo %.1e from kappa_cmb times the stress-free run\'s; the change of the damping rate is '
                                '%.2f times kappa_cmb δσtopo/κ' % (eD, ratio))

# 5. the frequencies of the topography's drag (params.dat columns 140, 141), a longer time unit, and a forced run
TH = (SO + "; " + REK + "; par.thermal = 1; par.stratification = 'linear'; par.bci_thermal = 0; par.bco_thermal = 0")
dirs['th'] = build('th', TH, 20)
dirs['th2'] = build('th2', TH + "; par.Gaspard = 2.0; par.Beyonce = 40.0; par.ViscosD = 2e-3; par.ThermaD = 2e-3; "
                         "par.itau = 1.96", 20)
drag_runs += ['th', 'th2']
dirs['forced'] = build('forced', N48 + "; par.m = 1; par.symm = -1; par.bco = 0; " + REK + "; par.topo_N = 0.5; "
                                       "par.forcing = 6; par.forcing_frequency = 0.9; par.forcing_amplitude_icb = 1.0", 20)
# NEW (2026-11-09): forcing 9 with the outer wall in place and forcing 13 at frequency 0 between stress-free walls, with
# kappa_cmb (they stopped before: the drag acts on the outer wall's slip, and on the rigid rotation that would otherwise
# also solve the steady problem of forcing 13)
dirs['forced9'] = build('forced9', N48 + "; par.m = 2; par.symm = 1; par.bco = 0; par.kappa_cmb = 0.05; par.forcing = 9; "
                                         "par.forcing_frequency = 0.9; par.forcing_amplitude_cmb = 0.0; "
                                         "par.forcing_amplitude_icb = 1.0", 21)
dirs['forced13'] = build('forced13', N48 + "; par.m = 0; par.symm = 1; par.bci = 0; par.bco = 0; par.kappa_cmb = 0.05; "
                                           "par.thermal = 1; par.stratification = 'linear'; par.forcing = 13; "
                                           "par.forcing_frequency = 0.0; par.forcing_amplitude_icb = 1.0", 19)
want = {'t_so': (0.5, 0.98), 't_m2': (0.0, 0.7), 'r_t': (0.3, 0.2), 'th': (np.sqrt(10), 0.98), 'th2': (np.sqrt(10), 0.98),
        'forced': (0.5, 0.9)}
bad = []
for name, (n, w) in want.items():
    p = table(dirs[name], 'params.dat')[-1]
    if len(p) < 141 or abs(p[139] - n) > 1e-9*max(1, n) or abs(p[140] - w) > 1e-9:   # (later columns appended after 141)
        bad.append('%s (%s)' % (name, ', '.join('%.9g' % v for v in p[139:141])))
err = closest(eig(dirs['th2']), 2*eig(dirs['th'])[:2])
ef = []
for name in ('forced', 'forced9', 'forced13'):
    f = table(dirs[name], 'flow.dat')
    ef.append(abs(f[0, 39] + f[0, 24])/max(abs(f[0, 39]), abs(f[0, 24])))
report('frequencies', not bad and err < 1e-10 and all(e < 1e-8 for e in ef),
       'N/Omega and w0/Omega of %d runs%s; time unit x2: eigenvalues %.1e from twice; forced runs (6, 9, 13): Dtopo = '
       '-Wv_cmb to %s' % (len(want) - len(bad), '; not: ' + ', '.join(bad) if bad else '', err,
                          ', '.join('%.1e' % e for e in ef)))

# 6. the power of the drag (the stiff runs left out: their slip is round-off)
res = {name: wall_power(dirs[name]) for name in drag_runs}
worst = max(res, key=res.get)
report('wall power', all(v < 1e-8 for v in res.values()), 'Dtopo = -Wv_cmb - 2 Re(omega_M* G_M) to %.1e (%s) in %d runs' %
       (res[worst], worst, len(res)))

# 7. the stops of background.check_drag
def stops(name, extra, words):
    '''whether submatrices.py stops (background.check) with a message that contains words'''
    d = os.path.join(W, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    base = "par.N = 24; par.m = 1; par.symm = -1; par.bco = 0; "
    open(d + '/bin/parameters.py', 'w').write(mk_run.T.format(ncpus=1, extra=(base + extra).replace('; ', '\n')))
    p = subprocess.run([sys.executable, 'bin/submatrices.py', '1'], cwd=d, capture_output=True, text=True)
    return p.returncode != 0 and words in p.stdout + p.stderr

KR = "par.topo_drag = 'rekier'; par.topo_hrms = 0.01"
cases = [('topo_drag', "par.topo_drag = 'Rekier'", "None (no topographic drag) or 'rekier'"),
         ('kappa_cmb < 0', "par.kappa_cmb = -1.0", 'the drag coefficient is a finite number >= 0'),
         ('both drags', KR + "; par.kappa_cmb = 0.01", 'topo_drag and kappa_cmb cannot be combined'),
         ('no-slip wall', "par.kappa_cmb = 0.01; par.bco = 1", 'and a stress-free outer wall (bco = 0)'),
         ('inviscid', "par.kappa_cmb = 0.01; par.ViscosD = 0.0; par.ricb = 0", 'a viscous fluid (ViscosD > 0)'),
         ('forcing 9', "par.kappa_cmb = 0.01; par.m = 2; par.symm = 1; par.forcing = 9", 'with forcing 9 is not covered'),
         ('diff_rot', KR + "; par.diff_rot = 1", 'with a differential rotation (diff_rot = 1) is not covered'),
         ('topo_hrms = 0', "par.topo_drag = 'rekier'", 'needs topo_hrms > 0, topo_lambda0 > 0 and topo_H > 1/2'),
         ('topo_H = 0.5', KR + "; par.topo_H = 0.5", 'needs topo_hrms > 0, topo_lambda0 > 0 and topo_H > 1/2'),
         ('w0 = 0', KR + "; par.itau = 0.0", 'needs a frequency of the flow other than 0'),
         ('scan', KR + "; par.forcing = 11; par.forcing_frequency = 0.5; par.rsf_frequencies = [0.4, 0.5]",
          'with a frequency scan (rsf_frequencies) needs topo_omega'),
         ('topo_N < 0', KR + "; par.topo_N = -1.0", 'None (from the run) or a number >= 0'),
         ('topo_K < 0', KR + "; par.topo_K = '(x, 1 + 0*x)'", 'kappa_tt and kappa_pp must be finite and >= 0'),
         ('topo_K error', KR + "; par.topo_K = 'y'", 'not a pair of numpy expressions of x')]
bad = [name for k, (name, extra, words) in enumerate(cases) if not stops('stop_%d' % k, extra, words)]
report('stops', not bad, '%d of %d stop with their message%s' % (len(cases) - len(bad), len(cases),
                                                                   '; not: ' + ', '.join(bad) if bad else ''))

print('drag_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
