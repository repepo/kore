#!/usr/bin/env python3
'''
usage: forced_check.py [WORKDIR]   (default: tests/forced_field/runs)

Forced problems with a field (magnetic = 1, forcing != 0, NEW 2026-11-06 from conductive_IC): the forced rows of the field
(A - i w B in the sections f, g, fic, gic, fm, gm: assemble.py), the inner core 'conducting, Bessel' (assemble.bc_b_bessel,
utils.bessel_ic, utils.xdlogjl, utils4pp.bessel_budget), the conducting solids under a radial boundary flow (forced_walls:
'rigid', the solids at rest, assemble.bc_b_radial; 'deforming', the solids deform with the boundary, the rows without u×B0,
and the motion inside the resolved mantle and the Chebyshev core, assemble.deform_rows, utils4pp.deform_work) and resid𝐄
where the walls have a boundary flow (utils4pp.wall_powers, lorentz_walls, magnetic_powers_grid, twa_budget). Exit code 1
if a check fails.

1. Against conductive_IC 2d84351: 8 forced runs ('Boussinesq', N 48, lmax 19 to 21, 2 ranks): libration (forcing 7, m 0)
   with insulating, conducting (Chebyshev and Bessel) and 'TWA' inner cores, 'TWA' and resolved mantles, B0 'axial' and
   'Luo_S2'; forcing 9 with insulating walls, and with 'TWA' walls and forced_walls = 'deforming' (conductive_IC's rows).
   The kinetic energy, the magnetic energy in the fluid and in the inner core of conductive_IC's solutions (this
   spin_doctor on its field files) to 1e-8, the 10 digits of flow.dat and magnetic.dat; resid𝐄 to 1e-8.
2. The Bessel inner core against the Chebyshev one (N_cic 32, 40 with the rotation) under a radial flow through the inner wall (forced_walls
   'rigid', the default of forcing 6): forcing 9 with the core turning with the differential rotation
   ('shellular_boussinesq', ΔΩ(ricb) = c != 0: the frequency in its frame is w + m c), with sigma_i2o 0.3, and with
   mu_i2o 1.5 where the forcing moves the outer wall only; and forcing 6. The solutions to 1e-10, the energy and the Ohmic loss of the core (Bessel: closed forms) to
   1e-8, resid𝐄 to 1e-9.
3. The forced matrix is A - i w B in every row: A(w1) - A(w2) = -i (w1 - w2) B, with B of the eigenvalue problem, and A(0)
   is the eigenvalue problem's A before its normalisation (1e-13); with a conducting inner core and the resolved mantle, and
   with a 'TWA' outer wall and mu_i2o != 1. A frequency scan of forcing 11 with a conducting inner core gives the same
   spectrum whatever the frequency A was assembled at (1e-10).
4. resid𝐄 closes (1e-9) where the walls have a boundary flow. With forced_walls 'rigid': a variable density ('Wu 2005',
   r_out 0.9, N 96) with forcing 9 and the Bessel core, and forcing 14 with a 'TWA' mantle; a toroidal B0 ('Malkus', not
   zero at the walls)
   with forcing 9 and 'TWA' walls, and with the resolved mantle; forcing 10 with a 'TWA' mantle; a full sphere (N 96) with
   forcing 9 and a 'TWA' mantle; forcing 9 under the resolved mantle with the Chebyshev core, also with a stress-free outer
   wall; forcing 7 with m 2 under the resolved mantle with the Bessel core; forcing 6 through a stress-free 'TWA' inner
   wall, and through a no-slip one while a permeable outer layer that does not conduct (muw_cmb 2, c_cmb 1e-3, c1_cmb 0:
   its budget is exact) stays in place; free walls (bco = 2, bci = 2) in eigenvalue problems. Where no wall moves radially ('Wu 2005'): forcing 13
   (entropy at the inner wall) with the Chebyshev core, and libration under the resolved mantle. With forced_walls
   'deforming' (the work of the motion inside
   the solids, utils4pp.deform_work): forcing 14 with 'TWA' walls; forcings 9 and 14 with the resolved mantle and the
   Chebyshev core at a variable density; B0 'G21 dipole', also with a stress-free outer wall, a conductivity that varies
   in the mantle and in a full sphere; 'Malkus' with the solids and with 'TWA' walls (the layers move across its drop);
   'Luo_S2'; a 'general' B0 of poloidal degrees 1, 3 and toroidal 2, 4, with the solids and with 'TWA' walls; forcing 6
   with the Chebyshev core, forcing 10 with the mantle, forcing 7 with m 2 with both; the differential rotation.
5. A resolved mantle of thickness h and conductivity c1/h tends to the 'TWA' mantle of conductance c1 (c_cmb = 0) as h -> 0
   under forcing 9 at the outer wall, with B0 'axial' and 'Malkus', for forced_walls 'rigid' and 'deforming' (the layer's
   motion across the drop of 'Malkus' to 0 gives the 'TWA' rows half the toroidal u×B0, background.thin_drop): the field
   differs by O(h) (the difference halves from h = 0.01 to 0.005, 1.8 to 2.2 times smaller, and is below 3e-2 at h = 0.01).
6. The angular algebra of the radial flow's u×B0 (assemble.bc_b_radial: S_l, T_l of uᵣ r̂×B0 for uᵣ = Y_k) against its
   projection on a Gauss grid, at both walls, for a poloidal (degrees 1, 3) and a toroidal (2, 4) B0, and half the
   toroidal part alone (tor_half): 1e-13.
7. Kinematic induction in a forced run: just_induction.py and solve_ind.py give the coupled run's field from its flow
   (1e-10), with the Bessel core under forcing 9, with the Chebyshev core under the resolved mantle, and with both deforming.
8. The stops: forced_walls = 'deforming' (also by default, forcing 9) with the Bessel core behind an inner wall with a
   radial flow, and by default under forcing 14 (its default 'deforming'); a permeability that jumps at a wall with a
   radial flow where B0 reaches it, in both pictures: mu_i2o != 1 with the Chebyshev core ('deforming' and 'rigid') and
   the Bessel core (forcing 6), muw_cmb != 1 of a conducting sheet (c_cmb = 0, c1_cmb != 0) with the toroidal 'Malkus'
   ('deforming' and 'rigid') and at rest ('rigid') with B0 'axial', and of a layer with a thickness that does not conduct
   with 'axial', mu != 1, mu = muw != 1, and muw_icb != 1 at an inner wall that moves alone; a forced_walls other than
   None, 'rigid' and 'deforming', also in an eigenvalue problem. But a permeability other than 1 runs at walls that stay
   in place (muw_cmb under forcing 6, mu_i2o where only the outer wall moves), as muw_cmb of an insulating layer
   (c_cmb = c1_cmb = 0, where it enters nothing), as muw of conducting sheets that deform with the boundary under 'axial'
   (both walls) and of a layer with a thickness that does not conduct under 'Malkus' alone (the parts of B0 that muw acts
   on are not at the wall), and where B0 does not reach the walls (B0_tor 'wall', 0 at both; a 'general' B0 with h = h' = 0 at the outer wall, whose
   interpolated slope there is not exactly 0); 'deforming' runs
   with the Chebyshev core and the resolved mantle, and with the Bessel core where the flow is at the outer wall only;
   forcing 6 with the Bessel core ('rigid' by default). The Bessel core in an eigenvalue problem and in a frequency scan;
   just_induction.py with a lambda other than i forcing_frequency in a run with the Bessel core, and in a run of forcing 9,
   which writes no B.npz. Forcing 10 defaults to 'deforming': its matrix with a thin outer layer is that of 'deforming'
   exactly, and not that of 'rigid'.
9. utils.xdlogjl, x j_l'(x)/j_l(x), against mpmath (50 digits) for l 0 to 300 and |x| from 1e-8 to 1.4e4 on the lines
   arg x = -π/4 (k ricb for w > 0) and π/4 (w < 0): 1e-12. Skipped without mpmath.
'''
import os, sys, shutil, subprocess
import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T, '..', 'tidal'))
import mk_run   # noqa: E402  (the run builder of tests/tidal: 'Boussinesq', ricb 0.35, NCPUS ranks)
sys.path.append(os.path.join(T, '..', '..', 'bin'))
import field_io   # noqa: E402  (the solution files, in either format)

# FIX (2026-11-08): absolute, since check 9 imports utils from a run directory after a chdir (a relative WORKDIR failed)
W = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(T, 'runs')
os.makedirs(W, exist_ok=True)
LMAX_LINE = 'par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)'
assert LMAX_LINE in mk_run.T
py = sys.executable


def setup(d, extra, lmax=None):
    '''the run directory d (bin and parameters.py, as mk_run.build); with lmax given, that lmax on 2 ranks (1 with NCPUS = 1)'''
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(mk_run.BIN):
        if os.path.isfile(mk_run.BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(mk_run.BIN + '/' + f, d + '/bin/' + f)
    t, n = mk_run.T, mk_run.NCPUS
    if lmax is not None:
        t, n = t.replace(LMAX_LINE, 'par.lmax = %d' % lmax), min(n, 2)
    open(d + '/bin/parameters.py', 'w').write(t.format(ncpus=n, extra=extra.replace('; ', '\n')))
    return str(n)

def run(d, n, steps='savd', check=True):
    '''the steps of a run in d: s(ubmatrices), a(ssemble), v (solve), d (spin_doctor); (returncode, output) of the last'''
    cmds = {'s': [py, 'bin/submatrices.py', n], 'a': mk_run.MPIEXEC.split() + ['-n', n, py, 'bin/assemble.py'],
            'v': mk_run.MPIEXEC.split() + ['-n', n, py, 'bin/solve.py'], 'd': [py, 'bin/spin_doctor.py', n]}
    for st in steps:
        p = subprocess.run(cmds[st], cwd=d, capture_output=True, text=True)
        out = p.stdout + p.stderr
        open(os.path.join(d, 'log_' + st), 'w').write(out)
        if p.returncode:
            if check:
                sys.exit('%s: step %s failed\n%s' % (d, st, out[-3000:]))
            return p.returncode, out
    return 0, out

def build(name, extra, lmax=None, steps='savd'):
    d = os.path.join(W, name)
    run(d, setup(d, extra, lmax), steps)
    return d

def field(d, name):   # the solutions of field name, from name.npy or the text files
    return field_io.load(name, folder=d)

def dat(d, f):
    return np.loadtxt(os.path.join(d, f), ndmin=2)

def load_csr(f):
    import scipy.sparse as ss
    z = np.load(f)
    return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))

def rel(a, b):
    return abs(a - b).max()/abs(b).max()

fail = 0
def report(name, ok, text):
    global fail
    fail |= not ok
    print('%-18s %s  (%s)' % (name, 'ok' if ok else 'FAIL', text))


MAG = "par.magnetic = 1; par.cnorm = 'mag_energy'; par.Hendrik = 1e-3; par.MagnetD = 1e-3; par.N = 48"
LIB = "par.m = 0; par.symm = 1; par.forcing = 7; par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 1.0"
F9 = ("par.m = 2; par.symm = 1; par.forcing = 9; par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 0.5; "
      "par.forcing_amplitude_icb = 1.0")
CHEB, BES = "par.innercore = 'conducting, Chebys'; par.N_cic = 32", "par.innercore = 'conducting, Bessel'"
TWA2 = "par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; par.innercore = 'TWA'; par.c_icb = 0.0; par.c1_icb = 0.05"
MAN = "par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 24; par.sigma_mantle = 0.5"
WU = ("par.thermal = 1; par.structure = 'Wu 2005'; par.stratification = 'top hat'; par.r_out = 0.9; par.Beyonce = 10.0; "
      "par.ThermaD = 1e-3; par.rpower_u = 5; par.rhopower_u = 1; par.rpower_v = 3; par.rhopower_v = 1; par.rpower_pp = 4; "
      "par.rhopower_pp = 4; par.N = 96")

# 1. against conductive_IC 2d84351: (changes, lmax, (KE, ME, MEic) of its solution). Its conducting inner core under a
# radial flow holds no u×B0 in its rows but no motion inside: neither forced_walls, so it is not compared
CIC = {
    'lib_ins':     ("par.B0 = 'axial'; " + LIB, 19, (1.089711624e-02, 1.810042996e-01, 2.571157841e-05)),
    'lib_cheb':    ("par.B0 = 'axial'; " + LIB + "; par.forcing_amplitude_icb = 0.5; " + CHEB, 19,
                    (1.097006396e-02, 1.835852006e-01, 8.682695960e-04)),
    'lib_bes':     ("par.B0 = 'axial'; " + LIB + "; par.forcing_amplitude_icb = 0.5; " + BES, 19,
                    (1.097006396e-02, 1.835852006e-01, 8.682695960e-04)),
    'lib_twa':     ("par.B0 = 'axial'; " + LIB + "; par.forcing_amplitude_icb = 0.5; " + TWA2, 19,
                    (1.114227460e-02, 3.776820686e-01, 2.770360352e-05)),
    'lib_man':     ("par.B0 = 'axial'; " + LIB + "; par.forcing_amplitude_icb = 0.5; " + MAN + "; " + CHEB, 19,
                    (1.112533970e-02, 4.082548424e-01, 9.133064957e-04)),
    'lib_s2_bes':  ("par.B0 = 'Luo_S2'; par.B0_l = 2; " + LIB + "; par.forcing_amplitude_icb = 0.5; " + BES, 19,
                    (1.088296592e-02, 5.776216611e-02, 3.183301808e-03)),
    'f9_ins':      ("par.B0 = 'axial'; " + F9, 21, (4.955221341e-01, 3.128535070e+00, 3.634769267e-01)),
    'f9_twa':      ("par.B0 = 'axial'; " + F9 + "; " + TWA2 + "; par.forced_walls = 'deforming'", 21,
                    (4.955276085e-01, 4.854061308e+00, 5.759154433e-01)),
}
runs = {}
for name, (extra, lmax, ref) in CIC.items():
    d = runs[name] = build(name, MAG + '; ' + extra, lmax)
    f, m = dat(d, 'flow.dat')[0], dat(d, 'magnetic.dat')[0]
    got = (f[0], m[0], m[6])
    err = max(abs(g/r - 1) for g, r in zip(got, ref))
    report(name, err < 1e-8 and m[10] < 1e-8, 'KE, ME, MEic %.1e from conductive_IC; resid𝐄 %.1e' % (err, m[10]))

# 2. the Bessel core against the Chebyshev one under a radial flow through the inner wall (forced_walls 'rigid'): forcing 9,
# also with the core turning with ΔΩ(ricb) and with sigma_i2o 0.3, and forcing 6; and a core of permeability 1.5 where
# forcing 9 moves the outer wall only
DR = ("par.B0 = 'axial'; " + F9 + "; par.diff_rot = 1; par.diff_rot_type = 'shellular_boussinesq'; par.diff_rot_ampl = 0.4; "
      "par.N = 64")
F6 = ("par.m = 1; par.symm = -1; par.forcing = 6; par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 0.0; "
      "par.forcing_amplitude_icb = 1.0")
RIG = "par.forced_walls = 'rigid'"   # (forcing 9 defaults to 'deforming', which the Bessel core does not take)
PAIRS = {'f9': ("par.B0 = 'axial'; " + F9 + "; " + RIG, 21, CHEB),
         'f6': ("par.B0 = 'axial'; " + F6, 20, CHEB),   # forcing 6 defaults to 'rigid'
         'dr': (DR + "; " + RIG, 21, "par.innercore = 'conducting, Chebys'; par.N_cic = 40"),
         'f9s': ("par.B0 = 'axial'; " + F9 + "; " + RIG + "; par.sigma_i2o = 0.3", 21, CHEB),
         # (a core of another permeability needs an inner wall that stays in place: background.check_forced_magnetic)
         'f9m': ("par.B0 = 'axial'; " + F9 + "; par.forcing_amplitude_icb = 0.0; " + RIG + "; par.sigma_i2o = 0.3; "
                 "par.mu_i2o = 1.5", 21, CHEB)}
for tag, (extra, lmax, cheb) in PAIRS.items():
    db = runs[tag + '_bes'] = build(tag + '_bes', MAG + '; ' + extra + '; ' + BES, lmax)
    dc = runs[tag + '_cheb'] = build(tag + '_cheb', MAG + '; ' + extra + '; ' + cheb, lmax)
    eu, eb = rel(field(db, 'flow'), field(dc, 'flow')), rel(field(db, 'magnetic'), field(dc, 'magnetic'))
    mb, mc = dat(db, 'magnetic.dat')[0], dat(dc, 'magnetic.dat')[0]
    ec = max(abs(mb[6]/mc[6] - 1), abs(mb[9]/mc[9] - 1))
    report('bessel_chebys_' + tag, max(eu, eb) < 1e-10 and ec < 1e-8 and max(mb[10], mc[10]) < 1e-9,
           'flow %.1e, field %.1e; energy and Ohmic loss of the core %.1e; resid𝐄 %.1e, %.1e' % (eu, eb, ec, mb[10], mc[10]))

# 3. the forced matrix is A - i w B in every row; the frequency scan of forcing 11
for name, extra in (('ident_man', "par.B0 = 'axial'; par.m = 0; par.symm = 1; " + MAN + "; " + CHEB),
                    ('ident_twa', "par.B0 = 'G21 dipole'; par.m = 0; par.symm = 1; par.mantle = 'TWA'; par.c_cmb = 0.0; "
                                  "par.c1_cmb = 0.02; " + CHEB + "; par.mu_i2o = 2.0; par.sigma_i2o = 0.5")):
    A = {}
    for w in (0.37, 0.11, 0.0):
        d = build('%s_%g' % (name, w), MAG + '; ' + extra + "; par.forcing = 7; par.forcing_frequency = %r; "
                  "par.forcing_amplitude_cmb = 1.0; par.forcing_amplitude_icb = 0.5" % w, 19, 'sa')
        A[w] = load_csr(d + '/A.npz')
    d = build(name + '_eig', MAG + '; ' + extra, 19, 'sa')
    import scipy.sparse.linalg as ssl
    nrm = ssl.norm(A[0.0])
    B = load_csr(d + '/B.npz')*nrm
    e0 = abs(A[0.0]/nrm - load_csr(d + '/A.npz')).max()
    e1 = abs(A[0.37] - A[0.11] + 1j*(0.37 - 0.11)*B).max()/abs((0.37 - 0.11)*B).max()
    report(name, e0 < 1e-13 and e1 < 1e-13, 'A(w1) - A(w2) + i (w1 - w2) B: %.1e; A(0) - A of the eigenproblem: %.1e' % (e1, e0))
RSF = ("par.forcing = 11; par.rsf_theta = 1.0; par.rsf_nu_floor = 0; par.rsf_lc = 0.05; par.rsf_taper = 0.1; "
       "par.rsf_nsamp = 2; par.B0 = 'axial'; par.m = 1; par.symm = 1; par.rsf_frequencies = [0.3, 0.5]; " + CHEB)
ka = dat(build('scan_a', MAG + '; ' + RSF + '; par.forcing_frequency = 0.5', 20, 'sav'), 'rsf_ke.dat')
kb = dat(build('scan_b', MAG + '; ' + RSF + '; par.forcing_frequency = 0.3', 20, 'sav'), 'rsf_ke.dat')
report('scan', ka.shape == kb.shape and rel(ka, kb) < 1e-10, 'the scans assembled at w 0.5 and 0.3: %.1e' % rel(ka, kb))

# 4. resid𝐄 where the walls have a boundary flow
F7M2 = ("par.m = 2; par.symm = 1; par.forcing = 7; par.forcing_frequency = 0.37; par.forcing_amplitude_cmb = 1.0; "
        "par.forcing_amplitude_icb = 0.5")
F9O = F9 + "; par.forcing_amplitude_icb = 0.0"
F14M1 = ("par.m = 1; par.symm = -1; par.forcing = 14; par.forcing_frequency = -0.6; par.forcing_amplitude_cmb = 1.0; "
         "par.forcing_amplitude_icb = 0.3")
MALKUS = "par.B0 = None; par.B0_tor = 'Malkus'; par.B0_tor_l = 1"
GEN = ("par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2) + 0.1'}; par.B0_tor = 'general'; "
       "par.B0_tor_profiles = {2: 'r**2*(2 - r)', 4: '0.2*r**4'}")
DEF = "par.forced_walls = 'deforming'"
BUD = {
    # forced_walls 'rigid': the solids at rest, the boundary flow passes through the walls
    'wu_f9_bes':    WU + "; par.B0 = 'axial'; " + F9 + "; " + BES + "; " + RIG,
    'wu_f14_twa':   (WU + "; par.B0 = 'axial'; " + F14M1 + "; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; " + CHEB
                     + "; " + RIG),
    # (wu_f13_cheb, wu_lib_man: no wall moves radially; forcing 13 sets the entropy, libration moves the walls along)
    'wu_f13_cheb':  (WU + "; par.B0 = 'G21 dipole'; par.m = 0; par.symm = 1; par.forcing = 13; par.forcing_frequency = 0.37; "
                     "par.forcing_amplitude_icb = 1.0; " + CHEB),
    'wu_lib_man':   WU + "; par.B0 = 'axial'; " + LIB + "; par.forcing_amplitude_icb = 0.5; " + MAN + "; " + BES,
    'malkus_twa':   MALKUS + "; " + F9 + "; " + TWA2 + "; " + RIG,
    'malkus_man':   MALKUS + "; " + F9O + "; " + MAN + "; " + RIG,
    'f10_twa':      ("par.B0 = 'axial'; par.m = 2; par.symm = 1; par.forcing = 10; par.forcing_frequency = 0.37; "
                     "par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; " + CHEB + "; " + RIG),
    'fs_f9_twa':    ("par.ricb = 0.0; par.N = 96; par.B0 = 'G21 dipole'; " + F9 + "; par.forcing_amplitude_icb = 0.0; "
                     "par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; " + RIG),
    'f9_man_cheb':  "par.B0 = 'G21 dipole'; " + F9 + "; " + MAN + "; " + CHEB + "; " + RIG,
    'slip_f9_man':  "par.B0 = 'G21 dipole'; " + F9 + "; par.bco = 0; " + MAN + "; " + CHEB + "; " + RIG,
    'f7m2_man':     "par.B0 = 'axial'; " + F7M2 + "; " + MAN + "; " + BES,   # forcing 7 defaults to 'rigid'
    'slip_f6_twa':  "par.B0 = 'axial'; " + F6 + "; par.bci = 0; " + TWA2,
    'f6_muw_fixed': ("par.B0 = 'axial'; " + F6 + "; " + TWA2 + "; par.c_cmb = 1e-3; par.c1_cmb = 0.0; "
                     "par.muw_cmb = 2.0"),   # (a permeable outer layer, which stays in place)
    'free_outer':   "par.B0 = 'axial'; par.m = 1; par.symm = 1; par.bco = 2; par.nev = 3; par.itau = 0.5",
    'free_inner':   "par.B0 = 'G21 dipole'; par.m = 1; par.symm = 1; par.bci = 2; par.nev = 3; par.itau = 0.5",
    # forced_walls 'deforming' (the default of forcings 9, 10 and 14): the solids deform with the boundary, the resolved
    # mantle and the Chebyshev core with their motion inside (assemble.deform_rows, utils4pp.deform_work)
    'f14_twa_def':  ("par.B0 = 'G21 dipole'; par.m = 2; par.symm = 1; par.forcing = 14; par.forcing_frequency = 0.37; "
                     "par.forcing_amplitude_cmb = 1.0; par.forcing_amplitude_icb = 0.3; " + TWA2 + "; " + DEF),
    'def_wu_f9':    WU + "; par.B0 = 'axial'; " + F9 + "; " + MAN + "; " + CHEB,
    'def_wu_f14':   WU + "; par.B0 = 'G21 dipole'; " + F14M1 + "; " + MAN + "; " + CHEB,
    'def_g21':      "par.B0 = 'G21 dipole'; " + F9 + "; " + MAN + "; " + CHEB,
    'def_g21_sig':  "par.B0 = 'G21 dipole'; " + F9 + "; " + MAN + "; " + CHEB + "; par.sigma_i2o = 0.3",
    'def_slip':     "par.B0 = 'G21 dipole'; " + F9 + "; par.bco = 0; " + MAN + "; " + CHEB,
    'def_sigma':    ("par.B0 = 'G21 dipole'; " + F9O + "; par.mantle = 'resolved'; par.r_surface = 1.3; par.N_mantle = 24; "
                     "par.sigma_mantle = '0.5*exp(-p*(r - 1))'; par.sigma_mantle_param = 5.0"),
    'def_fs_man':   "par.ricb = 0.0; par.N = 96; par.B0 = 'G21 dipole'; " + F9O + "; " + MAN,
    'def_malkus':   MALKUS + "; " + F9 + "; " + MAN + "; " + CHEB,
    'def_malkus_twa': MALKUS + "; " + F9 + "; " + TWA2,
    'def_s2':       "par.B0 = 'Luo_S2'; par.B0_l = 2; " + F9 + "; " + MAN + "; " + CHEB,
    'def_general':  GEN + "; " + F9 + "; " + MAN + "; " + CHEB,
    'def_gen_twa':  GEN + "; " + F9 + "; " + TWA2,
    'def_f6':       "par.B0 = 'axial'; " + F6 + "; " + CHEB + "; " + DEF,
    'def_f10':      "par.B0 = 'axial'; par.m = 2; par.symm = 1; par.forcing = 10; par.forcing_frequency = 0.37; " + MAN,
    'def_f7m2':     "par.B0 = 'axial'; " + F7M2 + "; " + MAN + "; " + CHEB + "; " + DEF,
    'def_dr':       DR + "; " + MAN + "; par.innercore = 'conducting, Chebys'; par.N_cic = 40",
}
LM = {'wu_f9_bes': 21, 'wu_f14_twa': 20, 'wu_f13_cheb': 19, 'wu_lib_man': 19, 'malkus_twa': 21, 'malkus_man': 21, 'f10_twa': 21,
      'fs_f9_twa': 21, 'f9_man_cheb': 21, 'slip_f9_man': 21, 'f7m2_man': 21, 'slip_f6_twa': 20, 'f6_muw_fixed': 20,
      'f14_twa_def': 21,
      'free_outer': 20, 'free_inner': 20, 'def_wu_f9': 21, 'def_wu_f14': 20, 'def_g21': 21, 'def_g21_sig': 21, 'def_slip': 21, 'def_sigma': 21,
      'def_fs_man': 21, 'def_malkus': 21, 'def_malkus_twa': 21, 'def_s2': 21, 'def_general': 21, 'def_gen_twa': 21,
      'def_f6': 20, 'def_f10': 21, 'def_f7m2': 21, 'def_dr': 21}
for name, extra in BUD.items():
    d = runs[name] = build(name, MAG + '; ' + extra, LM[name])
    m = dat(d, 'magnetic.dat')
    report(name, np.all(m[:, 10] < 1e-9), 'resid𝐄 %.1e, resid𝐛 %.1e' % (m[:, 10].max(), m[:, 3].max()))

# 5. a thin resolved mantle tends to the 'TWA' mantle of the same conductance (c_cmb = 0), with the solids at rest
# ('rigid') and deforming with the boundary ('deforming': with the toroidal B0 'Malkus' the layer moves across its drop)
C1 = 0.02
for fw in ('rigid', 'deforming'):
    for tag, b0 in (('axial', "par.B0 = 'axial'"), ('malkus', MALKUS)):
        base = MAG + '; ' + b0 + '; ' + F9O + "; par.forced_walls = '%s'" % fw
        dt = build('twa_%s_%s' % (tag, fw), base + "; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = %r" % C1, 21)
        err = []
        for h in (0.01, 0.005):
            dl = build('layer_%s_%s_%g' % (tag, fw, h), base + "; par.mantle = 'resolved'; par.r_surface = %r; "
                       "par.N_mantle = 12; par.sigma_mantle = %r" % (1 + h, C1/h), 21)
            err.append(rel(field(dl, 'magnetic'), field(dt, 'magnetic')))
        q = err[0]/err[1]
        report('layer_%s_%s' % (fw[:3], tag), 1.8 < q < 2.2 and err[0] < 3e-2,
               'field of the layer against the TWA %.1e at h = 0.01, %.1e at 0.005 (%.2f times smaller)' % (err[0], err[1], q))

# 6. the angular algebra of bc_b_radial against a projection on a Gauss grid (in a run directory, without a solve)
ALG = ("par.N = 24; par.B0 = 'general'; par.B0_profiles = {1: 'r/6 - r**3/10', 3: '0.3*r**3*(1 - r**2) + 0.1'}; "
       "par.B0_tor = 'general'; par.B0_tor_profiles = {2: 'r**2*(2 - r)', 4: '0.2*r**4'}; " + F9)
ALG_PY = """
import sys; sys.path.insert(0, 'bin')
import numpy as np
from parameters import par
import utils as ut
import assemble as asm
import angular
import background_field as bf
worst, big = 0.0, 0.0
for wall, half in (('outer', False), ('inner', False), ('outer', True), ('inner', True)):
    asm.bc_b_radial(par.m, 'g', wall, 0, 1.0, tor_half=half)   # (tor_half: half the terms of the toroidal B0 alone)
    Ms, Mt, nB = asm._radial_mats[(wall, 'tor_half') if half else wall]
    r = ut.rcmb if wall == 'outer' else par.ricb
    x, w = np.polynomial.legendre.leggauss(2*par.lmax + nB + 10)
    Y, dY, Z = angular.ylm_grid(par.m, par.lmax + 1, x)
    Y0, dY0, _ = angular.ylm_grid(0, max(nB, 4), x)
    rr = np.array([r])
    Bth = 0*x if half else sum((f(rr, 1)[0] + f(rr, 0)[0]/r)*dY0[n] for n, f in bf.pol_profiles().items())
    Bph = (0.5 if half else 1.0)*sum(-f(rr, 0)[0]*dY0[n] for n, f in bf.tor_profiles().items())
    for k in range(par.m, par.lmax + 1):
        Vt, Vp = -Y[k]*Bph, Y[k]*Bth             # ur r x B0_t for ur = Y_k
        for l in range(par.m, par.lmax + 1):
            nrm = np.sum(w*(dY[l]**2 + np.abs(Z[l])**2))
            s = np.sum(w*(Vt*dY[l] + Vp*np.conj(Z[l])))/nrm
            t = np.sum(w*(Vt*np.conj(Z[l]) - Vp*dY[l]))/nrm
            worst, big = max(worst, abs(s - Ms[l, k]), abs(t - Mt[l, k])), max(big, abs(s), abs(t))
print('%.3e %.3e' % (worst, big))
"""
d = os.path.join(W, 'algebra')
n_ = setup(d, MAG + '; ' + ALG, 13)
run(d, n_, 's')
p = subprocess.run([py, '-c', ALG_PY], cwd=d, capture_output=True, text=True)
try:
    worst, big = [float(v) for v in p.stdout.split()[-2:]]
except ValueError:
    worst, big = np.inf, 0.0
report('radial_algebra', p.returncode == 0 and worst < 1e-13*max(big, 1.0),
       'S_l, T_l of the algebra against the grid: %.1e (largest %.2f)' % (worst, big) if p.returncode == 0
       else (p.stdout + p.stderr).strip()[-200:])

# 7. kinematic induction in a forced run: the coupled field from the coupled flow
for name in ('f9_bes', 'lib_man', 'def_g21'):
    d = runs[name]
    names = [n for n in ('magnetic', 'magnetic_ic', 'magnetic_mantle') if field_io.exists(n, d)]
    coupled = {n: field(d, n) for n in names}
    for n in names:   # moved aside: a block that solve_ind.py does not write must fail, not compare with itself
        for f in field_io.files(n, 'npy', d) + field_io.files(n, 'text', d):
            if os.path.exists(f):
                os.replace(f, os.path.join(d, 'coupled_' + os.path.basename(f)))
    n_ = str(min(mk_run.NCPUS, 2))   # the ranks of the run (setup with lmax)
    for cmd in ([py, 'bin/just_induction.py'], mk_run.MPIEXEC.split() + ['-n', n_, py, 'bin/solve_ind.py']):
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        if p.returncode:
            sys.exit('%s: %s failed\n%s' % (d, cmd[-1], (p.stdout + p.stderr)[-3000:]))
    missing = [n for n in names if not field_io.exists(n, d)]
    err = max(rel(field(d, n), coupled[n]) for n in names if n not in missing) if len(missing) < len(names) else np.inf
    report('induction_' + name, not missing and err < 1e-10, 'the field (%s) %.1e from the coupled run%s'
           % (', '.join(names), err, '; not written: ' + ', '.join(missing) if missing else ''))

# 8. the stops
STOPS = [
    ('stop_deform_bes', BES + "; " + F9 + "; " + DEF, 'is not a Bessel function'),
    ('stop_default_bes', BES + "; " + F9, 'is not a Bessel function'),   # forcing 9 defaults to 'deforming'
    ('stop_default_f14', BES + "; " + F14M1, 'is not a Bessel function'),   # so does forcing 14 (10: below)
    ('stop_forced_walls', TWA2 + "; " + F9 + "; par.forced_walls = 'moving'", "forced_walls = 'moving'"),
    ('stop_walls_eig', "par.m = 1; par.forcing = 0; par.forced_walls = 'moving'", "forced_walls = 'moving'"),
    # a permeability that jumps at a wall that moves where B0 reaches it, in both pictures
    ('stop_mu_i2o_def', CHEB + "; " + F9 + "; " + DEF + "; par.mu_i2o = 2.0", 'inner wall radially where B0 reaches it'),
    ('stop_mu_i2o_rig', CHEB + "; " + F9 + "; " + RIG + "; par.mu_i2o = 2.0", ': mu_i2o = 2.0 must be 1'),
    ('stop_mu_i2o_bes', BES + "; " + F6 + "; par.mu_i2o = 2.0", ': mu_i2o = 2.0 must be 1'),   # forcing 6: 'rigid'
    ('stop_muw_def', MALKUS + "; " + F9 + "; " + TWA2 + "; par.muw_cmb = 2.0", 'outer wall radially where B0 reaches it'),
    ('stop_muw_rig', MALKUS + "; " + F9 + "; " + TWA2 + "; par.muw_cmb = 2.0; " + RIG, ': muw_cmb = 2.0 must be 1'),
    ('stop_muw_rig_pol', F9 + "; " + TWA2 + "; par.muw_cmb = 2.0; " + RIG, ': muw_cmb = 2.0 must be 1'),   # B0 'axial'
    ('stop_muw_thick', F9 + "; par.mantle = 'TWA'; par.c_cmb = 0.01; par.c1_cmb = 0.0; par.muw_cmb = 2.0",
     ': muw_cmb = 2.0 must be 1'),   # a layer with a thickness that does not conduct
    ('stop_mu', F9 + "; " + TWA2 + "; par.mu = 2.0", ': mu = 2.0 must be 1'),
    ('stop_mu_muw', F9 + "; " + TWA2 + "; par.c_cmb = 0.01; par.mu = 2.0; par.muw_cmb = 2.0; par.muw_icb = 2.0",
     ': mu = 2.0 and muw_cmb = 2.0 must be 1'),
    ('stop_muw_icb', F6 + "; " + TWA2 + "; par.c_icb = 0.01; par.muw_icb = 0.5", ': muw_icb = 0.5 must be 1'),
    # but not at walls that stay in place, as muw of an insulating layer, where B0 has no part that muw acts on (conducting
    # sheets that deform with the boundary under a poloidal B0, a layer with a thickness that does not conduct under a
    # toroidal B0), nor where B0 does not reach the walls
    ('muw_fixed', F6 + "; " + TWA2 + "; par.c_cmb = 0.01; par.muw_cmb = 2.0", None),   # forcing 6 moves the inner wall only
    ('mu_i2o_fixed', CHEB + "; " + F9O + "; par.mu_i2o = 2.0", None),   # the outer wall only
    ('muw_insulating', F9 + "; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.0; par.muw_cmb = 2.0", None),   # c = c1 = 0
    ('muw_sheet_def', F9 + "; " + TWA2 + "; par.muw_cmb = 2.0; par.muw_icb = 0.5", None),
    ('muw_thick_tor', MALKUS + "; " + F9O + "; par.mantle = 'TWA'; par.c_cmb = 0.01; par.c1_cmb = 0.0; par.muw_cmb = 2.0",
     None),
    ('mu_b0_off', "par.B0 = None; par.B0_tor = 'wall'; par.B0_tor_l = 1; " + F9 + "; " + TWA2 + "; par.c_cmb = 0.01; "
     "par.c_icb = 0.01; par.mu = 2.0; par.muw_cmb = 3.0; par.muw_icb = 0.5", None),   # 'wall' is 0 at both walls
    ('mu_b0_gen_off', "par.B0 = 'general'; par.B0_profiles = {1: 'r*(1 - r)**2*tanh(15*(r - 0.6))'}; " + F9O + "; " + TWA2
     + "; par.mu = 2.0", None),   # h = h' = 0 at the outer wall, the only one that moves
    ('deform_cheb', CHEB + "; " + F9 + "; " + DEF, None),
    ('deform_man', MAN + "; " + F9O + "; " + DEF, None),
    ('deform_bes_outer', BES + "; " + F9O + "; par.mantle = 'TWA'; par.c_cmb = 0.0; par.c1_cmb = 0.02; " + DEF, None),
    ('rigid_bes_f6', BES + "; " + F6, None),   # forcing 6 defaults to 'rigid'
    ('stop_bessel_eig', BES + "; par.m = 1; par.forcing = 0", 'needs a forced problem'),
    ('stop_bessel_scan', BES + "; " + RSF.replace(CHEB, '') + "; par.forcing_frequency = 0.3", 'frequency scan'),
]
for name, extra, msg in STOPS:
    d = os.path.join(W, name)
    rc, out = run(d, setup(d, MAG + "; par.B0 = 'axial'; " + extra, 21 if 'm = 2' in extra else 20), 's', check=False)
    ok = (rc == 0) if msg is None else (rc != 0 and msg in out)
    report(name, ok, 'runs' if msg is None and rc == 0 else (out.strip().splitlines() or ['(no output)'])[-1][:90])
# forcing 10 defaults to 'deforming': with a thin outer layer and B0 'axial' its matrix is that of 'deforming', in which the
# layer's rows take no u×B0, and not that of 'rigid'
F10T = ("par.B0 = 'axial'; par.m = 2; par.symm = 1; par.forcing = 10; par.forcing_frequency = 0.37; par.mantle = 'TWA'; "
        "par.c_cmb = 0.0; par.c1_cmb = 0.02")
A10 = {fw: load_csr(build('f10_' + fw, MAG + '; ' + F10T + ("; par.forced_walls = '%s'" % fw if fw != 'default' else ''), 21,
                          'sa') + '/A.npz') for fw in ('default', 'deforming', 'rigid')}
same, other = abs(A10['default'] - A10['deforming']).max(), abs(A10['default'] - A10['rigid']).max()
report('default_f10', same == 0 and other > 1e-8, "forcing 10's matrix by default against 'deforming' %.1e, against "
       "'rigid' %.1e" % (same, other))
p = subprocess.run([py, 'bin/just_induction.py', '0.1j'], cwd=runs['f9_bes'], capture_output=True, text=True)
report('stop_bessel_lambda', p.returncode != 0 and 'holds for lambda = i forcing_frequency' in p.stdout + p.stderr,
       ((p.stdout + p.stderr).strip().splitlines() or ['(no output)'])[-1][:90])
p = subprocess.run([py, 'bin/just_induction.py', '0.1j'], cwd=runs['def_g21'], capture_output=True, text=True)
report('stop_ind_lambda', p.returncode != 0 and 'writes no B.npz' in p.stdout + p.stderr,
       ((p.stdout + p.stderr).strip().splitlines() or ['(no output)'])[-1][:90])

# 9. x j_l'(x)/j_l(x) against mpmath
try:
    import mpmath as mp
except ImportError:
    mp = None
if mp is None:
    print('%-18s skipped (no mpmath)' % 'xdlogjl')
else:
    sys.path.insert(0, os.path.join(runs['f9_bes'], 'bin'))
    cwd = os.getcwd(); os.chdir(runs['f9_bes'])
    import utils as ut   # noqa: E402  (with that run's parameters.py)
    os.chdir(cwd)
    mp.mp.dps = 50
    worst = 0.0
    for l in (0, 1, 2, 3, 5, 10, 21, 40, 100, 300):
        for a in (1e-8, 1e-3, 0.3, 3.0, 6.7, 20.0, 80.0, 400.0, 3000.0, 1e4):
            for x in (a*(1 - 1j), a*(1 + 1j)):
                X = mp.mpc(x.real, x.imag)
                ref = complex(l - X*mp.besselj(l + 1.5, X)/mp.besselj(l + 0.5, X))
                worst = max(worst, abs(ut.xdlogjl(l, x) - ref)/max(abs(ref), 1.0))
    report('xdlogjl', worst < 1e-12, 'largest relative error %.1e (l to 300, |x| to 1.4e4)' % worst)

print('forced_check: ' + ('all cases ok' if not fail else 'FAILED'))
sys.exit(fail)
