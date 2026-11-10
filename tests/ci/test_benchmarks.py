#!/usr/bin/env python3
'''
Benchmarks from the literature that GitHub runs on every push to main (.github/workflows/anelastic_log.yml), ported from
the tests of the conductive_IC branch. To run them locally:
> cd tests/ci; pytest -v test_benchmarks.py

- The onset of convection (Boussinesq, no-slip, Pr = 1, m = 9): in a full sphere heated from within (Jones, Soward &
  Mussa 2000, Ta = 1e9) and in a shell, ricb 0.35, heated differentially (Dormy, Soward, Jones, Jault & Cardin 2004,
  Ek 2e-5). The leading growth rate, solved at 0.999 and 1.001 times the critical Rayleigh number that conductive_IC
  (2d84351) found at the same resolution, must change sign between them; the root, interpolated linearly, and the drift
  frequency there must equal conductive_IC's to the 6 digits it recorded. conductive_IC's 'internal' heating is
  stratification 'linear' with Beyonce = BV2 = -Ra Ek²/Pr, and its 'differential' heating 'inverse square' with Beyonce =
  BV2 ricb/(1 - ricb), Ra being its Rayleigh number on the outer radius (Ra_gap/(1 - ricb)³).
- The torsional modes of the quadrupolar field S2 (Luo & Jackson 2022, Proc. R. Soc. A 478, 20210982, table 1): inviscid
  full sphere, insulating wall, B0 'Luo_S2' of unit magnetic energy, Le 1e-4, Lundquist number 2e4, m 0, in Alfvén time
  units, N 128. Each eigenvalue must equal conductive_IC's (2d84351) at the same resolution to 5e-7 (in practice 3e-10;
  conductive_IC's own test allows 5e-7 for GitHub's MUMPS) and Luo & Jackson's converged values to 2e-6 and 1e-5 (kore is
  8e-7 and 5e-6 from them). With inviscid_basis 'high' (the rows in C^(3), C^(2), main's only basis before 2026-11-09) the
  modes are 2e-5 and 1.2e-4 from Luo & Jackson's, and this test fails.
- The onset of thermo-compositional convection with equal Prandtl and Schmidt numbers (Silva, Mather & Simitev 2019, GAFD
  113, section 5.1; from conductive_IC's test_thermocompositional): the two scalars then act as one co-density, and the
  critical curve is the straight line Rt + Rc = Ra0, Rt = Ra cos α, Rc = Ra sin α (their eq. 18). For α = 0 (heat only),
  π/2 (composition only) and π/4 (both) the leading growth rate at 0.999 and 1.001 times Ra0/(cos α + sin α) must change
  sign, the root interpolated linearly must give Ra0, and the drift frequency there ω_c, both conductive_IC's (2d84351,
  root found to 1e-12 at the same resolution) to 1e-5. Stress-free walls, fixed temperature and composition, sources
  inside (stratification and comp_stratification 'linear', Beyonce = -Rt Ek²/Pr/(1 - ricb)⁶, BeyonceC the same with Rc,
  Sc: conductive_IC's BV2 and BV2_comp), ricb 0.35, τ = 1e4 (Ek = 2(1 - ricb)²/τ), m 7, N 64.
- Gravito-inertial modes of a stably stratified shell (Dintrans, Rieutord & Valdettaro 1999, J. Fluid Mech. 398, 271,
  fig. 4; from conductive_IC's test_gravitoinertial): ricb 0.35, stress-free, fixed temperatures, Ek 2e-7, Pr 1, m 0,
  N² = Beyonce r² with N/2Ω = 0.2, 1, 2.5 (Beyonce = 0.16, 4, 25), N 150, lmax 311. Each eigenvalue must equal
  conductive_IC's (2d84351, tests/dintrans1999/reference.dintrans99) to 1e-6 and Dintrans et al.'s to 3e-3, 3e-3 and
  6e-4 (conductive_IC is 2.0e-3, 8e-5 and 5.7e-4 from them at this resolution).
- The viscous decay of the toroidal flow without rotation in a full sphere (from conductive_IC's norotation test): the
  rate of degree l is -k², k the first root of j_l(k) (no-slip) or of k j_l'(k) = j_l(k) (stress-free), the same for
  m = 0, 1, 2.
'''
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import spherical_jn

KORE = Path(__file__).resolve().parents[2]
NCPUS = 2

HEAD = '''from params_default import default_params
import numpy as np
par = default_params()
'''


def run(d, params, solve_args=('',)):
    '''builds the run directory d (a copy of bin/ with params as its parameters.py), runs submatrices and assemble, then
    solve.py once per entry of solve_args; returns the eigenvalues of each solve'''
    if not (d / 'bin').exists():
        shutil.copytree(KORE / 'bin', d / 'bin', ignore=shutil.ignore_patterns('__pycache__'))
    (d / 'bin' / 'parameters.py').write_text(HEAD + params + '\npar.ncpus = %d\n' % NCPUS, encoding='utf-8')
    for cmd in ('./bin/submatrices.py %d' % NCPUS, 'mpiexec -n %d ./bin/assemble.py' % NCPUS):
        subprocess.run(cmd, shell=True, check=True, cwd=d, stdout=subprocess.DEVNULL)
    out = []
    for a in solve_args:
        (d / 'eigenvalues0.dat').unlink(missing_ok=True)
        subprocess.run('mpiexec -n %d ./bin/solve.py %s' % (NCPUS, a), shell=True, check=True, cwd=d,
                       stdout=subprocess.DEVNULL)
        e = np.loadtxt(d / 'eigenvalues0.dat').reshape(-1, 2)
        out.append(e[:, 0] + 1j*e[:, 1])
    return out


# ------------------------------------------------------------------------------------------------- onset of convection
EK_JONES = np.sqrt(4/1e9)   # Ta = 1e9
ONSET = {   # case: (parameters but the buoyancy, Beyonce from the gap Rayleigh number, conductive_IC's Ra_c and ω_c)
    'jones2000': ('''par.structure = 'Boussinesq'; par.ricb = 0; par.m = 9; par.symm = 1
par.Ek = %r; par.Etherm = par.Ek; par.timescale = 'rotation'; par.bco = 1   # no-slip
par.thermal = 1; par.stratification = 'linear'; par.bco_thermal = 1   # fixed heat flux
par.N = 118; par.lmax = 124
''' % EK_JONES, lambda Ra: -Ra*EK_JONES**2, 4.66986e6, -1.93444e-2),
    'dormy2004': ('''par.structure = 'Boussinesq'; par.ricb = 0.35; par.m = 9; par.symm = 1
par.Ek = 2e-5; par.Etherm = par.Ek; par.timescale = 'rotation'; par.bci = 1; par.bco = 1   # no-slip
par.thermal = 1; par.stratification = 'inverse square'; par.bci_thermal = 0; par.bco_thermal = 0   # fixed temperatures
par.N = 74; par.lmax = 156
''', lambda Ra: -Ra/0.65**3*2e-5**2*0.35/0.65, 1.65404e6, -1.10162e-2),
}
SOLVER = '''
par.rtau = 0.0; par.itau = 0.0; par.which_eigenpairs = 'TR'; par.nev = 5; par.tol = 1e-15; par.maxit = 100
par.prescale = 1
'''


@pytest.mark.parametrize('case', ONSET)
def test_onset(tmp_path, case):
    params, beyonce, Rac, wc = ONSET[case]
    lead = []
    for f in (0.999, 1.001):
        eig = run(tmp_path / ('%s_%g' % (case, f)), params + 'par.set_scales()\npar.Beyonce = %r\n' % beyonce(f*Rac)
                  + SOLVER)[0]
        lead.append(eig[np.argmax(eig.real)])
    s0, s1 = lead[0].real, lead[1].real
    assert s0 < 0 < s1, '%s: growth rates %r, %r at 0.999 and 1.001 Ra_c: no onset between them' % (case, s0, s1)
    t = -s0/(s1 - s0)
    ra = Rac*(0.999 + 0.002*t)
    w = lead[0].imag + t*(lead[1].imag - lead[0].imag)
    assert abs(ra/Rac - 1) < 1e-5, '%s: Ra_c %.6e, conductive_IC %.6e' % (case, ra, Rac)
    assert abs(w/wc - 1) < 1e-5, '%s: ω_c %.6e, conductive_IC %.6e' % (case, w, wc)


# ------------------------------------------------------------------------------------- torsional modes of Luo & Jackson
def test_torsional_s2(tmp_path):
    params = '''par.structure = 'Boussinesq'; par.ricb = 0; par.m = 0; par.symm = 1
par.Ek = 0; par.magnetic = 1; par.B0 = 'Luo_S2'; par.cnorm = 'mag_energy'
par.Le = 1e-4; par.Em = par.Le/2e4; par.timescale = 'Alfven'
par.set_scales()
par.N = 128; par.lmax = 127
par.rtau = -0.0142; par.itau = 1.9095; par.nev = 3; par.prescale = 1
par.petsc_opts['st_mat_mumps_icntl_14'] = 100   # extra MUMPS workspace in per cent (INFOG(1)=-9 without it)
'''
    # Luo & Jackson's values and tolerances, and conductive_IC's eigenvalues (2d84351, tests/torsional_s2/reference.eig)
    modes = [(-0.0065952461 + 1.0335959942j, 2e-6, -6.594806397439181324e-03 + 1.033595241792488961e+00j),   # mode 1
             (-0.0141833345 + 1.9095123626j, 1e-5, -1.418571531701754047e-02 + 1.909522005464007854e+00j)]   # mode 2
    found = run(tmp_path, params, ('-eps_target -0.0066+1.0336i', ''))   # mode 1, then mode 2 (the target above)
    for lam, (lj, tol, cic) in zip(found, modes):
        z = lam[np.argmin(abs(lam - cic))]
        assert abs(z - cic)/abs(cic) < 5e-7, 'torsional mode %r, conductive_IC %r' % (z, cic)
        assert abs(z - lj)/abs(lj) < tol, 'torsional mode %r, Luo & Jackson %r' % (z, lj)


# ---------------------------------------------------------------------- thermo-compositional onset of Silva et al. 2019
SILVA_EK = 2*0.65**2/1e4   # τ = 1e4
SILVA = '''par.structure = 'Boussinesq'; par.ricb = 0.35; par.m = 7; par.symm = 1
par.Ek = %r; par.Etherm = par.Ek; par.Ecomp = par.Ek; par.timescale = 'rotation'   # Pr = Sc = 1
par.bci = 0; par.bco = 0   # stress-free
par.thermal = 1; par.stratification = 'linear'; par.bci_thermal = 0; par.bco_thermal = 0
par.compositional = 1; par.comp_stratification = 'linear'; par.bci_compositional = 0; par.bco_compositional = 0
par.N = 64; par.lmax = 70
par.set_scales()
par.Beyonce = %r; par.BeyonceC = %r
'''
SILVA_RA0, SILVA_WC = 2.580076883e5, -3.005663056e-2   # conductive_IC 2d84351, α = 0 and π/2 alike


@pytest.mark.parametrize('alpha', (0, np.pi/2, np.pi/4), ids=('heat', 'composition', 'both'))
def test_silva_codensity(tmp_path, alpha):
    lead = []
    for f in (0.999, 1.001):
        Ra = f*SILVA_RA0/(np.cos(alpha) + np.sin(alpha))
        by = [-Ra*np.cos(alpha)/0.65**6*SILVA_EK**2, -Ra*np.sin(alpha)/0.65**6*SILVA_EK**2]
        eig = run(tmp_path / ('silva_%g' % f), SILVA % (SILVA_EK, by[0], by[1]) + SOLVER)[0]
        lead.append(eig[np.argmax(eig.real)])
    s0, s1 = lead[0].real, lead[1].real
    assert s0 < 0 < s1, 'α %.4f: growth rates %r, %r at 0.999 and 1.001 Ra_c: no onset between them' % (alpha, s0, s1)
    t = -s0/(s1 - s0)
    ra0 = SILVA_RA0*(0.999 + 0.002*t)
    w = lead[0].imag + t*(lead[1].imag - lead[0].imag)
    assert abs(ra0/SILVA_RA0 - 1) < 1e-5, 'α %.4f: Ra0 %.9e, conductive_IC %.9e' % (alpha, ra0, SILVA_RA0)
    assert abs(w/SILVA_WC - 1) < 1e-5, 'α %.4f: ω_c %.9e, conductive_IC %.9e' % (alpha, w, SILVA_WC)


# ------------------------------------------------------------------- gravito-inertial modes of Dintrans et al. 1999
DINTRANS = [   # Beyonce = (N/Ω)², Dintrans et al.'s eigenvalue (fig. 4, in units of 2Ω, doubled to kore's 1/Ω), its tolerance,
               # and conductive_IC's at this resolution (2d84351, tests/dintrans1999/reference.dintrans99)
    ((2*0.2)**2, 2*-6.088e-3 + 2j*0.550, 3e-3, -1.438150177613099073e-02 + 1.100404695070275496e+00j),
    ((2*1.0)**2, 2*-3.854e-3 + 2j*0.755, 3e-3, -7.708280904927911764e-03 + 1.509875820573002958e+00j),
    ((2*2.5)**2, 2*-5.190e-3 + 2j*1.130, 6e-4, -9.316411720512379063e-03 + 2.260706561532888959e+00j)]


@pytest.mark.parametrize('k', range(len(DINTRANS)), ids=('N0.2', 'N1', 'N2.5'))
def test_dintrans_gravitoinertial(tmp_path, k):
    beyonce, din, tol, cic = DINTRANS[k]
    params = '''par.structure = 'Boussinesq'; par.ricb = 0.35; par.m = 0; par.symm = 1
par.Ek = 2e-7; par.Etherm = par.Ek; par.timescale = 'rotation'; par.bci = 0; par.bco = 0   # stress-free, Pr = 1
par.thermal = 1; par.stratification = 'linear'; par.bci_thermal = 0; par.bco_thermal = 0   # fixed temperatures
par.N = 150; par.lmax = 311
par.set_scales()
par.Beyonce = %r
par.rtau = %r; par.itau = %r; par.nev = 3; par.prescale = 1
''' % (beyonce, din.real, din.imag)
    lam = run(tmp_path, params)[0]
    z = lam[np.argmin(abs(lam - cic))]
    assert abs(z - cic)/abs(cic) < 1e-6, 'N/Ω %g: %r, conductive_IC %r' % (np.sqrt(beyonce), z, cic)
    assert abs(z - din)/abs(din) < tol, 'N/Ω %g: %r, Dintrans et al. %r' % (np.sqrt(beyonce), z, din)


# ---------------------------------------------------------------------------------------- toroidal decay, no rotation
def k_tor(l, bco):
    '''the first k of the toroidal decay mode T = j_l(k r) of degree l in a full sphere: j_l(k) = 0 (no-slip, bco 1)
    or k j_l'(k) = j_l(k) (stress-free, bco 0)'''
    f = (lambda k: spherical_jn(l, k)) if bco == 1 else \
        (lambda k: k*spherical_jn(l, k, derivative=True) - spherical_jn(l, k))
    k = np.linspace(0.5, 20, 2000)
    i = np.where(np.sign(f(k[:-1])) != np.sign(f(k[1:])))[0][0]
    return brentq(f, k[i], k[i + 1], xtol=1e-14, rtol=1e-15)


def test_toroidal_decay(tmp_path):
    for bco in (1, 0):
        for l in (2, 3):
            lam = -k_tor(l, bco)**2
            found = []
            for m in (0, 1, 2):
                symm = 1 if (l - m) % 2 == 1 else -1   # T_l has l - m odd for symm = 1
                params = '''par.structure = 'Boussinesq'; par.ricb = 0; par.m = %d; par.symm = %d; par.bco = %d
par.Ek = 1.0; par.timescale = 'viscous'
par.set_scales()
par.Gaspard = 0   # no rotation
par.N = 24; par.lmax = par.m + 3
par.rtau = %r; par.itau = 0.0; par.nev = 4
''' % (m, symm, bco, 1.001*lam)
                eig = run(tmp_path / ('l%d_m%d_bco%d' % (l, m, bco)), params)[0]
                found.append(eig[np.argmin(abs(eig - lam))])
            np.testing.assert_allclose(found, lam, rtol=1e-10, atol=1e-10, err_msg='l=%d bco=%d' % (l, bco))
