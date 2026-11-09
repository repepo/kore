#!/usr/bin/env python3
'''
The run helper of tests/wobble (NEW 2026-11-09): a homogeneous full sphere or exact spheroid, m 1, symm -1, no-slip wall,
timescale 'rotation', with or without the wobble of the mantle (bin/wobble.py).

  setup(d, **p)    builds the run directory d (a copy of kore's bin and a parameters.py: BASE, then p) and runs
                   submatrices.py there; p holds Ek, f (the flattening, 0: a sphere), N, lmax, ncpus and anything else
  solve(d, **p)    rewrites parameters.py with p (the parameters that change no operator: wobble, MoIX_M, dynflat_M,
                   itau, rtau, nev), runs assemble.py and solve.py on ncpus ranks, and returns one dict per eigenpair:
                   'lam' (complex) and, with wobble = 1, wobble.energies (torques, budget residual, energies)
  paper(q, alpha)  (MoIX_M, dynflat_M) of the mantle of Triana et al. (2019): a homogeneous shell whose two surfaces have
                   the flattening alpha, with q = rho_f/(rho_m (r0⁵ - 1)), in kore's units (the mean radius of the wall)
  ek_kore(E, alpha) kore's Ek for the paper's E = nu/(Omega a²) (a the equatorial radius of the wall)

MPIEXEC overrides 'mpiexec'.
'''
import os, sys, shutil, subprocess, textwrap, json
import numpy as np

BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))
MPIEXEC = os.environ.get('MPIEXEC', 'mpiexec').split()

BASE = textwrap.dedent('''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.Ek = {Ek!r}
par.set_scales()
par.structure = 'Boussinesq'; par.thermal = 0; par.ricb = 0; par.ThermaD = 0; par.Beyonce = 1.
par.bci = 1; par.bco = 1; par.bci_thermal = 0; par.bco_thermal = 0; par.diff_rot = 0; par.forcing = 0
par.m = 1; par.symm = -1; par.rpower_v = 4
par.centrifugal = {cent}; par.centrifugal_shape = 'spheroid'; par.centrifugal_f = {f!r}; par.centrifugal_q = 0.0
par.which_eigenpairs = 'TM'; par.field_format = 'npy'
par.petsc_opts['st_mat_mumps_icntl_14'] = 1000
''')

DEFAULTS = {'wobble': 0, 'MoIX_M': 1.0, 'dynflat_M': 0.0, 'itau': 1.0, 'rtau': 0.0, 'nev': 4}


def _write(d, p):
    q = dict(DEFAULTS); q.update({k: (v.item() if isinstance(v, np.generic) else v) for k, v in p.items()})
    lines = [BASE.format(Ek=q['Ek'], f=q['f'], cent=int(q['f'] != 0))]
    for k, v in q.items():
        if k not in ('Ek', 'f'):
            lines.append('par.%s = %r' % (k, v.item() if isinstance(v, np.generic) else v))
    open(os.path.join(d, 'bin', 'parameters.py'), 'w').write('\n'.join(lines) + '\n')
    json.dump(q, open(os.path.join(d, 'run.json'), 'w'))


def _run(cmd, d, log):
    with open(os.path.join(d, log), 'w') as fh:
        r = subprocess.run(cmd, cwd=d, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit('%s: %s failed\n%s' % (d, ' '.join(cmd[-2:]), open(os.path.join(d, log)).read()[-3000:]))


def setup(d, **p):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, 'bin'))
    for f in os.listdir(BIN):
        if os.path.isfile(os.path.join(BIN, f)) and f != 'parameters.py':
            shutil.copy(os.path.join(BIN, f), os.path.join(d, 'bin', f))
    _write(d, p)
    _run([sys.executable, 'bin/submatrices.py', str(p['ncpus'])], d, 'log_sub')
    return d


ANALYSE = textwrap.dedent('''
import sys, json
sys.path.insert(0, 'bin')
import numpy as np
from parameters import par
import utils as ut
ev = np.loadtxt('eigenvalues0.dat', ndmin=2)
lam = ev[:, 0] + 1j*ev[:, 1]
out = []
if par.wobble:
    import wobble as wb
    flow, M = ut.load_field('flow'), ut.load_field('wobble')
for i, l in enumerate(lam):
    rec = {'lam': [l.real, l.imag]}
    if par.wobble:
        e = wb.energies(flow[:, i], M[0, i], l)
        rec.update({k: ([v.real, v.imag] if isinstance(v, complex) else float(v)) for k, v in e.items()})
        rec['Mc'] = [M[0, i].real, M[0, i].imag]
    out.append(rec)
json.dump(out, open('result.json', 'w'))
''')


def solve(d, **p):
    q = json.load(open(os.path.join(d, 'run.json')))
    q.update(p)
    _write(d, q)
    for f in ('eigenvalues0.dat', 'result.json', 'no_conv_solution'):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    n = str(q['ncpus'])
    _run(MPIEXEC + ['-n', n, sys.executable, 'bin/assemble.py'], d, 'log_asm')
    _run(MPIEXEC + ['-n', n, sys.executable, 'bin/solve.py'], d, 'log_solve')
    if not os.path.exists(os.path.join(d, 'eigenvalues0.dat')):
        return []
    open(os.path.join(d, 'analyse.py'), 'w').write(ANALYSE)
    _run([sys.executable, 'analyse.py'], d, 'log_analyse')
    out = json.load(open(os.path.join(d, 'result.json')))
    for r in out:
        r['lam'] = complex(*r['lam'])
        for k, v in list(r.items()):
            if isinstance(v, list):
                r[k] = complex(*v)
    return out


def axes(f):
    return (1 - f)**(-1/3), (1 - f)**(2/3)


def paper(q, alpha):
    '''(MoIX_M, dynflat_M): A_M = (4π/15)(1 - α)(2 - 2α + α²)/q in units of rho_f a⁵ (their eq. 9), times a⁵ in kore's
    units; f(α) = α + α²/2 - α³/4 (their dynamical flattening)'''
    a = axes(alpha)[0]
    A = (4*np.pi/15)*(1 - alpha)*(2 - 2*alpha + alpha**2)/q*a**5
    return A, alpha + alpha**2/2 - alpha**3/4


def ek_kore(E, alpha):
    '''kore's Ek (on the mean radius) for E = nu/(Omega a²)'''
    return E*axes(alpha)[0]**2


def spin_over(alpha):
    '''the inviscid spin-over of the spheroid, 2/(1 + (1 - α)²) (the paper's omega_*)'''
    return 2/(1 + (1 - alpha)**2)
