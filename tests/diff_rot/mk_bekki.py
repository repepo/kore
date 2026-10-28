#!/usr/bin/env python3
'''
mk_bekki.py NAME [key=val ...]: builds the run directory NAME (a copy of the repo's bin/ with its own parameters.py) for
the solar inertial-mode benchmark of Bekki, Cameron & Gizon 2022 and Mukhopadhyay et al. 2025 (docs/benchmarks.md),
and runs submatrices.py and assemble.py there on one rank. Solve with solve_bekki.py.

Set-up: structure 'old model S prototype' (the Model S density), shell 0.71-0.985 R☉ (r_out = 0.985),
dS0/dr = 0 (δ = 0), ν = κ = 1e12 cm²/s, stress-free walls with ∂ᵣs = 0, time unit 1/Ω_ref with Ω_ref/2π = frame nHz.

keys (default):
  m (1), symm (-1), N (64), lmax (m + 63)
  dr (1): diff_rot; dr=0 for uniform rotation
  drtype ('tabulated'): diff_rot_type; 'tabulated' needs file=, 'Y20' with amp = P2/P0 is the radially uniform
                        Snodgrass profile (amp = -79.03/426.86 with frame = 426.86)
  file: the .npz of 'tabulated' (data/diff_rot.npz of the Mukhopadhyay et al. 2025 data set, doi:10.17617/3.NP26AI,
        or the output of ls_to_tabulated.py for the Larson & Schou 2018 table); a relative path is taken from the
        current directory
  nmax (12): diff_rot_nmax      frame (456.0): Ω_ref/2π in nHz, also diff_rot_frame      amp (1.0): diff_rot_ampl
  thermal (1)    tw (2): thermal_wind    rad (0): thermal_wind_radial (0 drops u_r ∂S_tw/∂r, as the papers do)
  ds (0.0): Δθs in erg/g/K for tw = 3 (dSdtheta = ds/cp, negative: warm poles)    cp (3.46e8, Model S; the papers use 4.17e8)
  nu (1e12), kappa (= nu): cm²/s
  extra: python statements appended to parameters.py, e.g. "extra=par.thermal_wind_amplitude = 0.5"

Writes NAME/bekki.txt (m, frame, the scales) for solve_bekki.py. The matrices take ~0.5 GB at the defaults.
'''
import os, sys, shutil, subprocess, textwrap
import numpy as np

BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))   # the repo's bin

if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
    sys.exit(__doc__)
d = sys.argv[1]
kv = dict(a.split('=', 1) for a in sys.argv[2:])
known = 'm symm N lmax dr drtype file nmax frame amp thermal tw rad ds cp nu kappa extra'.split()
bad = [k for k in kv if k not in known]
if bad:
    sys.exit('unknown keys %s; known: %s' % (bad, ' '.join(known)))
g = lambda k, v: type(v)(kv[k]) if k in kv else v
m = g('m', 1); symm = g('symm', -1); N = g('N', 64); lmax = g('lmax', m + 63)
dr = g('dr', 1); drtype = kv.get('drtype', 'tabulated'); fname = kv.get('file', '')
nmax = g('nmax', 12); frame = g('frame', 456.0); amp = g('amp', 1.0)
thermal = g('thermal', 1); tw = g('tw', 2); rad = g('rad', 0)
ds = g('ds', 0.0); cp = g('cp', 3.46e8); nu = g('nu', 1e12); kappa = g('kappa', nu)
extra = kv.get('extra', '')
if (lmax - m + 1) % 2:
    sys.exit('lmax - m + 1 must be even (one rank)')
if dr and drtype == 'tabulated':
    if not fname:
        sys.exit("drtype = 'tabulated' needs file=")
    fname = os.path.abspath(fname)
    if not os.path.isfile(fname):
        sys.exit('no such file: ' + fname)

RSUN, GSURF = 6.9599e10, 2.7361e4      # cm, cm/s² (Model S, as smoothed_sun.py)
r_out = 0.985
L = r_out*RSUN                         # unit of length
Om = 2*np.pi*frame*1e-9                # Ω_ref, rad/s
ViscosD = nu/(Om*L**2); ThermaD = kappa/(Om*L**2); Beyonce = GSURF/(L*Om**2)

T = textwrap.dedent(f'''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.set_scales()
par.m = {m}; par.symm = {symm}; par.ricb = 0.71/{r_out!r}; par.thermal = {thermal}
par.structure = 'old model S prototype'; par.r_out = {r_out!r}
par.stratification = 'odd top hat'; par.aux1 = 0.0; par.aux2 = 0.1; par.aux3 = 0.68; par.aux4 = 0.06; par.aux5 = 0.0   # dS0/dr = 0
par.diffusivity = 'uniform'
par.diff_rot = {dr}; par.diff_rot_type = {drtype!r}; par.diff_rot_ampl = {amp!r}
par.diff_rot_file = {fname!r}; par.diff_rot_nmax = {nmax}; par.diff_rot_frame = {frame!r}
par.thermal_wind = {tw}; par.thermal_wind_radial = {rad}; par.dSdtheta = {ds/cp!r}
par.Gaspard = 1.0; par.Beyonce = {Beyonce!r}; par.ViscosD = {ViscosD!r}; par.ThermaD = {ThermaD!r}
par.bci = 0; par.bco = 0; par.bci_thermal = 1; par.bco_thermal = 1
par.ncpus = 1; par.N = {N}; par.lmax = {lmax}
par.rtau = 0; par.itau = 1.0; par.nev = 20; par.which_eigenpairs = 'TM'
{extra}
''')

if os.path.isdir(d):
    shutil.rmtree(d)
os.makedirs(d + '/bin')
for f in os.listdir(BIN):
    if os.path.isfile(BIN + '/' + f) and f not in ('parameters.py', 'mpi4py.py'):
        shutil.copy(BIN + '/' + f, d + '/bin/' + f)
open(d + '/bin/parameters.py', 'w').write(T)
open(d + '/bekki.txt', 'w').write('m %d\nframe %r\nViscosD %r\nThermaD %r\nBeyonce %r\n' % (m, frame, ViscosD, ThermaD, Beyonce))
for cmd, log in (([sys.executable, 'bin/submatrices.py', '1'], 'log_sub'), ([sys.executable, 'bin/assemble.py'], 'log_asm')):
    p = subprocess.run(cmd, cwd=d, stdout=open(d + '/' + log, 'w'), stderr=subprocess.STDOUT)
    if p.returncode:
        print(d, 'FAILED', cmd[1]); print(open(d + '/' + log).read()[-3000:]); sys.exit(1)
print(d, 'ok: frame %g nHz, ViscosD %.6e, ThermaD %.6e, Beyonce %.2f' % (frame, ViscosD, ThermaD, Beyonce))
