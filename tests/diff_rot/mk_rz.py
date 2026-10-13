#!/usr/bin/env python3
'''
mk_rz.py NAME [key=val ...]: builds the run directory NAME (a copy of the repo's bin/ with its own parameters.py) for the
radiative-interior set-up of Mukhopadhyay, Bekki, Zhu & Gizon 2025, "Effects of the radiative interior on solar inertial
modes" (arXiv 2512.12691; docs/heat-equation.md §9), with any inner radius down to 0, and runs submatrices.py and
assemble.py there on one rank. Solve with solve_bekki.py (scipy, small N) or bin/solve.py (SLEPc + MUMPS).

Set-up: 'smoothed sun', Model S density fitted over the whole Sun (solar_density_window = 'sun'), their rotation
(diff_rot_type 'solar_P4', which is their eqs. 5-6) or a table, thermal wind balance (thermal_wind 2, thermal_wind_radial 0),
their radiative interior (solar_rz = 1: dS/dr of their eq. 10, ν and κ dropping to ν_RZ, κ_RZ below 0.70 R☉), stress-free
walls, zero entropy flux at the bottom and s = 0 at the top, outer radius 0.985 R☉, frame and Ω_ref/2π = 456 nHz.
kore is anelastic and has impenetrable walls (they are fully compressible, with a free surface).

keys (default):
  m (3), symm (-1), N (128), lmax (m + 63), rin (0.5): inner radius in R☉ (0 for the full sphere; N then counts both parities)
  rz (1): 0 for the aux1..aux5 top hat with A = 0 (δ = 0, no radiative interior, as mk_bekki.py) and constant ν, κ
  dsrz (1.32e-2): (∂s0/∂r)_RZ in erg/g/K/cm       nurz (1e10), kaprz (4e10): ν_RZ, κ_RZ in cm²/s
  nu (1e12), kappa (= nu): ν_CZ, κ_CZ             cp (3.46e8): c_p for the units of s
  drtype ('solar_P4'): or 'tabulated' with file= (diff_rot_window 'sun', degree 60, hold= x_h to hold Ω below x_h R☉)
  nmax (12), frame (456.0), tw (2), rad (0), bco (0): bco_thermal (their s = 0 at the top), dens ('model S'; 'fit' for smoothed_sun.py)
  extra: python statements appended to parameters.py

Writes NAME/bekki.txt (m, frame, the scales) for solve_bekki.py.
'''
import os, sys, shutil, subprocess, textwrap
import numpy as np

BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))   # the repo's bin

if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
    sys.exit(__doc__)
d = sys.argv[1]
kv = dict(a.split('=', 1) for a in sys.argv[2:])
known = 'm symm N lmax rin rz dsrz nurz kaprz nu kappa cp drtype file hold nmax frame tw rad bco dens extra'.split()
bad = [k for k in kv if k not in known]
if bad:
    sys.exit('unknown keys %s; known: %s' % (bad, ' '.join(known)))
g = lambda k, v: type(v)(kv[k]) if k in kv else v
m = g('m', 3); symm = g('symm', -1); N = g('N', 128); lmax = g('lmax', m + 63); rin = g('rin', 0.5)
rz = g('rz', 1); dsrz = g('dsrz', 1.32e-2); nurz = g('nurz', 1e10); kaprz = g('kaprz', 4e10)
nu = g('nu', 1e12); kappa = g('kappa', nu); cp = g('cp', 3.46e8)
drtype = kv.get('drtype', 'solar_P4'); fname = kv.get('file', ''); hold = kv.get('hold', None)
nmax = g('nmax', 12); frame = g('frame', 456.0); tw = g('tw', 2); rad = g('rad', 0); bco = g('bco', 0)
dens = kv.get('dens', 'model S'); extra = kv.get('extra', '')
if drtype == 'tabulated':
    if not fname:
        sys.exit("drtype = 'tabulated' needs file=")
    fname = os.path.abspath(fname)
    if not os.path.isfile(fname):
        sys.exit('no such file: ' + fname)
hold = None if hold is None else float(hold)

RSUN, GSURF = 6.9599e10, 2.7361e4      # cm, cm/s² (Model S, as smoothed_sun.py)
aux0 = 0.985
L = aux0*RSUN                          # unit of length
Om = 2*np.pi*frame*1e-9                # Ω_ref, rad/s
ViscosD = nu/(Om*L**2); ThermaD = kappa/(Om*L**2); Beyonce = GSURF/(L*Om**2)

T = textwrap.dedent(f'''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.set_scales()
par.m = {m}; par.symm = {symm}; par.ricb = {rin!r}/{aux0!r}; par.thermal = 1
par.model_type = 'smoothed sun'; par.solar_density = {dens!r}; par.solar_density_window = 'sun'
par.aux0 = {aux0!r}; par.aux1 = 0.0; par.aux2 = 0.1; par.aux3 = 0.68; par.aux4 = 0.06; par.aux5 = 0.0   # used only if solar_rz = 0
par.solar_rz = {rz}; par.solar_rz_dSdr = {dsrz*L/cp!r}
par.visc0 = {(nurz/nu if rz else 1.0)!r}; par.therm0 = {(kaprz/kappa if rz else 1.0)!r}; par.visc_pow = 0.0; par.therm_pow = 0.0
par.diff_rot = 1; par.diff_rot_type = {drtype!r}; par.diff_rot_frame = {frame!r}
par.diff_rot_file = {fname!r}; par.diff_rot_nmax = {nmax}; par.diff_rot_window = 'sun'; par.diff_rot_hold = {hold!r}
par.thermal_wind = {tw}; par.thermal_wind_radial = {rad}
par.Gaspard = 1.0; par.Beyonce = {Beyonce!r}; par.ViscosD = {ViscosD!r}; par.ThermaD = {ThermaD!r}
par.bci = 0; par.bco = 0; par.bci_thermal = 1; par.bco_thermal = {bco}
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
print(d, 'ok: rin %g R☉, frame %g nHz, ViscosD %.6e, ThermaD %.6e, Beyonce %.2f, solar_rz %d' % (rin, frame, ViscosD, ThermaD, Beyonce, rz))
