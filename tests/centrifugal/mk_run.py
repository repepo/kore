#!/usr/bin/env python3
'''
usage: mk_run.py NAME 'par.x = ..; par.y = ..' [solve]
builds NAME/ (a copy of the repo's bin/ and its parameters.py) with the centrifugal test defaults below and runs
submatrices.py and assemble.py (A.npz, B.npz); with 'solve' also solve.py and spin_doctor.py (mpiexec -n par.ncpus for
assemble and solve; set MPIEXEC to change the launcher). NEW (2026-11-05).
Defaults: the inviscid Boussinesq full sphere (ViscosD = 0, thermal = 0), m 1, symm -1 (the spin-over), N 24, lmax 20,
centrifugal = 1 with centrifugal_shape 'uniform' and f = 0, one rank, target 1.0i.
'''
import os, sys, shutil, subprocess, textwrap
BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))   # the repo's bin
T = textwrap.dedent('''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.set_scales()
par.structure = 'Boussinesq'; par.thermal = 0; par.ricb = 0; par.ViscosD = 0; par.ThermaD = 0; par.Gaspard = 1.; par.Beyonce = 1.
par.bci = 1; par.bco = 1; par.bci_thermal = 0; par.bco_thermal = 0; par.diff_rot = 0
par.m = 1; par.symm = -1; par.N = 24; par.lmax = 20; par.ncpus = 1; par.nev = 6; par.rtau = 0.0; par.itau = 1.0
par.which_eigenpairs = 'TM'; par.prescale = 0
par.rpower_u = 5; par.rhopower_u = 1; par.rpower_v = 3; par.rhopower_v = 1; par.rpower_pp = 4; par.rhopower_pp = 4
par.centrifugal = 1; par.centrifugal_shape = 'uniform'; par.centrifugal_f = 0.0; par.centrifugal_q = 0.0
{extra}
''')
d = sys.argv[1]; extra = sys.argv[2] if len(sys.argv) > 2 else ''; solve = 'solve' in sys.argv[3:]
if os.path.isdir(d): shutil.rmtree(d)
os.makedirs(d + '/bin')
for f in os.listdir(BIN):
    if os.path.isfile(BIN + '/' + f) and f not in ('parameters.py', 'mpi4py.py'): shutil.copy(BIN + '/' + f, d + '/bin/' + f)
open(d + '/bin/parameters.py', 'w').write(T.format(extra=extra))
np_ = subprocess.run([sys.executable, '-c', "import sys; sys.path.insert(0, 'bin'); from parameters import par; print(par.ncpus)"],
                     cwd=d, capture_output=True, text=True).stdout.strip()
mpi = os.environ.get('MPIEXEC', 'mpiexec').split()
par_ = (mpi + ['-n', np_]) if np_ != '1' else []
steps = [([sys.executable, 'bin/submatrices.py', np_], 'log_sub'), (par_ + [sys.executable, 'bin/assemble.py'], 'log_asm')]
if solve:
    steps += [(mpi + ['-n', np_, sys.executable, 'bin/solve.py'], 'log_solve'), ([sys.executable, 'bin/spin_doctor.py', np_], 'log_sd')]
for cmd, log in steps:
    p = subprocess.run(cmd, cwd=d, stdout=open(d + '/' + log, 'w'), stderr=subprocess.STDOUT)
    if p.returncode: print(d, 'FAILED', ' '.join(cmd[-2:])); print(open(d + '/' + log).read()[-3000:]); sys.exit(1)
print(d, 'ok')
