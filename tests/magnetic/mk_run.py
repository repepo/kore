#!/usr/bin/env python3
'''
usage: mk_run.py NAME 'par.x = ..; par.y = ..' [solve]
builds NAME/ (a copy of the repo's bin/ and its parameters.py) with the magnetic test defaults below and runs submatrices.py;
with 'solve' also assemble.py, solve.py and spin_doctor.py (mpiexec -n par.ncpus; set MPIEXEC to change the launcher).
Defaults: 'Wu 2005' (r_out 0.99), ricb 0.35, no-slip walls, B0 'axial' (cnorm 'mag_energy'), insulating walls, Ek = Em = 1e-3,
Le = sqrt(1e-3) (Lambda = Pm = 1), m 1, symm -1, N 96, lmax 24, 2 ranks, target 1.0i.
'''
import os, sys, shutil, subprocess, textwrap
BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))   # the repo's bin
T = textwrap.dedent('''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.Ek = 1e-3; par.Em = 1e-3; par.Le = np.sqrt(1e-3)
par.set_scales()
par.structure = 'Wu 2005'; par.r_out = 0.99
par.ricb = 0.35; par.bci = 1; par.bco = 1; par.m = 1; par.symm = -1
par.magnetic = 1; par.B0 = 'axial'; par.B0_l = 1; par.cnorm = 'mag_energy'; par.innercore = 'insulator'; par.mantle = 'insulator'
par.N = 96; par.lmax = 24; par.ncpus = 2; par.nev = 5; par.rtau = 0.0; par.itau = 1.0
{extra}
''')
d = sys.argv[1]; extra = sys.argv[2] if len(sys.argv) > 2 else ''; solve = 'solve' in sys.argv[3:]
if os.path.isdir(d): shutil.rmtree(d)
os.makedirs(d + '/bin')
for f in os.listdir(BIN):
    if os.path.isfile(BIN + '/' + f) and f != 'parameters.py': shutil.copy(BIN + '/' + f, d + '/bin/' + f)
open(d + '/bin/parameters.py', 'w').write(T.format(extra=extra))
np_ = subprocess.run([sys.executable, '-c', "import sys; sys.path.insert(0, 'bin'); from parameters import par; print(par.ncpus)"],
                     cwd=d, capture_output=True, text=True).stdout.strip()
mpi = os.environ.get('MPIEXEC', 'mpiexec').split()
steps = [([sys.executable, 'bin/submatrices.py', np_], 'log_sub')]
if solve:
    steps += [(mpi + ['-n', np_, sys.executable, 'bin/assemble.py'], 'log_asm'), (mpi + ['-n', np_, sys.executable, 'bin/solve.py'], 'log_solve'),
              ([sys.executable, 'bin/spin_doctor.py', np_], 'log_sd')]
for cmd, log in steps:
    p = subprocess.run(cmd, cwd=d, stdout=open(d + '/' + log, 'w'), stderr=subprocess.STDOUT)
    if p.returncode: print(d, 'FAILED', ' '.join(cmd[-2:])); print(open(d + '/' + log).read()[-3000:]); sys.exit(1)
print(d, 'ok')
