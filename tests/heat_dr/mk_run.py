#!/usr/bin/env python3
# usage: mk_run.py NAME 'par.x=..; par.y=..'  -> builds NAME/ (bin copy + parameters.py) and runs submatrices + assemble
import os, sys, shutil, subprocess, textwrap
BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))   # the repo's bin
T = textwrap.dedent('''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.set_scales()
par.ricb=0.35; par.m=1; par.symm=1; par.thermal=1; par.model_type='user def'
par.diff_rot=1; par.diff_rot_type='conical'; par.diff_rot_ampl=-0.25; par.thermal_wind=1
par.aux0=0.99; par.aux1=0.5; par.aux2=0.03; par.aux3=0.85; par.aux4=0.03; par.aux5=2.5
par.Gaspard=1.; par.Beyonce=10.; par.ViscosD=1e-3; par.ThermaD=1e-3
par.visc0=1.0; par.rvisc=0.6; par.hvisc=0.04
par.bci=0; par.bco=0; par.bci_thermal=0; par.bco_thermal=0
par.ncpus=1; par.N=48; par.lmax=20
par.rtau=0; par.itau=1.0; par.smopo=0; par.nev=20; par.which_eigenpairs='TM'
par.rpower_u=5; par.rhopower_u=1; par.rpower_v=3; par.rhopower_v=1; par.rpower_pp=4; par.rhopower_pp=4
par.prescale=0
{extra}
''')
d = sys.argv[1]; extra = sys.argv[2] if len(sys.argv) > 2 else ''
if os.path.isdir(d): shutil.rmtree(d)
os.makedirs(d + '/bin')
for f in os.listdir(BIN):
    if os.path.isfile(BIN + '/' + f) and f not in ('parameters.py', 'mpi4py.py'): shutil.copy(BIN + '/' + f, d + '/bin/' + f)
open(d + '/bin/parameters.py', 'w').write(T.format(extra=extra))
for cmd, log in (([sys.executable, 'bin/submatrices.py', '1'], 'log_sub'), ([sys.executable, 'bin/assemble.py'], 'log_asm')):
    p = subprocess.run(cmd, cwd=d, stdout=open(d + '/' + log, 'w'), stderr=subprocess.STDOUT)
    if p.returncode: print(d, 'FAILED', cmd[1]); print(open(d + '/' + log).read()[-3000:]); sys.exit(1)
print(d, 'ok')
