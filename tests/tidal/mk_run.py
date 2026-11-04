#!/usr/bin/env python3
'''
usage: mk_run.py NAME 'par.x = ..; par.y = ..'

Builds the run directory NAME (a copy of kore's bin and a parameters.py: the defaults below, then the changes) and runs
submatrices, assemble, solve and spin_doctor there on NCPUS ranks (environment variable, default 2; MPIEXEC, default
mpiexec). The *.mtx files are deleted at the end. Default run: an incompressible fluid ('Boussinesq') in a rotating
shell, ricb 0.35, no-slip walls, m 2, Ek 1e-3, N 96, lmax about N/2 (set g = lmax/N to change it), eigenproblem.
'''
import os, sys, shutil, subprocess, textwrap

BIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bin'))   # the repo's bin
NCPUS = int(os.environ.get('NCPUS', 2))
MPIEXEC = os.environ.get('MPIEXEC', 'mpiexec')

T = textwrap.dedent('''
from params_default import default_params
import numpy as np
par = default_params()
par.timescale = 'rotation'
par.set_scales()
par.ricb = 0.35; par.m = 2; par.symm = 1
par.thermal = 0; par.structure = 'Boussinesq'; par.stratification = None; par.r_out = 0.99
par.aux1 = 0.5; par.aux2 = 0.03; par.aux3 = 0.85; par.aux4 = 0.03; par.aux5 = 2.5
par.Gaspard = 1.; par.Beyonce = 10.; par.ViscosD = 1e-3; par.ThermaD = 1e-3
par.diff_rot = 0; par.diff_rot_type = 'conical'; par.diff_rot_ampl = -0.25; par.thermal_wind = 0
par.bci = 1; par.bco = 1; par.bci_thermal = 0; par.bco_thermal = 0
par.forcing = 0; par.forcing_frequency = 1.0; par.forcing_amplitude_cmb = 1.0; par.forcing_amplitude_icb = 0.0
par.rtau = 0; par.itau = 1.0; par.nev = 3; par.which_eigenpairs = 'TM'
par.ncpus = {ncpus}; par.N = 96
g = 0.5   # lmax/N, can be changed below
{extra}
par.lmax = par.ellmax(par.ncpus, max(g, 4.*par.ncpus/par.N), par.m, par.N)   # lmax - m + 1 a multiple of 2 ncpus, lmax >= 3
''')


def build(d, extra=''):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d + '/bin')
    for f in os.listdir(BIN):
        if os.path.isfile(BIN + '/' + f) and f != 'parameters.py':
            shutil.copy(BIN + '/' + f, d + '/bin/' + f)
    open(d + '/bin/parameters.py', 'w').write(T.format(ncpus=NCPUS, extra=extra.replace('; ', '\n')))
    py = sys.executable
    n = str(NCPUS)
    for cmd, log in (([py, 'bin/submatrices.py', n], 'log_sub'),
                     (MPIEXEC.split() + ['-n', n, py, 'bin/assemble.py'], 'log_asm'),
                     (MPIEXEC.split() + ['-n', n, py, 'bin/solve.py'], 'log_solve'),
                     ([py, 'bin/spin_doctor.py', n], 'out2')):
        p = subprocess.run(cmd, cwd=d, stdout=open(d + '/' + log, 'w'), stderr=subprocess.STDOUT)
        if p.returncode:
            sys.exit('%s: %s failed\n%s' % (d, cmd[-2] if cmd[-1] == n else cmd[-1], open(d + '/' + log).read()[-3000:]))
    for f in os.listdir(d):
        if f.endswith('.mtx'):
            os.remove(d + '/' + f)


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
    print(sys.argv[1], 'ok')
