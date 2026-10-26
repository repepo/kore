#!/usr/bin/env python3
'''
The test GitHub runs on every push to anelastic_log (.github/workflows/anelastic_log.yml). To run it locally:
> cd tests/ci; pytest -v

The smooth Sun example (examples/smooth_sun: full sphere, Model S structure, radiative interior, ν and κ step,
stress-free wall) at low resolution, with m = 1 and, in place of the observed rotation, a rigid one 10 % faster than the
frame (diff_rot_type 'rigid'). Its spin-over mode, a rigid tilt of the rotation axis, is then exact: the tilted rotation
is steady in the inertial frame, so in the frame it turns at the frame rate, λ = i (Gaspard = 1), whatever the
structure, stratification, diffusivities and rotation rate. The test runs the pipeline (submatrices, assemble, solve,
spin_doctor) and checks that λ: a 1 % error in the viscous force or in the rotation terms moves it by 1e-6 to 1e-3.
'''
import shutil
import subprocess
from pathlib import Path
import numpy as np

KORE = Path(__file__).resolve().parents[2]
NCPUS = 2

CHANGES = '''
# tests/ci/test_spinover.py: the spin-over, small and on %d ranks
par.m             = 1
par.symm          = -1          # the spin-over: the l = 1 toroidal flow of a rigid tilt
par.diff_rot_type = 'rigid'     # ΔΩ = diff_rot_ampl everywhere
par.diff_rot_ampl = 0.1
par.ncpus         = %d
par.N             = 32
par.lmax          = 16
par.rtau          = 0.0
par.itau          = 1 + 1e-7    # next to λ = i, not on it (A - τB would be singular)
par.nev           = 4
par.petsc_opts['st_mat_mumps_icntl_14'] = 100   # extra MUMPS workspace in per cent (INFOG(1)=-9 without it)
''' % (NCPUS, NCPUS)


def test_spinover(tmp_path):
    shutil.copytree(KORE / 'bin', tmp_path / 'bin', ignore=shutil.ignore_patterns('__pycache__'))
    example = (KORE / 'examples' / 'smooth_sun' / 'parameters.py').read_text(encoding='utf-8')
    (tmp_path / 'bin' / 'parameters.py').write_text(example + CHANGES, encoding='utf-8')

    for cmd in ('./bin/submatrices.py %d', 'mpiexec -n %d ./bin/assemble.py', 'mpiexec -n %d ./bin/solve.py',
                './bin/spin_doctor.py %d'):
        subprocess.run(cmd % NCPUS, shell=True, check=True, cwd=tmp_path)

    e = np.loadtxt(tmp_path / 'eigenvalues0.dat').reshape(-1, 2)
    eig = e[:, 0] + 1j*e[:, 1]
    lam = eig[np.argmin(abs(eig - 1j))]
    assert abs(lam - 1j) < 1e-10, 'spin-over λ = %r, expected 1j' % lam
