#!/usr/bin/env python3
'''
Checks of the operators that GitHub runs on every push to main (.github/workflows/anelastic_log.yml): no solve, only
submatrices.py, so they take seconds. To run them locally:
> cd tests/ci; pytest -v test_operators.py

- bin/ru_insane.py on two runs built as the suites build theirs: the anelastic 'Wu 2005' shell with the entropy and a
  conical differential rotation with its thermal wind (tests/heat_dr/mk_run.py), and the same structure with B0 'axial'
  between insulators (tests/magnetic/mk_run.py). It checks the inertia, Coriolis, viscous, buoyancy, entropy and
  diffusion blocks under the energy and entropy pairings, the differential-rotation and thermal-wind blocks against an
  (r, θ) grid, the magnetic blocks and the Lorentz/induction energy identity.
- tests/diff_rot/gen_check.py on the first run: the couplings of bin/diff_rot.py against their closed forms for P0 and
  P2, the Δl ranges kore assembles for n up to 16, and the thermal-wind σ2, σ4, σ6.
'''
import subprocess
import sys
from pathlib import Path
import pytest

TESTS = Path(__file__).resolve().parents[1]

RUNS = {   # name: (builder, its extra parameters)
    'heat_diff_rot': (TESTS / 'heat_dr' / 'mk_run.py', 'par.N = 96'),
    'magnetic':      (TESTS / 'magnetic' / 'mk_run.py', 'par.ncpus = 1'),
}


def build(tmp_path, name):
    mk, extra = RUNS[name]
    # (heat_dr's builder also runs assemble.py, which ru_insane does not need)
    subprocess.run([sys.executable, str(mk), name, extra], check=True, cwd=tmp_path)
    return tmp_path / name


@pytest.mark.parametrize('name', RUNS)
def test_ru_insane(tmp_path, name):
    d = build(tmp_path, name)
    subprocess.run([sys.executable, 'bin/ru_insane.py'], check=True, cwd=d)


def test_diff_rot_couplings(tmp_path):
    d = build(tmp_path, 'heat_diff_rot')
    subprocess.run([sys.executable, str(TESTS / 'diff_rot' / 'gen_check.py')], check=True, cwd=d)
