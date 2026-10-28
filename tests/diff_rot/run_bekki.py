#!/usr/bin/env python3
'''
run_bekki.py DIFF_ROT_NPZ [WORKDIR]: the like-for-like solar benchmark of docs/heat-equation.md §8. Builds the four runs
(mk_bekki.py with the Mukhopadhyay et al. 2025 Ω file, nmax 12, thermal_wind 2 without u_r ∂S_tw/∂r, N 64, lmax m + 63)
under WORKDIR (default ./bekki_runs), solves each near the published mode (solve_bekki.py) and prints kore against
Mukhopadhyay et al. 2025 (their compressible values, frame 456 nHz). The verdict compares kore with the values recorded
in heat-equation.md §8 (tolerance 2e-3 nHz), so it is a regression check; the differences from the paper are the
physics comparison. One rank, scipy only; about a minute per run and < 1 GB.

DIFF_ROT_NPZ is data/diff_rot.npz of the Mukhopadhyay et al. 2025 data set (doi:10.17617/3.NP26AI).
'''
import os, sys, subprocess
import numpy as np

if len(sys.argv) < 2:
    sys.exit(__doc__)
H = os.path.dirname(os.path.abspath(__file__))
npz = os.path.abspath(sys.argv[1])
work = sys.argv[2] if len(sys.argv) > 2 else 'bekki_runs'
os.makedirs(work, exist_ok=True)

# (m, symm, mode, Mukhopadhyay et al. 2025 f, growth (nHz), kore in heat-equation.md §8 f, growth)
CASES = [
    (1,  -1, 'high-latitude', -95.70,  +11.69, -95.759,  +11.301),
    (3,  -1, 'Rossby n0',     -262.07,  -0.88, -262.052,  -0.877),
    (3,  -1, 'Rossby n1',     -182.05,  -9.67, -181.934,  -9.700),
    (3,  +1, 'columnar',       210.84, -11.09,  210.908, -11.090),
    (10, +1, 'HFR',           -211.28, -20.00, -211.088, -19.963),
]
TOL = 2e-3

built = set()
rows, ok = [], True
for m, symm, name, pf, pg, kf, kg in CASES:
    run = os.path.join(work, 'm%d%s' % (m, 'a' if symm < 0 else 's'))
    if run not in built:
        subprocess.run([sys.executable, H + '/mk_bekki.py', run, 'm=%d' % m, 'symm=%d' % symm, 'file=' + npz], check=True)
        built.add(run)
    subprocess.run([sys.executable, H + '/solve_bekki.py', run, str(pf), str(pg), '8'], check=True,
                   stdout=open(run + '/solve_%s.txt' % name.replace(' ', '_'), 'w'))
    z = np.load(run + '/eig.npz')
    j = np.argmin(np.abs((z['f'] - pf) + 1j*(z['g'] - pg)))
    f, g = z['f'][j], z['g'][j]
    good = abs(f - kf) < TOL and abs(g - kg) < TOL
    ok &= good
    rows.append('m %2d symm %+d %-13s  kore %9.3f %+8.3f   paper %8.2f %+7.2f   kore - paper %+6.2f %+6.2f   res %.0e  %s'
                % (m, symm, name, f, g, pf, pg, f - pf, g - pg, z['res'][j], 'ok' if good else 'FAIL (§8: %.3f %+.3f)' % (kf, kg)))
print('\n'.join(rows))
print('run_bekki: %s' % ('ok' if ok else 'FAIL'))
