#!/usr/bin/env python3
'''
mkrhs.py RUN K [SEED]: in a kore run directory RUN built for forcing = 11 (bin/, B_forced.npz from assemble.py with rsf_nsamp K,
rsf_theta 'nu', no rsf_rcut), builds the right-hand sides of six forcing configurations with bin/reynolds_stress.py and writes
them side by side into RUN/B_forced.npz (K columns each, the original kept as B_forced_base.npz), so that one LU per
frequency serves all six (A and B do not depend on the forcing). Configurations: rsf_theta 'nu' and 'mlt' (rsf_lum 7.0e-7),
each without a cut, with the edge full above 0.74 R_sun and full above 0.76 R_sun (rsf_rcut_width 0.08, aux0 = r_out/R_sun).
Writes RUN/groups.txt (first column, last column + 1, theta, rcut, width, label) and checks that the 'nu' uncut group equals
B_forced_base.npz bitwise.
'''
import os, sys, io, contextlib
import numpy as np
import scipy.sparse as ss
run = os.path.abspath(sys.argv[1]); K = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.chdir(run); sys.path.insert(0, run + '/bin')
from parameters import par
import reynolds_stress as rs
assert par.forcing == 11 and par.rsf_nsamp == K and par.rsf_seed == seed and par.rsf_theta == 'nu' and par.rsf_rcut is None
W = 0.08
cuts = [(None, 'no cut'), (0.74/par.aux0 - W, 'full above 0.74'), (0.76/par.aux0 - W, 'full above 0.76')]
cfgs = [(th, rc, lab) for th in ('nu', 'mlt') for rc, lab in cuts]
cols, lines, j0 = [], [], 0
for th, rc, lab in cfgs:
    par.rsf_theta, par.rsf_rcut, par.rsf_rcut_width = th, rc, W
    par.rsf_lum = 7.0e-7 if th == 'mlt' else par.rsf_lum
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        B = rs.rhs(K, seed, verbose=True)
    tail = [l for l in f.getvalue().splitlines() if 'tail' in l]
    print('%-4s %-16s rcut %-8s %s' % (th, lab, '%.4f' % rc if rc is not None else '-', tail[0].strip() if tail else ''))
    cols.append(B.tocsc())
    lines.append('%d %d %s %s %g %s' % (j0, j0 + K, th, rc, W, lab.replace(' ', '_')))
    j0 += K
base = np.load('B_forced.npz')
if not os.path.exists('B_forced_base.npz'):
    os.rename('B_forced.npz', 'B_forced_base.npz')
else:
    base = np.load('B_forced_base.npz')
b0 = ss.csr_matrix((base['data'], base['indices'], base['indptr']), shape=tuple(base['shape']))
g0 = cols[0].tocsr(); g0.sort_indices(); b0.sort_indices()
same = b0.shape == g0.shape and abs(b0 - g0).max() == 0
print('nu, no cut vs assemble.py B_forced:', 'bitwise equal' if same else 'DIFFERENT (max %.1e)' % abs(b0 - g0).max())
Ball = ss.hstack(cols).tocsr()
np.savez('B_forced.npz', data=Ball.data, indices=Ball.indices, indptr=Ball.indptr, shape=Ball.shape)
open('groups.txt', 'w').write('\n'.join(lines) + '\n')
print('B_forced.npz:', Ball.shape, '->', len(cfgs), 'groups of', K)
sys.exit(0 if same else 1)
