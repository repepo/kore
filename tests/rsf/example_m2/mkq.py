#!/usr/bin/env python3
'''
mkq.py res RESFILE [CHUNK] | nan SCANDIR: queue chunks for runq.sh in q/todo (w values, kore units, one per line).
  res: the on-resonance frequencies of RESFILE (mkfreqs.py res), CHUNK (default 2) per file r000, r001, ... in this order: those
       within 0.5 nHz of the Rossby n = 0 m 2 mode (-331.932 nHz), nearest first, then the others spread over their range
       (bit-reversed order of F), so that a partial run already samples the whole radiative-zone cluster.
  nan: each frequency of SCANDIR/rsf_ke.dat whose KE is not finite (a failed factorisation) as its own file a_<SCANDIR>_<row>,
       unless that chunk was queued before (q/todo, q/taken or q/done); a files run before the r files. Safe to repeat.
'''
import os, sys, numpy as np
FR, FRo = 456.0, -331.932
for q in ('todo', 'taken', 'done'):
    os.makedirs('q/' + q, exist_ok=True)
def vdc(i):   # van der Corput radical inverse of i in base 2
    x, f = 0.0, 0.5
    while i:
        x, i, f = x + f*(i & 1), i >> 1, f/2
    return x
if sys.argv[1] == 'res':
    w = np.atleast_1d(np.loadtxt(sys.argv[2])); C = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    F = -w*FR; d = np.abs(F - FRo)
    near = np.where(d < 0.5)[0]; near = near[np.argsort(d[near], kind='stable')]
    rest = np.where(d >= 0.5)[0]; rest = rest[np.argsort(F[rest], kind='stable')]
    rest = rest[np.argsort([vdc(i) for i in range(rest.size)], kind='stable')]
    w = w[np.concatenate([near, rest]).astype(int)]
    for i in range(0, w.size, C):
        np.savetxt('q/todo/r%03d' % (i//C), w[i:i + C], fmt='%.12e')
    print('res: %d frequencies (%d near the Rossby mode) -> %d chunks' % (w.size, near.size, (w.size + C - 1)//C))
else:
    s = sys.argv[2].rstrip('/'); b = os.path.basename(s)
    d = np.loadtxt(s + '/rsf_ke.dat', ndmin=2)
    if d.size == 0:
        sys.exit('nan %s: no rows in rsf_ke.dat' % s)
    new = 0
    for i in np.where(~np.all(np.isfinite(d[:, 1:]), axis=1))[0]:
        c = 'a_%s_%02d' % (b, i)
        if not any(os.path.exists('q/%s/%s' % (q, c)) for q in ('todo', 'taken', 'done')):
            np.savetxt('q/todo/' + c, d[i:i + 1, 0], fmt='%.12e'); new += 1
    print('nan %s: %d of %d frequencies not finite, %d newly queued' % (s, int((~np.all(np.isfinite(d[:, 1:]), axis=1)).sum()), len(d), new))
