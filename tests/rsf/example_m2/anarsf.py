#!/usr/bin/env python3
'''
anarsf.py RESULTS OUTDIR: the forcing-11 example (run_example.sh). Reads RESULTS/s*/rsf_ke.dat (w, KE of each column, KE below
rsf_split of each column; non-finite rows, i.e. failed factorisations, are skipped, and a frequency solved twice is kept once), RESULTS/base/groups.txt (column groups: theta, cut), RESULTS/res.txt.modes (on-resonance modes) and
RESULTS/eig/ana.txt (KE<0.70 of each eigenmode). Writes OUTDIR/spectrum.csv (per frequency and group: mean KE, its standard
error, mean KE below 0.70 R_sun, its standard error; two-sided PSD in kore units) and OUTDIR/modes.csv (per on-resonance mode
and group: PSD at w_n, the mode energy PSD |sigma_n|/2 of an isolated Lorentzian, and the share of the response below 0.70).
'''
import sys, os, glob, numpy as np
R, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
FR = 456.0
groups = []
for l in open(R + '/base/groups.txt'):
    a = l.split(); groups.append((int(a[0]), int(a[1]), a[2], a[3], a[5]))
rows = []
for f in sorted(glob.glob(R + '/s*/rsf_ke.dat')):
    d = np.loadtxt(f, ndmin=2)
    for r in d:
        if np.all(np.isfinite(r)):   # a failed factorisation gives non-finite KE; its frequency is rerun (mkq.py nan)
            rows.append((os.path.basename(os.path.dirname(f)), r))
nc = groups[-1][1]
gw = np.concatenate([np.atleast_1d(np.loadtxt(f)) for f in (R + '/g1.txt', R + '/g2.txt') if os.path.exists(f)] or [np.zeros(0)])
out, seen = [], set()
for src, r in rows:
    w = r[0]; ke = r[1:1 + nc]; kb = r[1 + nc:1 + 2*nc]
    if round(w, 11) in seen:   # a frequency solved twice (a retry): keep the first finite one
        continue
    seen.add(round(w, 11))
    kind = ('grid' if np.any(np.abs(gw - w) < 1e-11) else 'res') if gw.size else ('grid' if src in ('s1', 's2') else 'res')
    line = {'src': src, 'kind': kind, 'w': w, 'F': -w*FR}
    for j0, j1, th, rc, lab in groups:
        k, b = ke[j0:j1], kb[j0:j1]
        n = k.size
        line[(th, lab)] = (k.mean(), k.std(ddof=1)/np.sqrt(n), b.mean(), b.std(ddof=1)/np.sqrt(n), np.mean(b/k))
    out.append(line)
out.sort(key=lambda x: x['F'])
keys = [(g[2], g[4]) for g in groups]
with open(O + '/spectrum.csv', 'w') as f:
    f.write('src,kind,F_nHz,w_kore,' + ','.join('%s|%s|%s' % (th, lab, q) for th, lab in keys for q in ('KE', 'KE_se', 'KEin', 'KEin_se', 'fin')) + '\n')
    for x in out:
        f.write('%s,%s,%.6f,%.12e,' % (x['src'], x['kind'], x['F'], x['w']) + ','.join('%.6e' % v for k in keys for v in x[k]) + '\n')
# on-resonance modes
modes = []
if os.path.exists(R + '/res.txt.modes'):
    M = np.loadtxt(R + '/res.txt.modes', ndmin=2)
    ana = {}
    if os.path.exists(R + '/eig/ana.txt'):
        for l in open(R + '/eig/ana.txt'):
            if l.startswith('#') or not l.strip(): continue
            a = l.split()
            try: ana[round(float(a[1]), 3), round(float(a[2]), 3)] = float(a[5])
            except (ValueError, IndexError): pass
    with open(O + '/modes.csv', 'w') as f:
        f.write('F_nHz,g_nHz,KE<0.70_mode,' + ','.join('%s|%s|%s' % (th, lab, q) for th, lab in keys for q in ('PSD', 'PSD_se', 'E', 'fin')) + '\n')
        for Fn, gn, wn, sn in M:
            x = min(out, key=lambda z: abs(z['w'] - wn))
            if abs(x["w"] - wn) > 1e-9: continue
            kin = ana.get((round(Fn, 3), round(gn, 3)), np.nan)
            vals = []
            for k in keys:
                m, se, b, bse, fin = x[k]
                vals += [m, se, m*abs(sn)/2, fin]
            f.write('%.6f,%.6f,%.4f,' % (Fn, gn, kin) + ','.join('%.6e' % v for v in vals) + '\n')
            modes.append((Fn, gn, kin, vals))
print('%d frequencies, %d groups, %d on-resonance modes -> %s' % (len(out), len(groups), len(modes), O))
