#!/usr/bin/env python3
'''
usage, in a run folder: python3 plot_check.py KIND [ARG]

Runs one of kore's plotting tools on solution 0 of the run in the current folder, with matplotlib's Agg backend, without
LaTeX, and saving the figure (plot.png) instead of showing it. The last line printed is PLOTCHECK and a JSON dictionary of
what tools_check.py checks:

load_solution u|b  tools/load_solution.py: 'diff', the largest difference of the coefficients it unpacks with those of
                   utils4pp.expand_sol; with 'b' and a conducting inner core, 'jump', the largest jump of f, df/dr and g
                   at the inner boundary between the fluid's and the core's expansions (relative)
profile            tools/plot_rad_con_tor_profile.py from wall to wall: 'rel', its kinetic energy relative to KE of flow.dat,
                   minus 1
field X            tools/plot_field.py X on 30 radii and 24 colatitudes: 'finite', whether the three components are
                   finite; with 'b', 'diff', the largest difference of the components q, s, t of each degree with those
                   from the Chebyshev coefficients of the field (relative)
'''
import json, os, runpy, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as ch

TOOLS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
_rc = plt.rc
plt.rc = lambda group, **kw: None if group == 'text' else _rc(group, **kw)   # no LaTeX
plt.show = lambda *a, **k: plt.savefig('plot.png', dpi=50)


def tool(name, argv):
    '''the globals of tools/name run with the arguments argv'''
    sys.argv = [name] + argv
    return runpy.run_path(os.path.join(TOOLS, name), run_name='__main__')


def cheb(C, x, nd=0, scale=1.):
    '''the nd-th derivative (x-derivatives times scale**nd) of each row of Chebyshev coefficients C at the points x'''
    return np.array([ ch.chebval(x, ch.chebder(c, nd) if nd else c) for c in C ]) * scale**nd


kind, arg = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
out = {}
if kind == 'load_solution':
    g = tool('load_solution.py', ['0', arg])
    ut, par = g['ut'], g['par']
    sys.path.insert(1, 'bin/')
    import utils4pp as upp
    vs = par.symm * (ut.symmB0 if arg == 'b' else 1)
    P, T = np.split(upp.expand_sol(ut.load_field('magnetic' if arg == 'b' else 'flow', 0), vs), 2)
    out['diff'] = float(max(np.abs(g['Plj'].ravel() - P).max(), np.abs(g['Tlj'].ravel() - T).max()))
    if arg == 'b' and ut.cic:
        Plj, Tlj, Pic, Tic = g['Plj'], g['Tlj'], g['Plj_ic'], g['Tlj_ic']
        pairs = [ (cheb(Plj, -1.), cheb(Pic, 1.)), (cheb(Tlj, -1.), cheb(Tic, 1.)),
                  (cheb(Plj, -1., 1, 2/g['gap']), cheb(Pic, 1., 1, 1/par.ricb)) ]
        out['jump'] = float(max(np.abs(a - b).max()/np.abs(a).max() for a, b in pairs))
elif kind == 'profile':
    sys.path.insert(1, 'bin/')
    from parameters import par
    import utils as ut
    g = tool('plot_rad_con_tor_profile.py', ['0', '200', repr(par.ricb), repr(ut.rcmb)])
    out['rel'] = float(0.5*(g['R2'] - g['R1'])*np.sum(g['wk']*g['ke'])/g['K'][0] - 1)
elif kind == 'field':
    g = tool('plot_field.py', ['30', '24', '1', '179', arg, 'abs', '0'])
    out['finite'] = bool(all(np.isfinite(g[k]).all() for k in ('ur', 'utheta', 'uphi')))
    if arg == 'b':
        ut, par = g['ut'], g['par']
        P, T = g['bsol']
        r = g['r']
        x = ut.xcheb(r, par.ricb, ut.rcmb)
        dx = 2/(ut.rcmb - par.ricb) if par.ricb > 0 else 1/ut.rcmb
        lp = ut.ell(par.m, par.lmax, ut.bsymm)[0]
        L = (lp*(lp + 1.))[:, None]
        f, f1 = cheb(P, x), cheb(P, x, 1, dx)
        pairs = [ (g['Qlr'], L*f/r), (g['Slr'], f1 + f/r), (g['Tlr'], cheb(T, x)) ]
        out['diff'] = float(max(np.abs(a - b).max()/np.abs(b).max() for a, b in pairs))
else:
    sys.exit(__doc__)
print('PLOTCHECK ' + json.dumps(out))
