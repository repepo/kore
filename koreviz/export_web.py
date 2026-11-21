#!/usr/bin/env python3
'''
NEW (2026-11-21): writes one kore solution as a small JSON file for the web viewer (docs/viewer/index.html, published with
the docs), where it can be rotated, zoomed, cut and animated in time.

kore solves one azimuthal wavenumber m at a time, so every component of a field is f(r, θ, φ, t) = Re[F(r, θ) e^(i(mφ + ωt))]
(times e^(σt)) with a complex amplitude F on the meridional half-plane. This script writes F of each component on an (r, θ)
grid; the viewer rebuilds any φ, any phase ωt and the cylindrical components u_s, u_z from it. The files are small
(about 1 MB per vector field at lmax ~ 120) and nothing is lost in φ.

Conventions as in koreviz/kmode.py: Schmidt semi-normalised Y_l^m, the real field Re[Σ c_l Y_l^m] for m = 0 and 2 Re[...]
for m > 0, a phase advance p multiplies the solution by e^(ip) (an eigenvalue λ = σ + iω gives p = ωt). Velocity and
induced field from the poloidal and toroidal scalars as in docs/numerical-method.md,
    u = (1/ρ0) ∇×∇×(ρ0 P r) + ∇×(T r):  u_r = L P/r Y,  u_h = (P' + P/r + P dlnρ0/dr) r∇Y + T ∇Y×r,
the same for b with ρ0 = 1; with a conducting inner core ('conducting, Chebys') its field is added below ricb.

Run it in a run folder (with bin/ and the solution files), e.g.
    python3 koreviz/export_web.py                      # solution 0, every field present, kore_solution.json
    python3 koreviz/export_web.py --solnum 2 --fields u,b --out to1.json
then open the viewer and load the file (button or drag and drop), or give its URL as viewer/?data=<url>.
'''
import argparse
import base64
import json
import os
import sys

import numpy as np
import numpy.polynomial.chebyshev as ch


def schmidt_legendre(m, lmax, theta):
    '''
    Schmidt semi-normalised associated Legendre functions without the Condon-Shortley phase, rows l = m..lmax, at the
    colatitudes theta, with their θ-derivatives: P[l-m], dP[l-m] of shape (lmax-m+1, ntheta). Stable recursion of the
    fully normalised functions (no factorials), then the factor sqrt(4π(2 - δ_m0)/(2l+1)).
    '''
    x, sx = np.cos(theta), np.sin(theta)
    nl = lmax - m + 1
    pbar = np.zeros((nl + 1, theta.size))   # one degree more, for the derivative
    pmm = np.full(theta.size, np.sqrt(1/(4*np.pi)))
    for k in range(1, m + 1):
        pmm = pmm*np.sqrt((2*k + 1)/(2*k))*sx
    pbar[0] = pmm
    if nl >= 1:
        pbar[1] = x*np.sqrt(2*m + 3)*pmm
    for i in range(2, nl + 1):
        l = m + i
        a = np.sqrt((4*l*l - 1)/(l*l - m*m))
        b = np.sqrt(((l - 1)**2 - m*m)/(4*(l - 1)**2 - 1))
        pbar[i] = a*(x*pbar[i - 1] - b*pbar[i - 2])
    ls = np.arange(m, lmax + 1)
    # sinθ dP_l/dθ = l cosθ P_l - sqrt((l²-m²)(2l+1)/(2l-1)) P_(l-1), fully normalised
    dpbar = np.zeros((nl, theta.size))
    for i, l in enumerate(ls):
        prev = pbar[i - 1] if i > 0 else 0.0
        c = np.sqrt((l*l - m*m)*(2*l + 1)/(2*l - 1)) if l > m else 0.0
        dpbar[i] = (l*x*pbar[i] - c*prev)/sx
    fac = np.sqrt(4*np.pi*(2 - (m == 0))/(2*ls + 1))[:, None]
    return fac*pbar[:nl], fac*dpbar


def chebyshev_radii(nr, r0, r1):
    '''nr radii from r0 to r1, both included, clustered at the ends (Gauss-Lobatto), ascending'''
    k = np.arange(nr)
    return r0 + (r1 - r0)*0.5*(1 - np.cos(np.pi*k/(nr - 1)))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()).decode('ascii')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solnum', type=int, default=0, help='solution number (column of the solution files), default 0')
    ap.add_argument('--fields', default='all', help="comma list of u, b, t (temperature/entropy), c (composition); "
                                                    "default every field the run solves for")
    ap.add_argument('--nr', type=int, default=None, help='radial points in the fluid (default min(N, 200))')
    ap.add_argument('--ntheta', type=int, default=None, help='colatitudes (default ~1.6 lmax, at least 97, odd)')
    ap.add_argument('--nr_ic', type=int, default=48, help='radial points in a conducting inner core (b), default 48')
    ap.add_argument('--out', default='kore_solution.json', help='output file (default kore_solution.json)')
    ap.add_argument('--title', default=None, help='title shown by the viewer (default the folder name)')
    ap.add_argument('--datadir', default='.', help='the run folder (default .)')
    args = ap.parse_args()

    os.chdir(args.datadir)
    sys.path.insert(0, 'bin')
    from parameters import par
    import utils as ut
    import utils4pp as upp
    import radial_profiles as rap
    import field_io

    m, lmax, ricb, rcmb = par.m, par.lmax, par.ricb, ut.rcmb
    fac = 1.0 if m == 0 else 2.0   # the real field: Re[...] for m = 0, 2 Re[...] for m > 0 (SHTns, as kmode)

    nr = args.nr or min(par.N, 200)
    nth = args.ntheta or max(97, int(1.6*lmax))
    nth += 1 - nth % 2   # odd: the equator is a grid line
    theta = (np.arange(nth) + 0.5)*np.pi/nth
    r = chebyshev_radii(nr, ricb if ricb > 0 else 1e-3*rcmb, rcmb)

    files = {'u': 'flow', 'b': 'magnetic', 't': 'thermal', 'c': 'composition'}
    # the fields this run solves for (parameters.py), so that solution files left by an earlier run in the same folder are
    # never read
    active = {'u': getattr(par, 'hydro', 1), 'b': par.magnetic, 't': par.thermal, 'c': getattr(ut, 'compositional', 0)}
    if args.fields == 'all':
        want = [f for f in 'ubtc' if active[f]]
    else:
        want = [f.strip() for f in args.fields.split(',')]
        for f in want:
            if f in active and not active[f]:
                sys.exit('export_web: this run does not solve for %s (parameters.py)' % files[f])

    Pl = {}   # Schmidt functions per m, cached

    def legendre():
        if 'P' not in Pl:
            Pl['P'], Pl['dP'] = schmidt_legendre(m, lmax + 1, theta)
        return Pl['P'], Pl['dP']

    def rows(ls):
        P, dP = legendre()
        return P[np.asarray(ls) - m], dP[np.asarray(ls) - m]

    def vector(Plj, Tlj, lp, lt, rr, dlnrho, eval_tab):
        '''the complex amplitudes (r, θ, φ components) on rr × theta from the poloidal/toroidal radial tables'''
        p0, p1 = eval_tab(Plj, rr)
        t0 = eval_tab(Tlj, rr)[0]
        L = (lp*(lp + 1.0))[:, None]
        Q = L*p0/rr
        S = p1 + p0*(1/rr + dlnrho)
        Yp, dYp = rows(lp)
        Yt, dYt = rows(lt)
        im_sin = 1j*m/np.sin(theta)
        Fr = Q.T @ Yp
        Ft = S.T @ dYp + (t0.T @ Yt)*im_sin
        Fp = (S.T @ Yp)*im_sin - t0.T @ dYt
        return [fac*Fr, fac*Ft, fac*Fp]

    def fluid_tab(C, rr):
        tab = upp.cheb2space_table(np.atleast_2d(C), 1, rr)
        return tab[0], tab[1]

    out_fields = []
    for f in want:
        name = files.get(f)
        if name is None:
            sys.exit('export_web: unknown field %r (u, b, t, c)' % f)
        if not ut.field_exists(name):
            sys.exit('export_web: no solution files of %s here' % name)
        nsol = field_io.count(name)
        if args.solnum >= nsol:
            sys.exit('export_web: %s has %d solutions (0..%d), not %d: are the files from this run?'
                     % (name, nsol, nsol - 1, args.solnum))
        sol = np.asarray(ut.load_field(name, args.solnum))
        rr = r.copy()
        if f in ('u', 'b'):
            vs = par.symm if f == 'u' else ut.bsymm
            lp, lt, _ = ut.ell(m, lmax, vs)
            Plj, Tlj = upp.expand_reshape_sol(sol, vs)
            dl = np.zeros_like(rr)
            if f == 'u':
                try:
                    dl = np.asarray(rap.prf.dlog_rho(rr), dtype=float)
                except Exception:
                    pass
            comps = vector(Plj, Tlj, np.asarray(lp), np.asarray(lt), rr, dl, fluid_tab)
            # NEW: a conducting inner core whose field is among the unknowns: Chebyshev series in x = r/ricb, one parity
            # per l (as utils4pp.em_budget)
            if f == 'b' and getattr(ut, 'cic', 0) and ut.field_exists('magnetic_ic'):
                bic = np.asarray(ut.load_field('magnetic_ic', args.solnum))
                Nic = ut.Nic
                nb = int((lmax - m + 1)/2)
                if ut.symmB0 == -1:
                    ixf, ixg = (m + ut.s) % 2, (m + 1 - ut.s) % 2
                else:
                    ixf, ixg = (m + 1 - ut.s) % 2, (m + ut.s) % 2
                ri = chebyshev_radii(args.nr_ic + 1, 0, ricb)[1:-1]   # (0, ricb): the wall itself is in the fluid grid

                def ic_rows(blocks, ix, nd):
                    out = [np.zeros((len(blocks), ri.size), complex) for _ in range(nd + 1)]
                    for k, a in enumerate(blocks):
                        cc = np.zeros(par.N_cic, complex)
                        cc[ix::2] = a
                        for j in range(nd + 1):
                            out[j][k] = ch.chebval(ri/ricb, cc)
                            cc = ch.chebder(cc)/ricb
                    return out

                fb = [bic[k*Nic:(k + 1)*Nic] for k in range(nb)]
                gb = [bic[nb*Nic + k*Nic: nb*Nic + (k + 1)*Nic] for k in range(nb)]
                F0, F1 = ic_rows(fb, ixf, 1)
                G0 = ic_rows(gb, ixg, 0)[0]
                tabs = {id(F0): (F0, F1), id(G0): (G0,)}
                ic = vector(F0, G0, np.asarray(lp), np.asarray(lt), ri, 0*ri,
                            lambda C, x: tabs[id(C)])
                rr = np.r_[ri, rr]
                comps = [np.r_[a, b] for a, b in zip(ic, comps)]
            out_fields.append({'key': f, 'label': {'u': 'velocity u', 'b': 'magnetic field b'}[f], 'vector': True,
                               'r': b64(rr), 'nr': int(rr.size),
                               'comps': {c: {'re': b64(a.real), 'im': b64(a.imag)} for c, a in zip('rtp', comps)}})
        else:
            lp = np.asarray(ut.ell(m, lmax, par.symm)[0])
            Plj = upp.expand_reshape_sol(sol, par.symm)
            p0 = fluid_tab(Plj, rr)[0]
            Y = rows(lp)[0]
            F = fac*(p0.T @ Y)
            out_fields.append({'key': f, 'label': {'t': 'temperature/entropy', 'c': 'composition'}[f], 'vector': False,
                               'r': b64(rr), 'nr': int(rr.size), 'comps': {'s': {'re': b64(F.real), 'im': b64(F.imag)}}})
        print('export_web: %s, %d x %d' % (name, rr.size, nth))

    if not out_fields:
        sys.exit('export_web: no solution files found')

    eig = None
    if os.path.isfile('eigenvalues0.dat') and par.forcing == 0:
        e = np.loadtxt('eigenvalues0.dat').reshape((-1, 2))   # column 0 real part, column 1 imaginary part
        if args.solnum < e.shape[0]:
            eig = [float(e[args.solnum, 0]), float(e[args.solnum, 1])]

    meta = {'m': int(m), 'lmax': int(lmax), 'N': int(par.N), 'ricb': float(ricb), 'rcmb': float(rcmb),
            'symm': int(par.symm), 'solnum': args.solnum, 'eigenvalue': eig}
    if par.forcing != 0:
        meta['forcing_frequency'] = float(getattr(par, 'forcing_frequency', 0.0))
    for k in ('timescale', 'Ek') + (('Le', 'Em', 'B0', 'innercore') if par.magnetic else ()):
        v = getattr(par, k, None)
        if isinstance(v, (int, float, str)) and not isinstance(v, bool) and v != 0:
            meta[k] = v
    doc = {'format': 'kore-web-1', 'title': args.title or os.path.basename(os.getcwd()), 'meta': meta,
           'theta': b64(theta), 'ntheta': int(nth), 'fields': out_fields}
    with open(args.out, 'w') as fh:
        json.dump(doc, fh, separators=(',', ':'))
    print('export_web: wrote %s (%.0f kB)' % (args.out, os.path.getsize(args.out)/1e3))


if __name__ == '__main__':
    main()
