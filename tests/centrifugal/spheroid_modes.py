#!/usr/bin/env python3
'''
spheroid_modes.py: the inertial modes of an inviscid, homogeneous fluid in a rotating spheroid, computed exactly and
independently of kore, against kore's centrifugal = 1 (NEW 2026-11-06, phase 2).

The exact modes: the polynomial method (Lebovitz 1989, Vantieghem 2014). In an ellipsoid the velocity fields of degree
at most n that are solenoidal and tangent to the wall are spanned by ∇(F x^i y^j z^k) × ê_d (i + j + k <= n - 1, d = x, y,
z), F = 1 - (x² + y²)/a² - z²/c² (their span has the dimension n(n + 1)(2n + 7)/6, checked), and the Coriolis force,
projected on such fields, maps that space into itself. So the Galerkin eigenvalues of λ <v, u> = -<v, 2ẑ × u> on it are
exact. All of it is in the physical spheroid, in Cartesian components: no map and no spherical harmonics. The integrals
are exact moments of the unit ball (in ξ = (x/a, y/a, z/c)); the modes are sorted by m and by their equatorial symmetry
with the rotation and the mirror that commute with the problem.

kore's body with centrifugal_shape 'uniform' is the spheroid a = 1 + f/3, c = 1 - 2f/3 (the map is linear), so at first
order in f its eigenvalues must move as the exact ones. The checks, each printed as one verdict line:

  shifts    dω/df of every mode of degree <= 7 with m = 0, 1, 2 and both symmetries (73 modes), from kore's runs at
            f = ±δ, ±2δ (δ = 1e-4, Richardson's extrapolation removes the δ² error), against the exact modes. The exact
            dω/df is computed two ways: by the same differences of the exact spheroid modes, and from the sphere's
            mode alone, 2 ω ∫|u_z|²/∫|u|² (in the reference sphere the spheroid is the sphere with the inertia
            diag(1, 1, c²/a²)); the two agree to 1e-9
  spin-over the exact solver at f = 0 gives the sphere's modes (the spin-over ω = 1 and the mode 2/(m + 1) of m = 2) and,
            with --ekman, Greenspan's Ekman damping of the spin-over, λ1 = -2.6205 + 0.2585i (see below)

--ekman also prints, for a few modes, the leading Ekman damping λ = iω + Ek^½ λ1 of a no-slip wall (Greenspan 1968:
λ1 = -∮ Σ± |e±*·u|² (i(ω ∓ 2 n̂·ẑ))^½ dS / ∫|u|² dV, e± = (t̂1 ± i t̂2)/√2) and its derivative in f. It is only a reference:
kore approaches λ1 as Ek^0.3 (at f = 0, 1 % at Ek 1e-6), and dλ1/df only as Ek^0.1, because of the critical latitudes
(where ω = ±2 n̂·ẑ, the layer thickens and its contribution to the derivative is of relative size Ek^(1/10)); the damping's
dependence on the shape is checked exactly at finite Ek by shape_check.py.

Usage: python3 spheroid_modes.py [--ekman]   (half a minute; builds its runs with mk_run.py in the current directory)
'''
import os, sys, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def dfact(n):
    r = 1
    while n > 1:
        r *= n; n -= 2
    return r

def monos(D):
    '''the monomials x^i y^j z^k of degree <= D'''
    return [(i, j, d - i - j) for d in range(D + 1) for i in range(d, -1, -1) for j in range(d - i, -1, -1)]

class Space:
    '''the solenoidal tangent polynomial fields of degree <= n in the spheroid (a, a, c), as coefficients on monos(n) of
    their three Cartesian components, functions of ξ = (x/a, y/a, z/c)'''
    def __init__(s, n):
        s.n = n
        s.mon = monos(n); s.idx = {e: q for q, e in enumerate(s.mon)}
        s.mon2 = monos(n + 1); s.idx2 = {e: q for q, e in enumerate(s.mon2)}
        Nm = len(s.mon)
        W = np.zeros((Nm, Nm))                     # ∫ over the unit ball of ξ^α ξ^β
        for p, a in enumerate(s.mon):
            for q, b in enumerate(s.mon):
                i, j, k = a[0] + b[0], a[1] + b[1], a[2] + b[2]
                if i % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                    W[p, q] = 4*np.pi*dfact(i - 1)*dfact(j - 1)*dfact(k - 1)/dfact(i + j + k + 3)
        s.W, s.W3 = W, np.kron(np.eye(3), W)

    def basis(s, a, c):
        '''∇(F ξ^α) × ê_d, |α| <= n - 1, F = 1 - |ξ|² (the physical gradient: ∂x = ∂ξ/a, ∂z = ∂ζ/c)'''
        cols = []
        for al in monos(s.n - 1):
            pot = np.zeros(len(s.mon2))
            pot[s.idx2[al]] += 1
            for d in range(3):
                e = list(al); e[d] += 2
                pot[s.idx2[tuple(e)]] -= 1
            g = np.zeros((3, len(s.mon)))
            for q, e in enumerate(s.mon2):
                for d, sc in enumerate((1/a, 1/a, 1/c)):
                    if pot[q] != 0 and e[d] > 0:
                        e2 = list(e); e2[d] -= 1
                        g[d, s.idx[tuple(e2)]] += sc*e[d]*pot[q]
            for d in range(3):
                v = np.zeros((3, len(s.mon)))
                if d == 0: v[1], v[2] = g[2], -g[1]
                if d == 1: v[0], v[2] = -g[2], g[0]
                if d == 2: v[0], v[1] = g[1], -g[0]
                cols.append(v.ravel())
        return np.array(cols).T

    def cross_z(s):
        Nm = len(s.mon); M = np.zeros((3*Nm, 3*Nm))
        M[0:Nm, Nm:2*Nm] = -np.eye(Nm); M[Nm:2*Nm, 0:Nm] = np.eye(Nm)
        return M

    def rotation(s):
        '''the generator of the rotations about z, ẑ × u - (x ∂y - y ∂x) u: -i m on fields e^{imφ}'''
        Nm = len(s.mon); A = np.zeros((Nm, Nm))
        for q, (i, j, k) in enumerate(s.mon):
            if j > 0: A[s.idx[(i + 1, j - 1, k)], q] += j
            if i > 0: A[s.idx[(i - 1, j + 1, k)], q] -= i
        return s.cross_z() - np.kron(np.eye(3), A)

    def mirror(s):
        '''u -> (u_x, u_y, -u_z)(x, y, -z): +1 on equatorially symmetric flows (kore's symm)'''
        return np.kron(np.diag([1, 1, -1]), np.diag([(-1)**e[2] for e in s.mon]))

    def modes(s, a, c, tol=1e-11):
        '''[(m, symm, λ, coefficients)], the coefficients normalised to ∫ |u|² dξ = 1 over the unit ball'''
        C = s.basis(a, c)
        G = C.T @ s.W3 @ C
        ev, V = np.linalg.eigh(G)
        keep = ev > tol*ev.max()
        s.dim = int(keep.sum())
        Bq = C @ (V[:, keep]/np.sqrt(ev[keep]))
        K = Bq.T @ s.W3 @ (-2*s.cross_z()) @ Bq
        Z = Bq.T @ s.W3 @ s.rotation() @ Bq
        P = Bq.T @ s.W3 @ s.mirror() @ Bq
        zl, zv = np.linalg.eig(Z)
        mm = np.rint((1j*zl).real).astype(int)
        out = []
        for m in sorted(set(mm)):
            Y, _ = np.linalg.qr(zv[:, mm == m])
            Pm = Y.conj().T @ P @ Y
            pl, pv = np.linalg.eigh((Pm + Pm.conj().T)/2)
            for sy in (1, -1):
                sel = abs(pl - sy) < 1e-6
                if sel.any():
                    Yp = Y @ pv[:, sel]
                    lam, X = np.linalg.eig(Yp.conj().T @ K @ Yp)
                    out += [(m, sy, lam[t], Bq @ (Yp @ X[:, t])) for t in range(len(lam))]
        return out

def find(out, m, sy, om):
    return min([t for t in out if t[0] == m and t[1] == sy], key=lambda t: abs(t[2].imag - om))

def evaluate(sp, coef, xi):
    Nm = len(sp.mon)
    P = np.array([xi[0]**e[0]*xi[1]**e[1]*xi[2]**e[2] for e in sp.mon])
    return np.array([coef[d*Nm:(d + 1)*Nm] @ P for d in range(3)])

def ekman(sp, a, c, lam, coef):
    '''Greenspan's λ1 of a no-slip wall for the mode (λ, coef) of the spheroid (a, a, c)'''
    from scipy.integrate import quad
    om = lam.imag
    vol = a*a*c*np.real(coef.conj() @ sp.W3 @ coef)
    def integrand(th, sg, part):
        st, ct = np.sin(th), np.cos(th)
        h = np.sqrt(c*c*st*st + a*a*ct*ct)
        u = evaluate(sp, coef, np.array([[st], [0.], [ct]]))[:, 0]            # φ = 0 (|e±*·u| does not depend on φ)
        n = np.array([c*st, 0., a*ct])/h
        t2 = np.array([0., 1., 0.]); t1 = np.cross(t2, n)
        v = (t1 @ u - sg*1j*(t2 @ u))/np.sqrt(2)
        r = np.sqrt(1j*(om - sg*2*n[2]) + 0j)
        r = r if r.real >= 0 else -r
        return (r.real if part == 'r' else r.imag)*abs(v)**2*2*np.pi*a*st*h
    crit = [np.arctan2(a*np.sqrt(1 - (sg*om/2)**2), c*sg*om/2) for sg in (1, -1) if abs(om/2) < 1]
    pts = sorted(set([0.] + crit + [np.pi]))
    tot = 0j
    for sg in (1, -1):
        for part, unit in (('r', 1), ('i', 1j)):
            tot += unit*sum(quad(integrand, t0, t1, args=(sg, part), epsabs=1e-14, epsrel=1e-13, limit=400)[0]
                            for t0, t1 in zip(pts[:-1], pts[1:]))
    return -tot/vol

def kore_eigs(d, sigma, k=4):
    import scipy.sparse as ss, scipy.sparse.linalg as ssl
    def load(M):
        z = np.load(os.path.join(d, M + '.npz'))
        return ss.csr_matrix((z['data'], z['indices'], z['indptr']), shape=z['shape']).astype(complex)
    A, B = load('A'), load('B')
    lu = ssl.splu((A - sigma*B).tocsc())
    op = ssl.LinearOperator(A.shape, matvec=lambda x: lu.solve(B @ x), dtype=complex)
    nu = ssl.eigs(op, k=k, which='LM', tol=1e-14, ncv=40, return_eigenvectors=False)
    lam = sigma + 1/nu
    return lam[np.argmin(abs(lam - sigma))]

def build(name, extra):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'mk_run.py'), name, extra], capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:]); sys.exit('spheroid_modes.py: %s could not be built' % name)

if __name__ == '__main__':
    fail = []
    def verdict(name, ok, msg):
        print('%-10s %s  %s' % (name + ':', 'ok  ' if ok else 'FAIL', msg))
        if not ok:
            fail.append(name)
    n, dl = 7, 1e-4
    sp = Space(n)
    o0 = sp.modes(1., 1.)
    so, m2 = find(o0, 1, -1, 1.)[2], find(o0, 2, -1, 2/3)[2]
    verdict('exact', sp.dim == n*(n + 1)*(2*n + 7)//6 and abs(so - 1j) < 1e-13 and abs(m2 - 2j/3) < 1e-13,
            'degree %d: %d fields (n(n + 1)(2n + 7)/6), spin-over %.15f, m = 2 sectoral %.15f (2/3)' % (n, sp.dim, so.imag, m2.imag))
    # the exact shifts: the sphere formula, and the differences of the exact spheroid modes at f = ±δ, ±2δ
    rich = lambda l, d: ((4*(l[1] - l[-1])/(2*d) - (l[2] - l[-2])/(4*d))/3).imag     # Richardson: the error is O(δ⁴)
    dx = 1e-3                                                                         # (the exact modes are 1e-13 apart)
    ox = {k: sp.modes(1 + k*dx/3, 1 - 2*k*dx/3) for k in (1, -1, 2, -2)}
    Nm = len(sp.mon)
    rows, worst_x = [], 0.
    for (m, sy, l0, c0) in o0:
        if m not in (0, 1, 2) or abs(l0.imag) < 1e-8:
            continue
        an = 2*l0.imag*np.real(c0[2*Nm:].conj() @ sp.W @ c0[2*Nm:])/np.real(c0.conj() @ sp.W3 @ c0)
        worst_x = max(worst_x, abs(an - rich({k: find(ox[k], m, sy, l0.imag)[2] for k in ox}, dx)))
        rows.append((m, sy, l0.imag, an))
    # kore: f = ±δ, ±2δ (small steps: the unresolved modes of degree > 7 crowd the spectrum)
    worst, nmodes = 0., 0
    for m in (0, 1, 2):
        for sy in (1, -1):
            names = {}
            for k in (1, -1, 2, -2):
                names[k] = 'sm_m%d_s%+d_%+d' % (m, sy, k)
                build(names[k], 'par.m = %d; par.symm = %d; par.centrifugal_f = %r; par.lmax = %d' % (m, sy, k*dl, 21 - m % 2))
            for (mm, ss_, om0, an) in rows:
                if (mm, ss_) == (m, sy):
                    worst = max(worst, abs(rich({k: kore_eigs(names[k], 1j*(om0 + an*k*dl)) for k in names}, dl) - an))
                    nmodes += 1
    verdict('shifts', nmodes == len(rows) > 70 and worst < 1e-9 and worst_x < 1e-9,
            '%d modes (m 0, 1, 2, degree <= %d): max |dω/df (kore) - exact| = %.1e (the exact one by two routes: %.1e)'
            % (nmodes, n, worst, worst_x))
    if '--ekman' in sys.argv:
        print('Ekman damping of a no-slip wall, λ = iω + Ek^½ λ1 (Greenspan), and dλ1/df:')
        for (m, sy, om) in [(1, -1, 1.0), (1, -1, 0.6119849), (0, -1, 0.8944272), (2, -1, 2/3), (2, 1, 1.2319251)]:
            e = [ekman(sp, a, c, *find(o, m, sy, om)[2:]) for a, c, o in ((1., 1., o0), (1 + dx/3, 1 - 2*dx/3, ox[1]),
                                                                           (1 - dx/3, 1 + 2*dx/3, ox[-1]))]
            print('  m %d symm %+d ω %.7f   λ1 %.6f%+.6fi   dλ1/df %.5f%+.5fi' % (m, sy, om, e[0].real, e[0].imag,
                                                                          ((e[1] - e[2])/(2*dx)).real, ((e[1] - e[2])/(2*dx)).imag))
    print('spheroid_modes.py: %s' % ('FAIL ' + ', '.join(fail) if fail else 'ok'))
    sys.exit(1 if fail else 0)
