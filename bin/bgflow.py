#!/usr/bin/env python3
'''
bgflow.py: perturbations of an axisymmetric background flow U(r, θ) in the momentum equation (docs/bgflow.md).

The linearised momentum equation gets, on the left with the Coriolis force 2ẑ×u, the term

    F = (U·∇)u + (u·∇)U .

U is a real flow with m = 0, read from par.bgflow_file: an .npz with the Chebyshev coefficients (on [ricb, 1]) of its
poloidal scalars P_n(r) of the degrees nP and its toroidal scalars T_n(r) of the degrees nT, and ricb (and Ro, for a
nonlinear flow, which holds at that bgflow_Ro only). It is multiplied by Ro = par.bgflow_Ro (times OmgTau in the code's
units). A flow of degree l, u or U (with m = 0), is in (r, θ, φ)
components

    u = q (L Y, 0, 0) + s (0, ∂_θY, i Z) + T (0, i Z, -∂_θY) ,    q = P/r ,    s = (rP)'/r ,    Z = m Y/sin θ .

So r F is a sum of products b ρ, b (r ρ') and (r b') ρ of a radial factor b (q, s or T) of U and one ρ of u, each
times an angular function (see advect). With r F = Σ (Fr Y r̂ + Fs r∇Y + Ft r̂×r∇Y), the rows of degree l' of
sections u and v, r⁵ r̂·∇×∇× and r³ r̂·∇× as in operators.py, get

    r⁵ r̂·∇×∇×F = L' (r² Fr - r³ ∂_r Fs) ,    r³ r̂·∇×F = -L' r Ft ,    L' = l'(l'+1) .

Fr, Fs and Ft are projections, by Gauss-Legendre quadrature in cos θ (exact: the integrands are polynomials), and a
factor of U of degree n couples l to the degrees l' with |l - l'| <= n.

Run as a script in the folder of a forcing = 12 problem (m = 0, symm = 1, forcing_frequency = 0) to write the
background file bgflow.npz of its steady flow, with the degrees n <= nmax:
    python3 bin/bgflow.py nmax              the linear flow, from the solution in the folder, per unit Ro
    python3 bin/bgflow.py nmax nonlinear    the nonlinear flow at Ro = forcing_amplitude_icb, by Newton's method
                                            (see newton; needs submatrices.py only, and bgflow_Ro = Ro when used)
'''
import numpy as np
import numpy.polynomial.chebyshev as ch
import numpy.polynomial.legendre as leg
import scipy.sparse as ss
from parameters import par
import utils as ut

_blocks = None  # set by setup() on the first call of terms()



def background():
    '''[(n, name, b, r b')]: the radial factors b of U, 'q' and 's' of each P_n and 't' of each T_n, times Ro OmgTau,
    as Chebyshev series in r'''
    bg  = np.load(par.bgflow_file)
    assert abs(bg['ricb'] - par.ricb) < 1e-12, 'bgflow_file is for ricb = %g' % bg['ricb']
    assert 'Ro' not in bg.files or abs(bg['Ro'] - par.bgflow_Ro) < 1e-9, \
        'bgflow_file is the nonlinear flow at bgflow_Ro = %g' % bg['Ro']
    dom = [par.ricb, ut.rcmb]
    r   = ch.Chebyshev.identity(domain=dom)
    Ro  = par.bgflow_Ro*par.OmgTau
    out = []
    for n, c in zip(bg['nP'], bg['P']):
        P = Ro*ch.Chebyshev(c, domain=dom)
        q = ch.Chebyshev.interpolate(lambda r: P(r)/r, P.degree() + 30, domain=dom)
        s = q + P.deriv()
        out += [ (n, 'q', q, r*q.deriv()), (n, 's', s, r*s.deriv()) ]
    for n, c in zip(bg['nT'], bg['T']):
        T = Ro*ch.Chebyshev(c, domain=dom)
        out += [ (n, 't', T, r*T.deriv()) ]
    return out



def harmonics(m, lw, x):
    '''Y_l^m at φ = 0 (seminormalized, as ut.Ylm_full_vec) and its θ-derivatives at x = cos θ, rows l = 0 .. lw:
    Y, ∂_θY, ∂²_θY, Z = m Y/sin θ and ∂_θZ'''
    sth = np.sqrt(1 - x**2)
    Y = np.zeros((lw+1, x.size))
    Y[m] = (-1)**m * np.prod(np.sqrt(1 - 1/(2*np.arange(1, m+1)))) * sth**m
    for l in range(m+1, lw+1):
        Y2 = Y[l-2] if l-2 >= m else 0
        Y[l] = ( (2*l-1)*x*Y[l-1] - np.sqrt((l-1)**2 - m**2)*Y2 ) / np.sqrt(l**2 - m**2)
    l   = np.arange(lw+1)[:,None]
    Y1  = np.vstack([ 0*x, Y[:-1] ])  # Y_{l-1}
    dY  = ( l*x*Y - np.sqrt(np.maximum(l**2 - m**2, 0))*Y1 ) / sth  # sin θ ∂_θY_l = l x Y_l - √(l² - m²) Y_{l-1}
    d2Y = -x/sth*dY - (l*(l+1) - (m/sth)**2)*Y                     # Legendre's equation
    return Y, dY, d2Y, m*Y/sth, m*(dY - x/sth*Y)/sth



def shapes(m, lw, x):
    '''Angular parts V and ∂_θV, as (r, θ, φ) components with rows l = 0 .. lw, of the radial factors 'q', 's', 't' of a
    flow (see top)'''
    Y, dY, d2Y, Z, dZ = harmonics(m, lw, x)
    L = np.arange(lw+1)[:,None]*np.arange(1, lw+2)[:,None]
    o = 0*Y
    V  = { 'q': (L*Y,  o, o), 's': (o, dY,  1j*Z),  't': (o, 1j*Z,  -dY)  }
    dV = { 'q': (L*dY, o, o), 's': (o, d2Y, 1j*dZ), 't': (o, 1j*dZ, -d2Y) }
    return V, dV



def advect(a, b, rdb, db, im, x):
    '''r (a·∇)b in (r, θ, φ) components, from those of a, b, r ∂_r b and ∂_θ b, with ∂_φ b = im b'''
    sth = np.sqrt(1 - x**2)
    g = [ a[0]*rdb[k] + a[1]*db[k] + a[2]*im*b[k]/sth for k in range(3) ]
    return ( g[0] - a[1]*b[1] - a[2]*b[2],
             g[1] + a[1]*b[0] - a[2]*b[2]*x/sth,
             g[2] + a[2]*b[0] + a[2]*b[1]*x/sth )



def radial(terms, gbx):
    '''
    Dense matrix of Σ_d c_d(r) (d/dr)^d acting on Chebyshev coefficients, terms = {d: c_d} with c_d Chebyshev series
    in r, in the C^(gbx) basis and with the first gbx rows left for the boundary conditions, as submatrices.py does
    '''
    N = par.N
    def S(a, b):  # C^(a) to C^(b) coefficients
        out = ss.identity(N, format='csr')
        for k in range(a, b):
            out = ut.Slam(k, N) @ out
        return out
    out = np.zeros((N,N))
    for d, c in terms.items():
        a = np.zeros(N)
        a[:min(N, c.coef.size)] = c.coef[:N]
        if np.any(a):
            D = ut.Dlam(d, N, par.ricb, ut.rcmb) if d > 0 else ss.identity(N)
            out += ( S(d, gbx) @ ut.Mlam(S(0, d) @ a, d, 0) @ D ).toarray()
    return np.vstack([ np.zeros((gbx,N)), out[:-gbx] ])



def dr(b):
    '''{d: c_d} of ∂_r Σ_d b_d (d/dr)^d, for b = {d: b_d}'''
    out = {}
    for d, bd in b.items():
        out[d]   = out.get(d, 0) + bd.deriv()
        out[d+1] = out.get(d+1, 0) + bd
    return out



def pieces(bg, x):
    '''
    The terms of r F at x = cos θ, [(n, j, op, F)]: Σ_d op_d(r) (d/dr)^d acting on the scalar P (j = 'q', 's') or T
    (j = 't') of a source of degree l, times the angular function F[:, l] in (r, θ, φ) components. n is the degree of
    the factor of U (bg from background()).
    '''
    dom = [par.ricb, ut.rcmb]
    r   = ch.Chebyshev.identity(domain=dom)
    one = ch.Chebyshev([1], domain=dom)
    ir  = ch.Chebyshev.interpolate(lambda r: 1/r, par.N + 30, domain=dom)
    rho = { 'q': {0: ir}, 's': {0: ir, 1: one}, 't': {0: one} }  # q = P/r, s = P/r + P' and T, as Σ_d c_d (d/dr)^d
    rdr = lambda op: { d: r*c for d, c in dr(op).items() }      # r ∂_r of such an operator
    Vu, dVu = shapes(par.m, par.lmax + 1, x)
    Vb, dVb = shapes(0, max(n for n, *_ in bg), x)
    o   = (0, 0, 0)  # the terms with r ∂_r are the last two below
    out = []
    for n, name, b, rdb in bg:
        U, dU = [ v[n] for v in Vb[name] ], [ v[n] for v in dVb[name] ]
        for j in 'qst':
            u, du = Vu[j], dVu[j]
            for c, op, F in ( (b,   rho[j],      np.add(advect(U, u, o, du, 1j*par.m, x), advect(u, U, o, dU, 0, x))),
                              (b,   rdr(rho[j]), [ U[0]*v for v in u ]),     # U_r r ∂_r u
                              (rdb, rho[j],      [ u[0]*v for v in U ]) ):   # u_r r ∂_r U
                if np.any(F):
                    out.append( (n, j, { d: c*a for d, a in op.items() }, F) )
    return out



def setup():
    '''
    Angular coefficients and radial operators of all the terms, stacked per (row section, source component), so that
    a block is L' Σ C[:, l', l] R (see terms)
    '''
    assert par.ricb > 0 and par.hydro and not (par.magnetic or par.thermal or par.compositional), \
        'bgflow needs ricb > 0, and the flow only (magnetic = thermal = compositional = 0)'
    r    = ch.Chebyshev.identity(domain=[par.ricb, ut.rcmb])
    rows = { 'r': lambda op: radial({ d: r**2*c for d, c in op.items() }, 4),      # section u: L' (r² Fr - r³ ∂_r Fs)
             's': lambda op: radial({ d: -r**3*c for d, c in dr(op).items() }, 4),
             't': lambda op: radial({ d: -r*c for d, c in op.items() }, 2) }       # section v: -L' r Ft
    bg   = background()
    lw   = par.lmax + 1
    x, w = leg.leggauss(lw + max(n for n, *_ in bg) + 4)
    Y, dY, _, Z, _ = harmonics(par.m, lw, x)
    l    = np.arange(lw+1)[:,None]
    L    = np.maximum(l*(l+1), 1)
    proj = lambda F, B: ((F*w) @ B.T).T * (2*l+1)/2  # [l', l] of ∫ F_l B_l' dx / ∫ Y_l'² dx

    stack = { (sec, comp): ([], []) for sec in 'uv' for comp in ('upol', 'utor') }
    for n, j, op, F in pieces(bg, x):
        C = { 'r': proj(F[0], Y), 's': (proj(F[1], dY) - 1j*proj(F[2], Z))/L,
                                  't': (proj(F[2], dY) + 1j*proj(F[1], Z))/L }
        for k in 'rst':
            if np.any(C[k]):
                key = ( 'v' if k == 't' else 'u', 'utor' if j == 't' else 'upol' )
                stack[key][0].append( (abs(l - l.T) <= n)*C[k] )
                stack[key][1].append( rows[k](op) )
    return { key: (np.array(C), np.array(R, dtype=complex)) for key, (C, R) in stack.items() }



def terms(l, section):
    '''
    [(matrix, column)] of the background flow terms in the row block of degree l of section 'u' or 'v'.
    assemble.py adds them to A, as the Coriolis terms.
    '''
    global _blocks
    if _blocks is None:
        _blocks = setup()
    nb = int((par.lmax - par.m + 1)/2)
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    out = []
    for comp, ls, base in ( ('upol', lp, 0), ('utor', lt, nb*ut.N1) ):
        C, R = _blocks[section, comp]
        for i, l1 in enumerate(ls):
            c = C[:, l, l1]
            if np.any(c):
                out.append( ( ss.csr_matrix( l*(l+1)*np.tensordot(c, R, 1) ), base + i*ut.N1 ) )
    return out



def matrix():
    '''The terms of sections u and v as one sparse matrix on the flow unknowns, laid out as in assemble.py, for the
    current bgflow_file and bgflow_Ro'''
    global _blocks
    _blocks = setup()
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    out = [ [], [], [] ]
    for section, ls, row0 in ( ('u', lp, 0), ('v', lt, ut.n) ):
        for i, l in enumerate(ls):
            for mtx, col in terms(l, section):
                out = ut.packit(out, mtx, row0 + i*ut.N1, col)
    data, row, col = [ np.concatenate(v) for v in out ]
    return ss.csr_matrix( (data, (row, col)), shape=(2*ut.n, 2*ut.n) )



def write(x, amp, nmax, Ro=None):
    '''Writes bgflow.npz from kore's field x of a forcing = 12 problem (the flow is x + c.c.), divided by amp, with the
    degrees n <= nmax. Ro marks a nonlinear flow, which holds at bgflow_Ro = Ro only.'''
    lp, lt = ut.ell(0, par.lmax, 1)[:2]
    P = 2*np.real( x[:ut.n] ).reshape(len(lp), ut.N1) / amp
    T = 2*np.real( x[ut.n:2*ut.n] ).reshape(len(lt), ut.N1) / amp
    nonlin = {} if Ro is None else {'Ro': Ro}
    np.savez('bgflow.npz', nP=lp[lp <= nmax], P=P[lp <= nmax], nT=lt[lt <= nmax], T=T[lt <= nmax], ricb=par.ricb,
             **nonlin)



def newton(nmax, dRo=0.15, tol=1e-9):
    '''
    The nonlinear steady flow at Ro = forcing_amplitude_icb (in units of Ω) of a forcing = 12 problem. kore's field x
    (the flow is U = 2x) solves A x + ½ B(2x) x = b, where A x = b is the linear problem and B(U) = matrix() with the
    background U. B(2x) is also the Jacobian of the nonlinear term, so a Newton step is

        (A + B_k) x_k+1 = b + ½ B_k x_k ,    B_k = B(2 x_k) ,

    with all the degrees of x_k in B_k (for quadratic convergence). Ro is reached from 0 in steps of at most dRo. The
    matrices are nearly dense, so the solves are dense. Writes bgflow.npz with the degrees n <= nmax.
    '''
    import scipy.linalg as sl
    import assemble
    par.bgflow = 0
    assemble.main()                                          # writes A.npz and B_forced.npz
    A, b = ut.load_csr('A.npz').real, ut.load_csr('B_forced.npz').toarray().ravel().real  # real for m = 0
    Ro = par.forcing_amplitude_icb/par.OmgTau
    x1 = sl.solve(A.toarray(), b)/Ro                         # the linear flow, per unit Ro
    par.bgflow, par.bgflow_file = 1, 'bgflow.npz'
    x, R0 = x1, 1.0
    nstep = int(np.ceil(abs(Ro)/dRo))
    for R in Ro*np.arange(1, nstep + 1)/nstep:
        x = R*x1 + (R/R0)**2*(x - R0*x1)                     # first guess, with the O(Ro²) part of the last step
        par.bgflow_Ro = R
        for k in range(20):
            write(x, R*par.OmgTau, par.lmax + 1, R)          # U_k = 2 x_k, all degrees
            B = matrix().real
            xn = sl.solve((A + B).toarray(), (R/Ro)*b + 0.5*(B @ x))
            dx, x = np.linalg.norm(xn - x)/np.linalg.norm(xn), xn
            print('Ro = %.4f, step %d: |x_k+1 - x_k|/|x_k+1| = %.1e' % (R, k, dx), flush=True)
            if dx < tol:
                break
        R0 = R
    print('|U - Ro U_lin|/|Ro U_lin| = %.3f' % (np.linalg.norm(x - Ro*x1)/np.linalg.norm(Ro*x1)))
    write(x, Ro*par.OmgTau, nmax, Ro)



def power(usol2):
    '''
    Rate of working of the background flow on a solution, -2 Re ∫ u*·F dV = -2 Re ∫ u*·(u·∇)U dV (the term (U·∇)u does
    no work, as ∇·U = 0 and u = 0 on the walls), in the units of spin_doctor's KE. usol2 as from
    utils4pp.expand_reshape_sol, on the grid of utils4pp.setup_grid.
    '''
    import utils4pp as upp
    lp, lt = ut.ell(par.m, par.lmax, par.symm)[:2]
    bg   = background()
    nmax = max(n for n, *_ in bg)
    x, w = leg.leggauss(par.lmax + nmax + 6)
    rk = upp.rk
    P  = upp.cheb2space_table(usol2[0], 1)  # P and P' at the radii rk, rows l of lp
    T  = upp.cheb2space_table(usol2[1], 0)[0]
    V  = shapes(par.m, par.lmax + 1, x)[0]
    u  = [ (P[0]/rk).T @ V['q'][k][lp] + (P[0]/rk + P[1]).T @ V['s'][k][lp] + T.T @ V['t'][k][lt] for k in range(3) ]
    Vb, dVb = shapes(0, nmax, x)
    U   = [ sum( np.outer(b(rk),   Vb[name][k][n])  for n, name, b, rdb in bg ) for k in range(3) ]
    rdU = [ sum( np.outer(rdb(rk), Vb[name][k][n])  for n, name, b, rdb in bg ) for k in range(3) ]
    dU  = [ sum( np.outer(b(rk),   dVb[name][k][n]) for n, name, b, rdb in bg ) for k in range(3) ]
    F   = advect(u, U, rdU, dU, 0, x)  # r (u·∇)U
    w_r = -2*np.real( sum( np.conj(u[k])*F[k] for k in range(3) ) ) @ (2*np.pi*w) / rk  # angular integral at each radius
    return upp.cg_quad( rk**2*w_r, par.ricb, ut.rcmb, par.N, upp.sqx )



if __name__ == '__main__':
    # python3 bin/bgflow.py nmax [nonlinear], in the folder of a forcing = 12 problem (see top)
    import sys
    nmax = int(sys.argv[1])
    assert par.forcing == 12 and par.forcing_frequency == 0 and par.m == 0 and par.symm == 1 \
        and par.forcing_amplitude_icb != 0
    if sys.argv[2:] == ['nonlinear']:
        newton(nmax)
    else:
        write(np.loadtxt('real_flow.field') + 1j*np.loadtxt('imag_flow.field'), par.forcing_amplitude_icb, nmax)
    print('bgflow.npz written, with the degrees n <=', nmax)
