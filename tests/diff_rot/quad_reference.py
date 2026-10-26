'''
quad_reference.py: the Gauss-Legendre quadrature builder of the differential-rotation primitives that bin/diff_rot_generic.py
used before it was rebuilt on the recurrence algebra of bin/angular.py, kept as an independent reference for
tests/diff_rot/angular_check.py. Standalone (no parameters.py).

The background flow U0 = G ΔΩ r sinθ e_φ, ΔΩ = f_n(r) P_n(cos θ), acts on the perturbation u through
    W = G [ ΔΩ (i m u + 2 ẑ×u) + sinθ (r ∂ᵣΔΩ u_r + ∂_θΔΩ u_θ) e_φ ],
with u of degree l: u_r = L P Y/r, u_θ = a ∂_θY + i m T Y/sinθ, u_φ = i m a Y/sinθ - T ∂_θY, a = (rP)'/r + h' P. Each
component of W is a radial shape (q = L P/r, a, T, or q with r f_n' in place of f_n, 'dq') times an angular function;
projecting W of source degree l on the target degree l' (W = Σ [qW Y r̂ + sW r∇Y + tW r∇Y×r̂]) gives the 12 primitive
matrices C[k][shape][l', l], k = q, s, t, here by Gauss-Legendre quadrature in cos θ (exact for these polynomial integrands).
Limitation: scipy.special.sph_harm_y returns NaN from l = 646 on, so use lw <= 645.
'''
import numpy as np
import scipy.special as scsp

def harmonics(m, lw, nq):
    '''Gauss-Legendre nodes, weights and Schmidt Y_l^m (as utils.Ylm at φ = 0), ∂_θY, i m Y/sinθ for l = 0..lw'''
    if lw > 645:
        raise ValueError('quad_reference: scipy sph_harm_y is NaN for l >= 646 (lw = %d)' % lw)
    x, w = np.polynomial.legendre.leggauss(nq)
    th = np.arccos(x)
    sth = np.sqrt(1 - x**2)
    Y = np.zeros((lw + 1, nq), dtype=complex)
    dY = np.zeros_like(Y)
    Z = np.zeros_like(Y)
    for l in range(m, lw + 1):
        y, dy = scsp.sph_harm_y(l, m, th, 0.0, diff_n=1)
        c = np.sqrt(4*np.pi/(2*l + 1))
        Y[l] = y*c
        dY[l] = dy[:, 0]*c
        if m:
            Z[l] = 1j*m*Y[l]/sth
    return x, w, sth, Y, dY, Z

def primitives(n, m, lw):
    '''C[k][shape] (k = 'q', 's', 't'; shape = 'q', 'a', 'T', 'dq') on the degrees 0..lw, row = target, column = source'''
    nq = 2*lw + n + 12
    x, w, sth, Y, dY, Z = harmonics(m, lw, nq)
    Pn = np.polynomial.legendre.Legendre.basis(n)
    pn, dpn = Pn(x), Pn.deriv()(x)
    l = np.arange(lw + 1)
    L = l*(l + 1.)
    nl = 4*np.pi/(2*l + 1)
    z = 0*Y
    # components of W/G per unit radial shape, rows = source l, columns = nodes
    Wr = {'q': 1j*m*pn*Y, 'a': -2j*m*pn*Y, 'T': 2*sth*pn*dY, 'dq': z}
    Wt = {'q': z, 'a': pn*(1j*m*dY - 2*x*Z), 'T': pn*(1j*m*Z + 2*x*dY), 'dq': z}
    Wp = {'q': 2*sth*pn*Y, 'a': pn*(1j*m*Z + 2*x*dY) - sth**2*dpn*dY, 'T': pn*(-1j*m*dY + 2*x*Z) - sth**2*dpn*Z, 'dq': sth*pn*Y}
    proj = lambda F, B: 2*np.pi*(F*w) @ np.conj(B).T   # [source l, target l'] of ∫ F B* dΩ
    with np.errstate(divide='ignore', invalid='ignore'):
        C = {'q': {}, 's': {}, 't': {}}
        for sh in ('q', 'a', 'T', 'dq'):
            C['q'][sh] = (proj(Wr[sh], Y)/nl[None, :]).T
            C['s'][sh] = ((proj(Wt[sh], dY) + proj(Wp[sh], Z))/(L*nl)[None, :]).T
            C['t'][sh] = ((proj(Wt[sh], Z) - proj(Wp[sh], dY))/(L*nl)[None, :]).T
    for k in C:
        for sh in C[k]:
            C[k][sh][~np.isfinite(C[k][sh])] = 0   # only the target l' = 0 (L' = 0), outside kore's basis
    return C
