'''
Angular algebra at a fixed azimuthal order m (docs/differential_rotation_details.md, §3).

Fields are expanded on the Schmidt harmonics of utils.Ylm, Y_l = sqrt(4π/(2l+1)) Y_l^m (θ-part, Condon–Shortley phase),
∫|Y_l|² dΩ = n_l = 4π/(2l+1). A function of x = cos θ multiplying a field is a matrix on the degree index (row: target l',
column: source l), built from two exact recurrences:
    x Y_l = p_l Y_{l+1} + q_l Y_{l-1}                         -> X,   p_l = sqrt((l+1)² - m²)/(2l+1),  q_l = sqrt(l² - m²)/(2l+1)
    (1 - x²) dY_l/dx = (l+1) q_l Y_{l-1} - l p_l Y_{l+1}       -> D    (= -sinθ ∂θ)
and the adjoint of D under ∫ dΩ, D† = N⁻¹ Dᵀ N = 2X - D. Vector fields are expanded as
    F = Σ_l [ q_l Y_l r̂ + s_l r∇Y_l + t_l r∇Y_l × r̂ ]
(a Field: three sparse matrices q, s, t whose columns are the source "shapes"), and the operations needed by the
differential rotation (multiplication by g(x), ẑ×, sinθ F_θ) are exact sparse products of X, D and diagonals:
    g R_l -> q: g(X);   g S_l -> s: (g(X) L - g'(X) D)/L', t: -i m g'(X)/L';   g T_l -> s: i m g'(X)/L', t: (g(X) L - g'(X) D)/L'
    ẑ × (c Y r̂) -> s: -i m c/L', t: D† c/L';   ẑ × S_l = -i m R_l - x T_l;   ẑ × T_l = -(D Y_l) r̂ + x S_l
    sinθ F_θ -> -D s + i m t
(R_l = Y_l r̂, S_l = r∇Y_l, T_l = r∇Y_l × r̂; 1/L' is 0 on the row l' = 0). Products of truncated matrices are wrong only
within a few degrees of the truncation, so build with a margin above the largest degree used (lmax + n + 6 is enough).
Also: Schmidt harmonics and their θ-derivative on a grid of x by the same recurrence (ylm_grid, ylm), valid at any l and m
(rescaled by powers of two where sin^m θ would underflow)
(scipy.special.sph_harm_y returns NaN from l = 646 on).
Depends on numpy and scipy only (no parameters), so it can be imported and tested anywhere.
'''
import numpy as np
import scipy.sparse as sp

_cache = {}


def coefficients(m, lw):
    '''p_l, q_l of the recurrence x Y_l = p_l Y_{l+1} + q_l Y_{l-1}, l = 0..lw (zero for l < m)'''
    l = np.arange(lw + 1, dtype=float)
    am = abs(m)
    p = np.where(l >= am, np.sqrt(np.maximum((l + 1)**2 - am**2, 0.))/(2*l + 1), 0.)
    q = np.where(l >= am, np.sqrt(np.maximum(l**2 - am**2, 0.))/(2*l + 1), 0.)
    return p, q


class Algebra:
    '''
    The matrices of the angular algebra at order m on the degrees 0..lw (sparse CSR, row = target degree):
    X, D, Dd = D†, I (identity on l >= m), L = diag(l(l+1)), Li = diag(1/L, 0 at l = 0), N = diag(4π/(2l+1)).
    Use Algebra.get(m, lw) for a cached instance.
    '''
    def __init__(self, m, lw):
        self.m, self.lw = m, lw
        p, q = coefficients(m, lw)
        l = np.arange(lw + 1, dtype=float)
        # X[l+1, l] = p_l, X[l-1, l] = q_l;  D[l+1, l] = -l p_l, D[l-1, l] = (l+1) q_l
        self.X = (sp.diags(p[:-1], -1) + sp.diags(q[1:], 1)).tocsr()
        self.D = (sp.diags(-l[:-1]*p[:-1], -1) + sp.diags((l[1:] + 1)*q[1:], 1)).tocsr()
        self.Dd = (2*self.X - self.D).tocsr()
        self.I = sp.diags((l >= abs(m)).astype(float)).tocsr()
        self.Lv = l*(l + 1)
        self.L = sp.diags(self.Lv).tocsr()
        self.Li = sp.diags(np.where(self.Lv > 0, 1/np.where(self.Lv > 0, self.Lv, 1.), 0.)).tocsr()
        self.N = sp.diags(4*np.pi/(2*l + 1)).tocsr()
        self._leg = [(self.I, 0*self.I), (self.X, self.I)]   # (P_k(X), P_k'(X)), k = 0, 1

    @classmethod
    def get(cls, m, lw):
        key = (m, lw)
        if key not in _cache:
            _cache[key] = cls(m, lw)
        return _cache[key]

    def legendre(self, n):
        '''P_n(X) and P_n'(X), from (k+1) P_{k+1} = (2k+1) X P_k - k P_{k-1} and P_{k+1}' = (k+1) P_k + X P_k' '''
        while len(self._leg) <= n:
            k = len(self._leg) - 1
            (Pk, dPk), (Pm, _) = self._leg[k], self._leg[k - 1]
            self._leg.append((((2*k + 1)*(self.X @ Pk) - k*Pm)/(k + 1), (k + 1)*Pk + self.X @ dPk))
        P, dP = self._leg[n]
        return P.tocsr(), dP.tocsr()

    # --- vector fields (q, s, t), see Field -------------------------------------------------------------------------
    def zero(self):
        return 0*self.I

    def field(self, q=None, s=None, t=None):
        z = self.zero()
        return Field(z if q is None else q, z if s is None else s, z if t is None else t)

    def mult(self, F, g, dg):
        '''the field g(x) F, with g = g(X) and dg = g'(X) given as matrices'''
        m, Li, L, D = self.m, self.Li, self.L, self.D
        gL_D = lambda c: g @ (L @ c) - dg @ (D @ c)
        return Field(g @ F.q,
                     Li @ (gL_D(F.s) + 1j*m*(dg @ F.t)),
                     Li @ (gL_D(F.t) - 1j*m*(dg @ F.s)))

    def zcross_r(self, c):
        '''ẑ × (c r̂) for a scalar field with coefficients c (= c sinθ e_φ)'''
        return Field(self.zero(), self.Li @ (-1j*self.m*c), self.Li @ (self.Dd @ c))

    def zcross(self, F):
        '''ẑ × F:  ẑ × R_l = (c sinθ e_φ, zcross_r),  ẑ × S_l = -i m R_l - x T_l,  ẑ × T_l = -(D Y_l) r̂ + x S_l'''
        X = self.X
        out = self.zcross_r(F.q)
        out = out + Field(-1j*self.m*F.s, self.zero(), self.zero()) - self.mult(self.field(t=F.s), X, self.I)
        out = out + Field(-(self.D @ F.t), self.zero(), self.zero()) + self.mult(self.field(s=F.t), X, self.I)
        return out

    def sin_theta(self, F):
        '''the scalar sinθ F_θ:  sinθ ∂θY = -D Y,  sinθ (i m Y/sinθ) = i m Y'''
        return -(self.D @ F.s) + 1j*self.m*F.t


class Field:
    '''A vector field (q, s, t) on R_l = Y_l r̂, S_l = r∇Y_l, T_l = r∇Y_l × r̂: three matrices, rows = degree'''
    def __init__(self, q, s, t):
        self.q, self.s, self.t = q, s, t

    def __add__(self, o):
        return Field(self.q + o.q, self.s + o.s, self.t + o.t)

    def __sub__(self, o):
        return Field(self.q - o.q, self.s - o.s, self.t - o.t)

    def __mul__(self, a):   # by a scalar
        return Field(a*self.q, a*self.s, a*self.t)

    __rmul__ = __mul__

    def __getitem__(self, k):
        return {'q': self.q, 's': self.s, 't': self.t}[k]


def band(M, B):
    '''the diagonals -B..B of a square matrix as an array A[dl + B, l] = M[l, l + dl] (0 outside the matrix)'''
    M = sp.csr_matrix(M)
    n = M.shape[0]
    A = np.zeros((2*B + 1, n), dtype=M.dtype)
    for k in range(-B, B + 1):
        d = M.diagonal(k)
        if k >= 0:
            A[k + B, :n - k] = d
        else:
            A[k + B, -k:] = d
    return A


# --- Schmidt harmonics on a grid -------------------------------------------------------------------------------------
_BIG = 600   # rescaling step of ylm_grid, as a power of two

def _sin_power(sth, n):
    '''sin^n θ as a mantissa f and a power-of-two exponent k (f 2^k): sth**n where that is a normal number (k = 0), and
    otherwise a product renormalised at each step by frexp, which cannot underflow (large m near the poles)'''
    f = sth**n
    k = np.zeros(sth.shape, dtype=np.int64)
    small = (f < 2.0**-900) & (sth > 0)
    if small.any():
        fs, ks = np.frexp(sth[small])
        g = np.ones_like(fs); kk = np.zeros(fs.shape, dtype=np.int64)
        for _ in range(n):
            g, e = np.frexp(g*fs)
            kk += e + ks
        f[small], k[small] = g, kk
    return f, k

def ylm_grid(m, lmax, x):
    '''
    Schmidt harmonics Y_l (θ-part of utils.Ylm, including the Condon–Shortley phase), ∂θY_l and Z_l = i m Y_l/sinθ at the
    points x = cos θ, for l = 0..lmax (rows l < |m| are zero; m >= 0). By the recurrence of X, started at
    Y_m = (-1)^m sqrt(Π_{k=1..m} (2k-1)/(2k)) sin^m θ, Y_{m-1} = 0; ∂θY by the same recurrence differentiated in θ, and
    Y/sinθ by the same recurrence started at sin^(m-1) θ, so that nothing is divided by sinθ.
    The recurrence is linear, so it is carried as a mantissa times 2^k per point: for large m, sin^(m-1) θ would underflow
    near the poles (from m ≈ 330 at l of a few thousand, where Y_l there is not small); the mantissas are rescaled by
    2^-600 when they grow past 2^600 and the values returned with ldexp. Powers of two scale exactly, and where the start
    is a normal number k stays 0, so the values are then those of the plain recurrence.
    '''
    x = np.atleast_1d(np.asarray(x, dtype=float))
    sth = np.sqrt(np.maximum(1 - x**2, 0.))
    p, q = coefficients(m, lmax)
    Y = np.zeros((lmax + 1, x.size)); dY = np.zeros_like(Y); V = np.zeros_like(Y)   # V = Y/sinθ (m >= 1)
    if m > lmax:
        return Y, dY, 0j*Y
    c = (-1)**m*np.sqrt(np.prod((2*np.arange(1, m + 1) - 1)/(2*np.arange(1, m + 1)))) if m > 0 else 1.
    if m > 0:
        sm1, k = _sin_power(sth, m - 1)           # sin^(m-1) θ = sm1 2^k
    else:
        sm1, k = 0*x, np.zeros(x.shape, dtype=np.int64)
    # mantissas of the levels l - 1 (a0, b0, v0) and l (a1, b1, v1) of Y, ∂θY, Y/sinθ, all in units of 2^k
    a1 = c*sm1*sth if m > 0 else 1. + 0*x
    b1 = c*m*x*sm1 if m > 0 else 0*x              # ∂θ sin^m θ = m cosθ sin^(m-1) θ
    v1 = c*sm1 if m > 0 else 0*x
    a0 = b0 = v0 = 0*x
    Y[m], dY[m], V[m] = np.ldexp(a1, k), np.ldexp(b1, k), np.ldexp(v1, k)
    for l in range(m, lmax):                      # Y_{l+1} = (x Y_l - q_l Y_{l-1})/p_l
        a2 = (x*a1 - q[l]*a0)/p[l]
        b2 = (x*b1 - sth*a1 - q[l]*b0)/p[l]
        v2 = (x*v1 - q[l]*v0)/p[l]
        big = (np.abs(a2) > 2.0**_BIG) | (np.abs(b2) > 2.0**_BIG) | (np.abs(v2) > 2.0**_BIG)
        if big.any():
            for arr in (a1, b1, v1, a2, b2, v2):
                arr[big] = np.ldexp(arr[big], -_BIG)
            k = k + _BIG*big
        a0, b0, v0, a1, b1, v1 = a1, b1, v1, a2, b2, v2
        Y[l + 1], dY[l + 1], V[l + 1] = np.ldexp(a1, k), np.ldexp(b1, k), np.ldexp(v1, k)
    return Y, dY, 1j*m*V


def ylm(l, m, theta, phi):
    '''utils.Ylm: the Schmidt harmonic sqrt(4π/(2l+1)) Y_l^m(θ, φ) of scipy's sph_harm_y convention, any l; m < 0 by
    Y_l^{-m} = (-1)^m conj(Y_l^m)'''
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    am = abs(m)
    if l < am:
        return np.zeros(theta.shape, dtype=complex)
    Y = ylm_grid(am, l, np.cos(theta).ravel())[0][l].reshape(theta.shape)
    out = Y*np.exp(1j*am*phi)
    return out if m >= 0 else (-1)**am*np.conj(out)
