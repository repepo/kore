#!/usr/bin/env python3
'''
tw_mukhopadhyay.py: kore's thermal-wind entropy S_tw for diff_rot_type = 'tabulated' against the entropy gradient of the
data set of Mukhopadhyay et al. 2025 (doi 10.17617/3.NP26AI, data/diff_rot.npz: ome, dsdt = ∂θ(s0/c_p) on (θ, r)).
Run in a run directory whose diff_rot_file is that diff_rot.npz (thermal_wind 1 or 2):

    python3 tw_mukhopadhyay.py path/to/diff_rot.npz [c_p]

Prints ∂θS_tw against dsdt at a few latitudes and radii, the pole-to-equator contrast c_p (s_pole - s_eq) = -c_p ∫ ∂θS dθ (erg/g/K) (c_p default 3.46e8,
the papers use 4.17e8; the contrast scales with c_p only through this conversion), and the rms mismatch over the shell.
Their θ grid runs from the south pole (θ = π) to the north pole and the stored dsdt is the gradient on that grid, so
the physical ∂θ(s0/c_p) at colatitude θ_k is -dsdt[k] with θ_k = π - theta[k]; both are printed in that convention.
'''
import sys, os
sys.path.insert(0, os.path.abspath('bin'))
import numpy as np
from parameters import par
import radial_profiles as rap
import diff_rot as dr

V = np.load(sys.argv[1]); cp = float(sys.argv[2]) if len(sys.argv) > 2 else 3.46e8
th_e, r_e, dsdt = V['theta'], V['r'], V['dsdt']
k = (r_e >= par.ricb*par.r_out + 0.005) & (r_e <= par.r_out - 0.005)
x_e = r_e[k]; r = x_e/par.r_out
sig = dr.sigma_coefficients(r)            # σ_n(r), n = 0 ... the highest degree of S_tw
th = np.pi - th_e                         # colatitude of each θ node of their table
xx = np.cos(th)
Lg = np.polynomial.legendre
dPdx = np.array([Lg.Legendre.basis(n).deriv()(xx) for n in range(len(sig))])     # (n, θ)
dS_k = -np.sin(th)[:, None]*(dPdx.T @ sig)                                        # (θ, r): ∂θ S_tw
dS_e = -dsdt[:, k]
print('thermal_wind %d, diff_rot_type %s, nmax %d, %d radii in [%.3f, %.3f] R_sun'
      % (par.thermal_wind, par.diff_rot_type, dr.n_max(), len(r), x_e[0], x_e[-1]))
print('  r/R    lat:   ∂θS kore at 20, 45, 70 deg          | Mukhopadhyay                       | c_p (s_pole - s_eq) kore, Mukhopadhyay')
north = th <= np.pi/2
for rr in (0.72, 0.75, 0.80, 0.85, 0.90, 0.95):
    i = np.argmin(abs(x_e - rr))
    js = [np.argmin(abs(90 - np.rad2deg(th) - lat)) for lat in (20, 45, 70)]
    o = np.argsort(th[north])
    I = lambda f: cp*np.trapezoid(f[north][o], th[north][o])
    print('  %.3f       %s | %s | %9.1f  %9.1f' % (x_e[i], ' '.join('%10.3e' % dS_k[j, i] for j in js),
          ' '.join('%10.3e' % dS_e[j, i] for j in js), -I(dS_k[:, i]), -I(dS_e[:, i])))
rms = np.sqrt(np.mean((dS_k - dS_e)**2))/np.sqrt(np.mean(dS_e**2))
print('rms(kore - Mukhopadhyay)/rms(Mukhopadhyay) over the shell: %.3f' % rms)
