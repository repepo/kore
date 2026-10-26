#!/usr/bin/env python3
'''
Physical units, outside kore: parameter files and scripts import this module; assemble.py, solve.py and spin_doctor do
not. kore is dimensionless: time in 1/Ω (Gaspard = 1), length L = aux0 star radii, ν and κ through ViscosD and
ThermaD, N² through Beyonce, eigenvalues λ in units of Ω. The one unit it reads is that of the solar rotation data
('solar_P4', 'tabulated': nHz), divided by the reference rate diff_rot_frame.

In a parameter file (after aux0 is set):
    import units
    units.set_physical(par, frame_nHz=456.0, nu=1e12, kappa=1e12)   # Gaspard, Beyonce, ViscosD, ThermaD, diff_rot_frame
    units.set_target(par, omega=-254.8, sigma=-33.5)                 # itau, rtau from ω, σ in nHz
In a run directory, the eigenvalues as ω and σ in nHz:
    python3 bin/units.py [eigenvalues0.dat]
'''
import sys
import numpy as np

RSUN  = 6.9599e10   # cm, as examples/smooth_sun and the builders
GSURF = 2.7361e4    # cm/s²


def set_physical(par, frame_nHz, nu, kappa, radius=RSUN, gsurf=GSURF):
    '''
    Gaspard = 1, Beyonce = gsurf/(L Ω²), ViscosD = ν/(Ω L²), ThermaD = κ/(Ω L²) and diff_rot_frame = frame_nHz, with
    Ω = 2π frame_nHz 1e-9 s⁻¹ and L = aux0 radius; ν, κ in cm²/s, radius in cm, gsurf in cm/s² (Beyonce as for the
    solar structures, whose g is 1 at the surface).
    '''
    Omega = 2*np.pi*frame_nHz*1e-9
    L = par.aux0*radius
    par.Gaspard = 1.0
    par.Beyonce = gsurf/(L*Omega**2)
    par.ViscosD = nu/(Omega*L**2)
    par.ThermaD = kappa/(Omega*L**2)
    par.diff_rot_frame = frame_nHz


def set_target(par, omega, sigma):
    '''the target λ = rtau + i itau of the frequency ω and the growth rate σ in nHz: itau = -ω G/F, rtau = σ G/F
    (F = diff_rot_frame, G = Gaspard)'''
    par.itau = -omega*par.Gaspard/par.diff_rot_frame
    par.rtau = sigma*par.Gaspard/par.diff_rot_frame


def nHz(lam, frame_nHz, Gaspard=1.0):
    '''ω = -Im λ F/G and σ = Re λ F/G in nHz of kore eigenvalues λ (F = frame_nHz, G = Gaspard)'''
    lam = np.asarray(lam)
    return -lam.imag*frame_nHz/Gaspard, lam.real*frame_nHz/Gaspard


if __name__ == '__main__':
    from parameters import par
    f = sys.argv[1] if len(sys.argv) > 1 else 'eigenvalues0.dat'
    ev = np.atleast_2d(np.loadtxt(f))                   # columns Re λ, Im λ (solve.py)
    om, sg = nHz(ev[:, 0] + 1j*ev[:, 1], par.diff_rot_frame, par.Gaspard)
    print('%s, frame %g nHz, Gaspard %g' % (f, par.diff_rot_frame, par.Gaspard))
    print('%18s %18s %14s %14s' % ('Re λ', 'Im λ', 'ω (nHz)', 'σ (nHz)'))
    for e, o, s in zip(ev, om, sg):
        print('%18.10e %18.10e %14.6f %14.6f' % (e[0], e[1], o, s))
