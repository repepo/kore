#!/usr/bin/env python3
"""Plot full linear CMB Maxwell torque profiles using Kore's shared routine.

Run from the repository root with fields matching bin/parameters.py:
    python3 tools/plot_cmb_torque_latitude.py 0 --component x --quantity real

Profiles are per radian; coordinates are in degrees. Mode amplitudes are
arbitrary. Values are dynamic nondimensional, not N m.
"""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import parameters as par
import utils as ut
import utils_pp as upp


def compute_profile(solnum, component='z', ntheta=181, nphi=361,
                    theta0=0.0, theta1=180.0, phase=0.0, profile='latitude'):
    """Return coordinate in degrees, complex profile and integrated torque."""
    b = (np.loadtxt('real_magnetic.field', usecols=solnum)
         + 1j*np.loadtxt('imag_magnetic.field', usecols=solnum))
    b = upp.expand_sol(np.asarray(b), ut.bsymm)
    result = ut.gamma_magnetic_cmb_distribution(
        b, ntheta, nphi, np.deg2rad(theta0), np.deg2rad(theta1))
    k = 'xyz'.index(component)
    scale = par.Le2*par.OmgTau**2*np.exp(1j*phase)
    if profile == 'latitude':
        x = 90 - np.rad2deg(result['theta'])
        values = result['profile_xyz'][:, k]
    else:
        x = np.rad2deg(result['phi'])
        values = np.sum(result['density_xyz'][..., k]
                        * result['solid_angle_weights'][:, None], axis=0)
    return x, values*scale, result['torque_xyz'][k]*scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solnum', type=int)
    parser.add_argument('--component', choices=['x', 'y', 'z'], default='z')
    parser.add_argument('--ntheta', type=int, default=181)
    parser.add_argument('--nphi', type=int, default=361)
    parser.add_argument('--theta0', type=float, default=0.0)
    parser.add_argument('--theta1', type=float, default=180.0)
    parser.add_argument('--phase', type=float, default=0.0, help='Phase in degrees')
    parser.add_argument('--profile', choices=['latitude', 'longitude'], default='latitude')
    parser.add_argument('--quantity', choices=['physical_abs', 'real', 'imag', 'complex_abs'],
                        default='complex_abs')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--save')
    args = parser.parse_args()
    x, values, total = compute_profile(
        args.solnum, args.component, args.ntheta, args.nphi, args.theta0,
        args.theta1, np.deg2rad(args.phase), args.profile)
    transforms = {'physical_abs': lambda z: np.abs(z.real), 'real': np.real,
                  'imag': np.imag, 'complex_abs': np.abs}
    y = transforms[args.quantity](values)
    if args.normalize and np.max(np.abs(y)) > 0:
        y = y/np.max(np.abs(y))
    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(x)
    ax.plot(x[order], y[order], lw=1.7)
    ax.set_xlabel(args.profile.capitalize() + ' (degrees)')
    variable = r'\theta' if args.profile == 'latitude' else r'\phi'
    ax.set_ylabel(rf'$d\Gamma_{args.component}/d{variable}$ ({args.quantity}, per radian)'
                  + (' normalized' if args.normalize else ''))
    ax.set_title(f'CMB torque on mantle: {args.component}, mode {args.solnum}')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    print(f'Integrated Gamma_{args.component} = {total:.8e} (dynamic nondimensional)')
    print('Integrate the signed complex profile, not its absolute value.')
    if args.save:
        fig.savefig(args.save, dpi=220)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
