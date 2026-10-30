#!/usr/bin/env python3
"""
Plot the magnetic field across all resolved solid/fluid layers.

Usage:
    python3 tools/plot_b_layers.py 0 160 360 0 180 abs

Arguments:
    solnum   solution number
    nR       radial points per layer
    ntheta   latitudinal points
    theta0   first colatitude in degrees
    theta1   last colatitude in degrees
    opt      'abs' or 'raw'

The script uses, when available:
    real_magnetic_ic.field / imag_magnetic_ic.field
    real_magnetic.field / imag_magnetic.field
    real_magnetic_mantle.field / imag_magnetic_mantle.field
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.polynomial.chebyshev as ch

sys.path.insert(1, "bin/")

import parameters as par
import utils as ut
import utils_pp as upp


def set_contour_edges_to_face(contour_set):
    for collection in getattr(contour_set, "collections", []):
        collection.set_edgecolor("face")


def load_complex(real_name, imag_name, solnum):
    real_path = Path(real_name)
    imag_path = Path(imag_name)
    if not (real_path.exists() and imag_path.exists()):
        return None
    real = np.loadtxt(real_path, usecols=solnum)
    imag = np.loadtxt(imag_path, usecols=solnum)
    return np.asarray(real) + 1j*np.asarray(imag)


def shell_grid(ra, rb, nr, include_center=False):
    if include_center:
        # Avoid r=0 in the spherical component formulas.
        r = np.linspace(ra, rb, nr + 1)[1:]
    else:
        r = np.linspace(ra, rb, nr)
        if ra == 0:
            r = r[1:]
    return r


def split_full_poltor(vec, vsymm, lmax, n_radial, expand_origin):
    if expand_origin:
        vec = upp.expand_sol(vec, vsymm)

    llpol, lltor, ll = ut.ell(par.m, lmax, vsymm)
    nblocks = len(llpol)
    ncoeff = vec.size // 2
    n_radial = ncoeff // nblocks

    pol = vec[:ncoeff].reshape((nblocks, n_radial))
    tor = vec[ncoeff:ncoeff+ncoeff].reshape((nblocks, n_radial))
    return pol, tor, llpol, lltor, ll, n_radial


def split_inner_core_poltor(vec):
    vsymm = ut.bsymm
    llpol, lltor, ll = ut.ell(par.m, par.lmax_cic, vsymm)
    nblocks = len(llpol)
    ncoeff = vec.size // 2

    pol_reduced = vec[:ncoeff].reshape((nblocks, ut.Nic))
    tor_reduced = vec[ncoeff:ncoeff+ncoeff].reshape((nblocks, ut.Nic))

    pol = np.zeros((nblocks, par.N_cic), dtype=complex)
    tor = np.zeros((nblocks, par.N_cic), dtype=complex)

    if ut.symmB0 == -1:
        ipol = (par.m + ut.s) % 2
        itor = (par.m + 1 - ut.s) % 2
    else:
        ipol = (par.m + 1 - ut.s) % 2
        itor = (par.m + ut.s) % 2

    pol[:, ipol::2] = pol_reduced
    tor[:, itor::2] = tor_reduced
    return pol, tor, llpol, lltor, ll, par.N_cic


def reconstruct_shell(name, vec, ra, rb, nr, theta, inner_core=False, expand_origin=False):
    if inner_core:
        pol, tor, llpol, lltor, ll, n_radial = split_inner_core_poltor(vec)
        lmax = par.lmax_cic
        vsymm = ut.bsymm
        r = shell_grid(0.0, rb, nr, include_center=True)
        x = r / rb
    else:
        lmax = par.lmax
        vsymm = ut.bsymm
        pol, tor, llpol, lltor, ll, n_radial = split_full_poltor(
            vec, vsymm, lmax, None, expand_origin
        )
        r = shell_grid(ra, rb, nr)
        x = upp.xcheb(r, ra, rb)

    chx = ch.chebvander(x, n_radial - 1)
    pol_r = pol @ chx.T
    tor_r = tor @ chx.T

    dpol = np.zeros_like(pol)
    for k in range(len(llpol)):
        dpol[k, :] = ut.Dcheb(pol[k, :], ra, rb)
    dpol_r = dpol @ chx.T

    rpol = pol_r / r[np.newaxis, :]
    qlr = (llpol*(llpol+1))[:, np.newaxis] * rpol
    slr = rpol + dpol_r

    ntheta = theta.size
    s = np.zeros(r.size*ntheta)
    z = np.zeros(r.size*ntheta)
    br = np.zeros(r.size*ntheta, dtype=complex)
    btheta = np.zeros(r.size*ntheta, dtype=complex)
    bphi = np.zeros(r.size*ntheta, dtype=complex)

    clm = np.zeros((lmax - par.m + 2, 1))
    for i, ell in enumerate(ll):
        clm[i] = np.sqrt((ell - par.m)*(ell + par.m))

    sy = int(vsymm*0.5 + 0.5)
    idp = (np.sign(par.m) + sy) % 2
    idt = (np.sign(par.m) + sy + 1) % 2
    plx = idp + lmax - par.m + 1
    tlx = idt + lmax - par.m + 1

    phi = 0.0
    pos = 0
    for th in theta:
        ylm = np.r_[ut.Ylm_full(lmax, par.m, th, phi), 0]
        sin_th = np.sin(th)
        tan_th = np.tan(th)
        for kr, rr in enumerate(r):
            s[pos] = rr*sin_th
            z[pos] = rr*np.cos(th)

            br[pos] = np.dot(qlr[:, kr], ylm[idp:plx:2])

            term1 = np.dot(-(llpol+1)*slr[:, kr]/tan_th, ylm[idp:plx:2])
            term2 = np.dot(clm[idp+1:plx+1:2, 0]*slr[:, kr]/sin_th, ylm[idp+1:plx+1:2])
            term3 = np.dot(1j*par.m*tor_r[:, kr]/sin_th, ylm[idt:tlx:2])
            btheta[pos] = term1 + term2 + term3

            term1 = np.dot((lltor+1)*tor_r[:, kr]/tan_th, ylm[idt:tlx:2])
            term2 = np.dot(-clm[idt+1:tlx+1:2, 0]*tor_r[:, kr]/sin_th, ylm[idt+1:tlx+1:2])
            term3 = np.dot(1j*par.m*slr[:, kr]/sin_th, ylm[idp:plx:2])
            bphi[pos] = term1 + term2 + term3
            pos += 1

    print(
        f"{name}: r=[{ra:g}, {rb:g}], N={n_radial}, "
        f"max|br|={np.max(np.abs(br)):.3e}, "
        f"max|btheta|={np.max(np.abs(btheta)):.3e}, "
        f"max|bphi|={np.max(np.abs(bphi)):.3e}"
    )
    return s, z, br, btheta, bphi


def draw_boundary(ax, radius, **kwargs):
    theta = np.linspace(0.0, np.pi, 400)
    ax.plot(radius*np.sin(theta), radius*np.cos(theta), **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("solnum", type=int)
    parser.add_argument("nR", type=int)
    parser.add_argument("ntheta", type=int)
    parser.add_argument("theta0", type=float)
    parser.add_argument("theta1", type=float)
    parser.add_argument("opt", choices=["abs", "raw"])
    parser.add_argument("--save", default=None, help="Optional output image path.")
    args = parser.parse_args()

    theta = np.linspace(np.deg2rad(args.theta0), np.deg2rad(args.theta1), args.ntheta + 2)[1:-1]

    layers = []

    ic = load_complex("real_magnetic_ic.field", "imag_magnetic_ic.field", args.solnum)
    if ic is not None and ut.icflag and par.ricb > 0:
        layers.append(reconstruct_shell("inner core", ic, 0.0, par.ricb, args.nR, theta, inner_core=True))

    core = load_complex("real_magnetic.field", "imag_magnetic.field", args.solnum)
    if core is None:
        raise FileNotFoundError("Missing real_magnetic.field or imag_magnetic.field")
    layers.append(reconstruct_shell("fluid core", core, par.ricb, ut.rcmb, args.nR, theta, expand_origin=True))

    mantle = load_complex("real_magnetic_mantle.field", "imag_magnetic_mantle.field", args.solnum)
    if mantle is not None and ut.resolved_mantle:
        layers.append(reconstruct_shell("mantle", mantle, ut.rcmb, par.r_surface, args.nR, theta))

    s = np.concatenate([layer[0] for layer in layers])
    z = np.concatenate([layer[1] for layer in layers])
    components = [np.concatenate([layer[i] for layer in layers]) for i in [2, 3, 4]]

    triang = tri.Triangulation(s, z)
    radius_mid = np.sqrt(
        s[triang.triangles].mean(axis=1)**2
        + z[triang.triangles].mean(axis=1)**2
    )
    inner_limit = 0.0 if (ic is not None and ut.icflag and par.ricb > 0) else par.ricb
    outer_limit = par.r_surface if (mantle is not None and ut.resolved_mantle) else ut.rcmb
    mask = (radius_mid < inner_limit - 1e-12) | (radius_mid > outer_limit + 1e-12)
    triang.set_mask(mask)

    labels = [
        r"$b_r$",
        r"$b_\theta$",
        r"$b_\phi$",
    ]
    if args.opt == "abs":
        values = [np.abs(comp) for comp in components]
        labels = [
            r"$|b_r|$",
            r"$|b_\theta|$",
            r"$|b_\phi|$",
        ]
    else:
        values = [np.real(comp) for comp in components]

    # Match the canvas and layout used by tools/plot_field.py.
    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    for ax, value, label in zip(axes, values, labels):
        im = ax.tricontourf(triang, value, levels=80, cmap="plasma")
        set_contour_edges_to_face(im)
        draw_boundary(ax, par.ricb, color="white", lw=0.8, alpha=0.9)
        draw_boundary(ax, ut.rcmb, color="white", lw=1.0, alpha=0.95)
        if mantle is not None and ut.resolved_mantle:
            draw_boundary(ax, par.r_surface, color="white", lw=0.8, alpha=0.9)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$s$")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, aspect=70)
    axes[0].set_ylabel(r"$z$")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200)
        print("wrote", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
