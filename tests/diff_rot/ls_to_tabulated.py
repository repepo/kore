'''ls_to_tabulated.py DIR OUT.npz: Convert the Larson & Schou 2018 rot2d.hmiv72d.ave table (151 radii x 49 latitudes 90..0 deg) to the
layout diff_rot_type = 'tabulated' reads (that of data/diff_rot.npz of Mukhopadhyay et al. 2025): ome[theta, r] in nHz with theta 0..pi (symmetric extension), r ascending.'''
import sys, numpy as np
LS, out = sys.argv[1], sys.argv[2]
tab = np.loadtxt(LS + '/rot2d.hmiv72d.ave'); rm = np.loadtxt(LS + '/rmesh.orig')
r = rm[::4][:tab.shape[0]]; lat = 90 - 1.875*np.arange(tab.shape[1])          # descending latitude 90..0
north = tab[:, :]                                                              # [r, lat 90..0] = colatitude 0..90
theta_n = np.deg2rad(90 - lat)                                                 # 0..pi/2 ascending
ome_n = north.T                                                                # [theta, r]
theta = np.concatenate([theta_n, np.pi - theta_n[-2::-1]])                     # 0..pi, equator once
ome = np.concatenate([ome_n, ome_n[-2::-1, :]], axis=0)
o = np.argsort(r); r, ome = r[o], ome[:, o]
np.savez(out, ome=ome, r=r, theta=theta)
print('wrote', out, 'ome', ome.shape, 'r', r[0], r[-1], 'theta', theta[0], theta[-1], 'nHz range', ome[:, (r>0.7)&(r<0.99)].min(), ome[:, (r>0.7)&(r<0.99)].max())
