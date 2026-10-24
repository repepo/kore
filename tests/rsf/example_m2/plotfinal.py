#!/usr/bin/env python3
'''plotfinal.py SPECTRUM.csv PNG: solved forcing-11 spectrum of the m = 2 example (spectrum.csv from anarsf.py: per configuration
KE and KEin, the kinetic energy below 0.70 R_sun): PSD (total and below 0.70) and the share below 0.70 against omega.'''
import sys, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = list(csv.DictReader(open(sys.argv[1])))
F = np.array([float(r['F_nHz']) for r in R]); o = np.argsort(F); F = F[o]; R = [R[i] for i in o]
res = np.array([r['kind'] == 'res' for r in R])
CUT = [('no_cut', 'no cut', '#2a78d6'), ('full_above_0.74', 'edge: full above 0.74 R☉', '#eb6834'),
       ('full_above_0.76', 'edge: full above 0.76 R☉', '#1baf7a')]
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, facecolor=BG, gridspec_kw={'height_ratios': [2, 1.2]})
for j, th in enumerate(('nu', 'mlt')):
    a0, a1 = ax[0, j], ax[1, j]
    for a in (a0, a1):
        a.set_facecolor(BG); a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        a.axvspan(-334.34, -333.43, color='#d9d6cc', alpha=0.45, lw=0, zorder=0)
        a.axvline(-331.932, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=0)
    for key, lab, c in CUT:
        KE = np.array([float(r['%s|%s|KE' % (th, key)]) for r in R]); KI = np.array([float(r['%s|%s|KEin' % (th, key)]) for r in R])
        a0.semilogy(F, KE, color=c, lw=1.8, label=lab)
        a0.semilogy(F, KI, color=c, lw=1.2, ls=(0, (4, 2)))
        a0.semilogy(F[res], KE[res], 'o', ms=2.6, color=c, mec='none')
        a1.plot(F, KI/KE, color=c, lw=1.8)
        a1.plot(F[res], (KI/KE)[res], 'o', ms=2.6, color=c, mec='none')
    a1.axhline(0.211, color=INK2, lw=0.8, ls=(0, (1, 2)))
    a1.set_title("dotted: the Rossby eigenmode's own share (0.21)", loc='left', fontsize=8.5, color=INK2)
    a0.text(-333.885, a0.get_ylim()[1], 'RZ modes\n(94, all KE\nbelow 0.70 R☉)', ha='center', va='top', fontsize=8, color=INK2)
    a0.text(-331.70, a0.get_ylim()[1], 'Rossby (CZ)\nω −331.93, σ −0.26', ha='left', va='top', fontsize=8, color=INK2)
    a0.set_title("rsf_theta '%s'" % th, loc='left', fontsize=10, color=INK)
    a0.set_ylabel('kinetic energy per unit ω (PSD, kore units)\nsolid: total   dashed: below 0.70 R☉')
    a1.set_ylabel('share of KE below 0.70 R☉'); a1.set_ylim(0, 0.7)
    a1.set_xlabel('ω (nHz, frame 456 nHz)'); a1.set_xlim(-336.1, -329.4)
ax[0, 0].legend(frameon=False, fontsize=8.5, loc='lower right')
fig.suptitle('Forcing 11, m = 2, symm −1, full sphere (N 896): solved response at all 91 frequencies, 16 realisations per point '
             '(same draws at every frequency); dots = solved on an RZ-mode resonance', fontsize=9, color=INK2, x=0.01, ha='left')
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=150, facecolor=BG)
print('wrote', sys.argv[2])
