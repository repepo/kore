#!/usr/bin/env python3
'''
usage: figs.py WORKDIR [OUTDIR]   (matplotlib; OUTDIR defaults to WORKDIR)

Plots the results of triana2019.py (WORKDIR/fig2.json, fig3.json, fits.json, whichever exist; fig3b.json too if present) on
the axes of Triana, Rekier, Trinh & Dehant (2019): kore_fig2.png (their Fig. 2), kore_fig3.png (Fig. 3), kore_fig10_4.png
(Figs. 10 and 4), kore_fcn_fits.png (Figs. 6-7 and eqs. 28-29), kore_cw_fits.png (Fig. 8 and eqs. 32-33), and the numbers
behind them in numbers.txt. NEW (2026-11-09).

Two kinds of eigenpairs returned by the shift-invert solver are dropped. (1) Pairs whose angular momentum budget fails
(wobble.budget, residual > RESID = 1e-6): not eigenpairs of the coupled problem. (2) A ladder of under-resolved short waves
at the truncation edge (dampings Ek k^2 for Bessel zeros at l ~ lmax), which hardly feel the mantle: eigenvalues that move
by less than 1e-6 between q = 0 and q = 1, and anything within 3e-5 of them at larger q. Some of these pass the budget at
larger q, so (1) alone does not remove them.
The paper's approximations, with lambda = sigma + i omega and omega_* = 2/(1 + (1 - alpha)^2):
  FCN  omega - omega_* = b_w + q m_w,  sigma = b_s + q m_s     (eqs. 28, 29)
  CW   omega = alpha[-1 - 1.02e-5 - 4.75e-3 E^1/2 + 0.237 E + q(-1 + 1.21e-5 + 3.292 E^1/2 + 12.53 E)],
       sigma = alpha q[-3.579 E^1/2 - 17.37 E + q(-3.579 E^1/2 + 7.511 E)]     (eqs. 32, 33)
'''
import os, sys, json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

W = os.path.abspath(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else W
os.makedirs(OUT, exist_ok=True)
RESID = 1e-6


def ws(a):
    return 2/(1 + (1 - a)**2)


def fcn_fit(E, a, q):
    mw = 0.599*E**0.543 + a - 0.427*a**2*E**-0.499
    bw = 0.558*E**0.539 + 2.915*a*E**0.527
    ms = -2.705*E**0.502 + 11.58*a*E**0.553 + 0.088*a**2*E**-0.37
    bs = -2.630*E**0.5 - 25.40*E + a*(-0.566*E**0.5 + 299.3*E)
    return ws(a) + bw + q*mw, bs + q*ms


def cw_fit(E, a, q):
    return (a*(-1 - 1.02e-5 - 4.75e-3*E**0.5 + 0.237*E + q*(-1 + 1.21e-5 + 3.292*E**0.5 + 12.53*E)),
            a*q*(-3.579*E**0.5 - 17.37*E + q*(-3.579*E**0.5 + 7.511*E)))


def load(name):
    f = os.path.join(W, name + '.json')
    if not os.path.exists(f):
        return None
    cases = json.load(open(f))
    for c in cases:
        if 'error' in c:
            print('%s: %s failed: %s' % (name, c.get('name'), c['error'][-300:]))
    return [c for c in cases if 'error' not in c]


def lam(m):
    return complex(*m['lam'])


def derived(m):
    '''ratio fields from the raw records (fig3b.json has only the raw ones)'''
    if 'KE' not in m:
        return m
    m = dict(m)
    m.setdefault('ratio_srf', m['KM_srf']/m['KE_srf'])
    m.setdefault('T11', m['KE_T11']/m['KE'])
    m.setdefault('DP_Dkin', m['DP']/m['Dkin'])
    v = abs(complex(*m['viscous']))
    m.setdefault('torque_ratio_reduced', abs(complex(*m['pressure']))/v)
    m.setdefault('torque_ratio', abs(complex(*m['pressure']) + complex(*m['centrifugal']))/v)
    return m


def q_of(p, a):
    if not p.get('wobble'):
        return 0.0
    ak = (1 - a)**(-1/3)
    return round((4*np.pi/15)*(1 - a)*(2 - 2*a + a**2)*ak**5/p['MoIX_M'], 9)


def fig3_modes():
    '''{q: [mode]} from fig3.json (first search) and fig3b.json (corrected search), E 1e-6, alpha 1e-4,
    with duplicates merged, the budget filter applied (q > 0), and the truncation-edge ladder removed:
    eigenvalues that move by less than 1e-6 between q = 0 and q = 1 hardly feel the mantle (the genuine inertial
    modes move by 2e-5 or more); they and anything within 3e-5 of them at larger q are dropped.'''
    raw = defaultdict(list)
    for name in ('fig3', 'fig3b'):   # fig3b.json: the rerun of 2026-11-09 with the targets now in triana2019.py
        for c in load(name) or []:
            for s in c['solves']:
                raw[float(s['tag'].split('_')[0][1:])] += s['modes']
    out = {}
    for q, ms in raw.items():
        keep = []
        for m in sorted(map(derived, ms), key=lambda m: m.get('resid', 0)):
            if q > 0 and m['resid'] > RESID:
                continue
            if any(abs(lam(m) - lam(k)) < 1e-9 for k in keep):
                continue
            keep.append(m)
        out[q] = keep
    l0 = [lam(m) for m in out[0.0]]
    l1 = [lam(m) for m in raw[1.0]]
    ladder = [l for l in l0 if min(abs(l - x) for x in l1) < 1e-6]
    for q in out:
        out[q] = [m for m in out[q] if min(abs(lam(m) - l) for l in ladder) > 3e-5]
    return out, ladder


plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'savefig.dpi': 220, 'savefig.bbox': 'tight',
                     'axes.grid': False, 'legend.fontsize': 7.5, 'legend.frameon': True, 'legend.framealpha': 0.9})
C1, C2, C3, INK = '#2a78d6', '#eb6834', '#1baf7a', '#3d3d3a'
CM = 'plasma_r'        # the paper's colour map for sigma/sqrt(E): dark = weakly damped, yellow = strongly damped
log = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log.append(s)


def fig2():
    d = load('fig2')[0]
    E, a = d['E'], d['alpha']
    modes = []
    for s in d['solves']:
        for m in map(derived, s['modes']):
            if m['resid'] < RESID and not any(abs(lam(m) - lam(k)) < 1e-9 for k in modes):
                modes.append(m)
    L = np.array([lam(m) for m in modes]); R = np.array([m['ratio_srf'] for m in modes])
    S = L.real/np.sqrt(E)
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.0), gridspec_kw={'width_ratios': [3, 1]})
    norm2 = Normalize(-49, -2)
    for k, x in enumerate(ax):
        sc = x.scatter(L.imag, R, c=S, cmap=CM, norm=norm2, s=14 if k == 0 else 22, edgecolors='none', zorder=3)
        x.set_yscale('log'); x.set_xlabel(r'$\omega$')
    ax[0].set_xlim(-2.02, 2.02); ax[0].set_ylim(1e-18, 1e4)
    ax[0].set_ylabel(r'$K^{\rm SRF}_{\rm Mantle}/K^{\rm SRF}_{\rm Fluid}$')
    iC = int(np.argmax(np.where(L.imag < 0.5, R, -1))); iF = int(np.argmax(np.where(L.imag > 0.5, R, -1)))
    ax[0].annotate('CW', (L[iC].imag, R[iC]), xytext=(-60, -14), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=8)
    ax[0].annotate('FCN', (L[iF].imag, R[iF]), xytext=(-60, 4), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=8)
    ax[0].text(-1.9, 1e-16, 'kore searched only near the CW and the FCN (4 shift-invert targets, 12 modes each)', fontsize=7.5,
               color=INK)
    ax[1].set_xlim(0.93, 1.07); ax[1].set_ylim(1e-9, 1)
    ax[1].set_title(r'zoom near the FCN', fontsize=8.5)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.03); cb.set_label(r'$\sigma/\sqrt{E}$')
    ax[0].set_title(r'kore: mantle/fluid kinetic energy ratio, $q$ = 0.1, $\alpha$ = $10^{-4}$, $E$ = $10^{-6}$')
    fig.savefig(os.path.join(OUT, 'kore_fig2.png')); plt.close(fig)
    mF, mC = modes[iF], modes[iC]
    say('fig2 CW', L[iC], 'ratio %.4g' % R[iC], '| FCN', L[iF], 'ratio %.4g T11 %.4f torque %.4f reduced %.4f'
        % (R[iF], mF['T11'], mF['torque_ratio'], mF['torque_ratio_reduced']))
    wc, sc_ = cw_fit(E, a, 0.1); wf, sf = fcn_fit(E, a, 0.1)
    say('fig2 CW fit', complex(sc_, wc), 'FCN fit', complex(sf, wf))


def fig3():
    F3, ladder = fig3_modes()
    say('fig3 ladder eigenvalues dropped:', len(ladder))
    W0, W1 = 1.0, 1.018
    rows = []          # (q, mode) in the paper's window
    for q, ms in F3.items():
        for m in ms:
            if W0 <= lam(m).imag <= W1:
                rows.append((q, m))
    fcn = {}
    for q in sorted(F3):
        ms = [m for qq, m in rows if qq == q]
        if q == 0:
            wf, sf = fcn_fit(1e-6, 1e-4, 0.0)
            fcn[q] = min(ms, key=lambda m: abs(lam(m) - complex(sf, wf)))
        elif q <= 14:
            fcn[q] = max(ms, key=lambda m: m['T11'])
    Q = np.array([q for q, _ in rows]); LL = np.array([lam(m) for _, m in rows]); SS = LL.real/1e-3
    norm3 = Normalize(-90, -2)


    def ring(x, xs, ys):
        x.scatter(xs, ys, s=46, facecolors='none', edgecolors='k', linewidths=0.9, zorder=4)


    fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.3), gridspec_kw={'wspace': 0.42})
    ax[0].scatter(Q, LL.imag, c=SS, cmap=CM, norm=norm3, s=14, zorder=3)
    ax[0].set_ylim(1.0, 1.018); ax[0].set_ylabel(r'$\omega$'); ax[0].set_title(r'Frequency $\omega$')
    ax[1].scatter(Q, SS, c=SS, cmap=CM, norm=norm3, s=14, zorder=3)
    ax[1].set_ylim(-90, 0); ax[1].set_ylabel(r'$\sigma/\sqrt{E}$'); ax[1].set_title(r'Scaled damping $\sigma/\sqrt{E}$')
    pos = Q > 0
    ax[2].scatter(Q[pos], [m['ratio_srf'] for q, m in rows if q > 0], c=SS[pos], cmap=CM, norm=norm3, s=14, zorder=3)
    ax[2].set_yscale('log'); ax[2].set_ylim(1e-9, 10); ax[2].set_ylabel(r'$K^{\rm SRF}_{\rm Mantle}/K^{\rm SRF}_{\rm Fluid}$')
    ax[2].set_title('Mantle/fluid kinetic energy ratio')
    sc = ax[3].scatter(Q[pos], [m['T11'] for q, m in rows if q > 0], c=SS[pos], cmap=CM, norm=norm3, s=14, zorder=3)
    ax[3].set_yscale('log'); ax[3].set_ylim(1e-5, 1.4); ax[3].set_ylabel(r'$K^{\rm MF}_{11}/K^{\rm MF}_{\rm Fluid}$')
    ax[3].set_title(r'Toroidal $l$ = 1, $m$ = 1 kinetic energy')
    qf = np.array(sorted(fcn)); lf = np.array([lam(fcn[q]) for q in qf])
    ring(ax[0], qf, lf.imag); ring(ax[1], qf, lf.real/1e-3)
    ring(ax[2], qf[qf > 0], [fcn[q]['ratio_srf'] for q in qf if q > 0]); ring(ax[3], qf[qf > 0], [fcn[q]['T11'] for q in qf if q > 0])
    ax[0].annotate('FCN', (2, lam(fcn[2.0]).imag), xytext=(25, -4), textcoords='offset points', fontsize=8,
                   arrowprops=dict(arrowstyle='->', lw=0.8))
    for x in ax:
        x.set_xlim(-0.3, 30.3); x.set_xlabel('$q$')
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.02); cb.set_label(r'$\sigma/\sqrt{E}$')
    fig.savefig(os.path.join(OUT, 'kore_fig3.png')); plt.close(fig)
    for q in qf:
        m = fcn[q]
        say('fig3 FCN q %-4g' % q, lam(m), 's/sqrtE %.2f' % (lam(m).real/1e-3),
            *('%s %.4g' % (k, m[k]) for k in ('T11', 'ratio_srf', 'DP_Dkin', 'torque_ratio', 'torque_ratio_reduced') if k in m))

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 3.3), gridspec_kw={'wspace': 0.4})
    ax[0].scatter(Q[pos], [m['DP_Dkin'] for q, m in rows if q > 0], c=SS[pos], cmap=CM, norm=norm3, s=14, zorder=3)
    ax[0].set_ylim(0, 7); ax[0].set_ylabel(r'$D_P/D_{\rm kin}$'); ax[0].set_title(u'Poincaré power to viscous dissipation ratio')
    sc = ax[1].scatter(Q[pos], [m['torque_ratio'] for q, m in rows if q > 0], c=SS[pos], cmap=CM, norm=norm3, s=14,
                       zorder=3)
    ax[1].set_ylim(0, 0.235); ax[1].set_ylabel(r'$\Gamma_p/\Gamma_\nu$'); ax[1].set_title('Pressure torque to viscous torque ratio')
    ring(ax[0], qf[qf > 0], [fcn[q]['DP_Dkin'] for q in qf if q > 0])
    ring(ax[1], qf[qf > 0], [fcn[q]['torque_ratio'] for q in qf if q > 0])
    for x in ax[:2]:
        x.set_xlim(-0.3, 30.3); x.set_xlabel('$q$')
    cb = fig.colorbar(sc, ax=ax[:2], pad=0.01, fraction=0.025); cb.set_label(r'$\sigma/\sqrt{E}$')
    allq = [(q, m) for q, ms in F3.items() for m in ms if W0 <= lam(m).imag <= W1]
    qq = np.array([q for q, _ in allq]); l3 = np.array([lam(m) for _, m in allq])
    sc2 = ax[2].scatter(l3.imag, l3.real/1e-3, c=qq, cmap='viridis', norm=Normalize(0, 30), s=16, zorder=3)
    ax[2].plot(lf.imag, lf.real/1e-3, '-', color=INK, lw=0.7, zorder=2)
    ax[2].set_xlim(1.0, 1.018); ax[2].set_ylim(-90, 0); ax[2].set_xlabel(r'$\omega$'); ax[2].set_ylabel(r'$\sigma/\sqrt{E}$')
    ax[2].set_title(r'$\alpha = 10^{-4}$, $E = 10^{-6}$ (line: FCN branch, $q$ = 0 to 14)')
    cb = fig.colorbar(sc2, ax=ax[2], pad=0.01, fraction=0.05); cb.set_label('$q$')
    fig.savefig(os.path.join(OUT, 'kore_fig10_4.png')); plt.close(fig)
    # torque ratio: kore against values read off the paper's Fig. 10b at q = 30 (top lines) and the FCN plateau
    m30 = sorted([m for q, m in rows if q == 30.0], key=lambda m: -m['torque_ratio'])
    say('fig10 q=30 torque ratios (largest first):', ['%.4f@%.4f' % (m['torque_ratio'], lam(m).imag) for m in m30[:6]])


def fits():
    FT = {}   # (E, a) -> {(kind, q): lam}
    for c in load('fits'):
        E_, a_ = c['E'], c['alpha']
        for s in c['solves']:
            kind, qs = s['tag'].split('_q'); q = float(qs)
            ms = [derived(m) for m in s['modes']]
            if kind == 'fcn':
                fw, fs = fcn_fit(E_, a_, q)
                ok = [m for m in ms if q == 0 or m['resid'] < RESID]
                m = min(ok, key=lambda m: abs(lam(m) - complex(fs, fw))) if ok else None
            else:
                ok = [m for m in ms if m['resid'] < RESID]
                m = max(ok, key=lambda m: m['ratio_srf']) if ok else None
            if m is not None:
                FT.setdefault((E_, a_), {})[(kind, q)] = lam(m)
    Es = sorted(set(k[0] for k in FT)); A1, A2 = 1e-6, 10**-4.6
    A2 = [k[1] for k in FT if abs(np.log10(k[1]) + 4.6) < 1e-6][0]
    Eg = np.logspace(np.log10(5e-9), np.log10(2e-4), 100)


    def slope(d, kind):
        q = np.array([0, 0.05, 0.1]); l = np.array([d[(kind, x)] for x in q])
        return np.polyfit(q, l.imag, 1)[0], np.polyfit(q, l.real, 1)[0], np.polyfit(q, l.real, 2)[0]


    coef = []
    for E_ in Es:
        d1, d2 = FT[(E_, A1)], FT[(E_, A2)]
        bw1 = d1[('fcn', 0.0)].imag - ws(A1); bw2 = d2[('fcn', 0.0)].imag - ws(A2)
        bs1 = d1[('fcn', 0.0)].real; bs2 = d2[('fcn', 0.0)].real
        b1w = (bw2 - bw1)/(A2 - A1); b1s = (bs2 - bs1)/(A2 - A1)
        b0w = bw1 - A1*b1w; b0s = bs1 - A1*b1s
        mw1, ms1, curv = slope(d1, 'fcn')
        m0w = mw1 - A1          # m_1,omega = 1 (their Fig. 6 inset); the alpha^2 term is below 1e-4 of m0 here
        m0s = ms1
        coef.append((E_, b0w, b1w, b0s, b1s, m0w, m0s))
        say('fcn coef E %.0e  b0w %.4e  b1w %.4e  b0s %.4e  b1s %+.3e  m0w %.4e  m0s %.4e  (q^2 curvature of sigma %.1e)'
            % (E_, b0w, b1w, b0s, b1s, m0w, m0s, curv))
    coef = np.array(coef)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    x = ax[0]
    x.loglog(coef[:, 0], coef[:, 1], 'o', color=C1, ms=6, label=r'$b_{0,\omega}$ kore')
    x.loglog(Eg, 0.558*Eg**0.539, '--', color=C1, lw=1, label=r'$0.558\,E^{0.539}$, fit')
    x.loglog(Eg, 0.258*Eg**0.5, '-', color=C1, lw=0.8, alpha=0.6, label=r'$0.258\,E^{1/2}$, theory')
    x.loglog(coef[:, 0], coef[:, 2], 's', color=C2, ms=6, label=r'$b_{1,\omega}$ kore')
    x.loglog(Eg, 2.915*Eg**0.527, '--', color=C2, lw=1, label=r'$2.915\,E^{0.527}$, fit')
    x.loglog(Eg, 1.533*Eg**0.5, '-', color=C2, lw=0.8, alpha=0.6, label=r'$1.533\,E^{1/2}$, theory')
    x.set_title(r'$b_\omega = b_{0,\omega} + b_{1,\omega}\,\alpha$'); x.set_ylabel(r'$b_{0,\omega},\ b_{1,\omega}$')
    x = ax[1]
    x.loglog(coef[:, 0], -coef[:, 3], 'o', color=C1, ms=6, label=r'$|b_{0,\sigma}|$ kore')
    x.loglog(Eg, 2.630*Eg**0.5 + 25.40*Eg, '--', color=C1, lw=1, label=r'$2.630\,E^{1/2}+25.40\,E$, fit')
    x.loglog(Eg, 2.620*Eg**0.5, '-', color=C1, lw=0.8, alpha=0.6, label=r'$2.620\,E^{1/2}$, theory')
    neg = coef[:, 4] < 0
    x.loglog(coef[neg, 0], abs(coef[neg, 4]), 's', color=C2, ms=6, label=r'$|b_{1,\sigma}|$ kore ($b_{1,\sigma}<0$)')
    x.loglog(coef[~neg, 0], abs(coef[~neg, 4]), 's', color=C2, ms=6, mfc='none', label=r'$|b_{1,\sigma}|$ kore ($b_{1,\sigma}>0$)')
    Eh = Eg[Eg < 2e-6]
    x.loglog(Eh, 0.566*Eh**0.5 - 299.3*Eh, '--', color=C2, lw=1, label=r'$0.566\,E^{1/2}-299.3\,E$, fit')
    x.loglog(Eg, 0.853*Eg**0.5, '-', color=C2, lw=0.8, alpha=0.6, label=r'$0.853\,E^{1/2}$, theory')
    x.set_title(r'$b_\sigma = b_{0,\sigma} + b_{1,\sigma}\,\alpha$'); x.set_ylabel(r'$b_{0,\sigma},\ b_{1,\sigma}$')
    x = ax[2]
    x.loglog(coef[:, 0], coef[:, 5], 'o', color=C1, ms=6, label=r'$m_{0,\omega}$ kore')
    x.loglog(Eg, 0.599*Eg**0.543, '--', color=C1, lw=1, label=r'$0.599\,E^{0.543}$, fit')
    x.loglog(coef[:, 0], -coef[:, 6], 's', color=C2, ms=6, label=r'$-m_{0,\sigma}$ kore')
    x.loglog(Eg, 2.705*Eg**0.502, '--', color=C2, lw=1, label=r'$2.705\,E^{0.502}$, fit')
    x.set_title(r'$m_{0,\omega}$ and $m_{0,\sigma}$ (slopes in $q$ at $\alpha=10^{-6}$)'); x.set_ylabel(r'$m_{0,\omega},\ -m_{0,\sigma}$')
    for x in ax[:3]:
        x.set_xlim(5e-9, 2e-4); x.set_xlabel('$E$'); x.legend(loc='upper left', fontsize=6.5)
    # relative error of eqs 28-29 for every run
    x = ax[3]
    for i, A in enumerate((A1, A2)):
        for j, q in enumerate((0.0, 0.05, 0.1)):
            e_ = np.array([E_ for E_ in Es if ('fcn', q) in FT[(E_, A)]])
            l_ = np.array([FT[(E_, A)][('fcn', q)] for E_ in e_])
            fw, fs = fcn_fit(e_, A, q)
            col = (C1, C2, C3)[j]
            x.loglog(e_*(1 + 0.04*(i - 0.5)), abs(l_.imag - fw)/abs(fw - ws(A)), 'o' if i == 0 else 'D', color=col,
                     mfc=col if i == 0 else 'none', ms=5, ls='-', lw=0.6)
            x.loglog(e_*(1 + 0.04*(i - 0.5)), abs(l_.real - fs)/abs(fs), 's' if i == 0 else 'v', color=col,
                     mfc=col if i == 0 else 'none', ms=5, ls=':', lw=0.6)
    hl = [Line2D([], [], color=c, lw=2) for c in (C1, C2, C3)] + \
         [Line2D([], [], color=INK, marker='o', ls='-', lw=0.6), Line2D([], [], color=INK, marker='s', ls=':', lw=0.6),
          Line2D([], [], color=INK, marker='o', ls='none'), Line2D([], [], color=INK, marker='D', mfc='none', ls='none')]
    x.legend(hl, ['$q$ = 0', '$q$ = 0.05', '$q$ = 0.1', r'$|\Delta\omega|/|\omega_{\rm fit}-\omega_*|$',
                  r'$|\Delta\sigma|/|\sigma_{\rm fit}|$', r'$\alpha = 10^{-6}$', r'$\alpha = 10^{-4.6}$'],
             fontsize=6.3, loc='lower right', ncol=2)
    x.set_ylim(1e-7, 0.5); x.set_xlim(5e-9, 2e-4); x.set_xlabel('$E$')
    x.set_title('kore against eqs. (28)-(29) in full'); x.set_ylabel('relative difference')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'kore_fcn_fits.png')); plt.close(fig)
    for A in (A1, A2):
        for q in (0.0, 0.05, 0.1):
            for E_ in Es:
                l_ = FT[(E_, A)][('fcn', q)]; fw, fs = fcn_fit(E_, A, q)
                say('fcn eq28-29 a %.2e q %.2f E %.0e  dw/(w-w*) %+.2e  dsigma/sigma %+.2e'
                    % (A, q, E_, (l_.imag - fw)/(fw - ws(A)), (l_.real - fs)/fs))

    cw = []
    for (E_, A), d in sorted(FT.items()):
        if ('cw', 0.05) in d and ('cw', 0.1) in d:
            q1, q2 = 0.05, 0.1
            y1, y2 = d[('cw', q1)]/A, d[('cw', q2)]/A
            m1w = (y2.imag - y1.imag)/(q2 - q1); m0w = y1.imag - q1*m1w
            # sigma/alpha = m1 q + m2 q^2
            M = np.array([[q1, q1**2], [q2, q2**2]]); m1s, m2s = np.linalg.solve(M, [y1.real, y2.real])
            cw.append((E_, A, m0w, m1w, m1s, m2s))
            say('cw coef E %.0e a %.2e  m0w %.7f  m1w %.6f  m1s %.4e  m2s %.4e' % (E_, A, m0w, m1w, m1s, m2s))
    cw = np.array(cw)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    for i, A in enumerate((A1, A2)):
        s_ = np.isclose(cw[:, 1], A); mk = dict(ms=6, mfc=None if i == 0 else 'none')
        lab = r' kore, $\alpha=10^{%s}$' % ('-6' if i == 0 else '-4.6')
        ax[0].semilogx(cw[s_, 0], cw[s_, 2], 'o' if i == 0 else 'D', color=C1, label=r'$m_{0,\omega}$' + lab, **mk)
        ax[0].semilogx(cw[s_, 0], cw[s_, 3], 's' if i == 0 else 'v', color=C2, label=r'$m_{1,\omega}$' + lab, **mk)
        ax[1].loglog(cw[s_, 0], -cw[s_, 4], 's' if i == 0 else 'v', color=C2, label=r'$-m_{1,\sigma}$' + lab, **mk)
        ax[1].loglog(cw[s_, 0], -0.5*cw[s_, 5], 'o' if i == 0 else 'D', color=INK, label=r'$-\frac{1}{2}m_{2,\sigma}$' + lab, **mk)
    ax[0].semilogx(Eg, -1 - 1.02e-5 - 4.75e-3*Eg**0.5 + 0.237*Eg, '--', color=C1, lw=1, label='fit')
    ax[0].semilogx(Eg, -1 + 1.21e-5 + 3.292*Eg**0.5 + 12.53*Eg, '--', color=C2, lw=1, label='fit')
    ax[0].set_ylim(-1.001, -0.964); ax[0].set_title(r'$m_\omega = m_{0,\omega} + m_{1,\omega}\,q$')
    ax[0].set_ylabel(r'$m_{0,\omega},\ m_{1,\omega}$')
    ax[1].loglog(Eg, 3.579*Eg**0.5 + 17.37*Eg, '--', color=C2, lw=1, label='fit')
    ax[1].loglog(Eg, 0.5*(3.579*Eg**0.5 - 7.511*Eg), '--', color=INK, lw=1, label='fit')
    ax[1].set_title(r'$m_\sigma = m_{1,\sigma}\,q + m_{2,\sigma}\,q^2$'); ax[1].set_ylabel(r'$m_{1,\sigma},\ m_{2,\sigma}$')
    for x in ax[:2]:
        x.set_xlim(5e-9, 2e-4); x.set_xlabel('$E$'); x.legend(fontsize=6.3, loc='upper left')
    x = ax[2]
    for i, A in enumerate((A1, A2)):
        for j, q in enumerate((0.05, 0.1)):
            e_ = np.array([E_ for E_ in Es if ('cw', q) in FT[(E_, A)]])
            l_ = np.array([FT[(E_, A)][('cw', q)] for E_ in e_])
            fw, fs = cw_fit(e_, A, q); col = (C2, C3)[j]
            x.loglog(e_, abs(l_.imag - fw)/abs(fw), 'o' if i == 0 else 'D', color=col, mfc=col if i == 0 else 'none', ms=5,
                     ls='-', lw=0.6)
            x.loglog(e_, abs(l_.real - fs)/abs(fs), 's' if i == 0 else 'v', color=col, mfc=col if i == 0 else 'none', ms=5,
                     ls=':', lw=0.6)
    hl = [Line2D([], [], color=c, lw=2) for c in (C2, C3)] + \
         [Line2D([], [], color=INK, marker='o', ls='-', lw=0.6), Line2D([], [], color=INK, marker='s', ls=':', lw=0.6)]
    x.legend(hl, ['$q$ = 0.05', '$q$ = 0.1', r'$|\Delta\omega|/|\omega_{\rm fit}|$', r'$|\Delta\sigma|/|\sigma_{\rm fit}|$'],
             fontsize=6.5, loc='lower right')
    x.set_xlim(5e-9, 2e-4); x.set_ylim(3e-7, 0.2); x.set_xlabel('$E$'); x.set_ylabel('relative difference')
    x.set_title(r'kore against eqs. (32)-(33); filled $\alpha=10^{-6}$, open $10^{-4.6}$')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'kore_cw_fits.png')); plt.close(fig)
    for (E_, A), d in sorted(FT.items()):
        for q in (0.05, 0.1):
            if ('cw', q) in d:
                fw, fs = cw_fit(E_, A, q); l_ = d[('cw', q)]
                say('cw eq32-33 a %.2e q %.2f E %.0e  dw/w %+.2e  ds/s %+.2e' % (A, q, E_, (l_.imag - fw)/fw, (l_.real - fs)/fs))


if load('fig2'):
    fig2()
if load('fig3') or load('fig3b'):
    fig3()
if load('fits'):
    fits()
open(os.path.join(OUT, 'numbers.txt'), 'w').write('\n'.join(log) + '\n')
