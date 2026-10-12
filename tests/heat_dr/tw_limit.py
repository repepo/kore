#!/usr/bin/env python3
# Small-amplitude limit of the exact thermal wind: H(thermal_wind = 2) - H(thermal_wind = 1) = O(amplitude^2) while
# H = O(amplitude), so the relative difference must fall like the amplitude. H are heat_check.py's matrix-side pairings
# (heat_check_H.npy, same random fields in every run of the same configuration).
# usage: tw_limit.py AMP1 TW1_DIR TW2_DIR [AMP2 TW1_DIR TW2_DIR ...]
import sys, numpy as np
a = sys.argv[1:]; prev = None
print('%-10s %-14s %-14s %s' % ('amplitude', 'max |H1|', 'max |H2 - H1|', 'rel. diff / amplitude'))
for k in range(0, len(a), 3):
    amp = float(a[k]); H1 = np.load(a[k + 1] + '/heat_check_H.npy'); H2 = np.load(a[k + 2] + '/heat_check_H.npy')
    rel = np.max(abs(H2 - H1))/np.max(abs(H1))
    print('%-10g %-14.3e %-14.3e %.3f' % (amp, np.max(abs(H1)), np.max(abs(H2 - H1)), rel/abs(amp)))
