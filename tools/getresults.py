'''
Reads the tables collected with tools/reap.sh (or the .dat files of the current folder) into python, by column name.
In IPython:

    %run -i tools/getresults.py somename

where somename is the prefix used when creating and collecting the results (tools/dodirs.sh, tools/reap.sh). It sets df,
a pandas DataFrame with one row per solution and the named columns of tools/get_data.py (docs/output.md describes them),
and a few numpy arrays from it: ricb, bci, bco, N, lmax, ViscosD; KE, KP, KT, Dkin, ldom, lwidth, lconv, t2p = KT/KP, and
resid1 (residσ); for eigenvalue problems sigma and omega, the real and imaginary parts of the eigenvalues. Any other column
is df['name'].

NEW (2026-11-08): by name, with tools/get_data.py; it read fixed column numbers of an older params.dat.
'''
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import get_data

df = get_data.load_results(sys.argv[1]) if len(sys.argv) == 2 else get_data.load_result()
if df.empty:
    sys.exit('getresults.py: no table found (%s)' % ('%s.par, ...' % sys.argv[1] if len(sys.argv) == 2 else 'params.dat, ...'))
col = lambda name: df[name].to_numpy() if name in df else np.full(len(df), np.nan)

ricb, bci, bco = col('ricb'), col('bci'), col('bco')
N, lmax, ViscosD = col('N'), col('lmax'), col('ViscosD')
KE, KP, KT, Dkin = col('KE'), col('KP'), col('KT'), col('Dkin')
ldom, lwidth, lconv = col('ldom'), col('lwidth'), col('lconv')
t2p = KT/KP
resid1 = col('resid1')
if 'Re_lambda' in df:
    sigma, omega = col('Re_lambda'), col('Im_lambda')
