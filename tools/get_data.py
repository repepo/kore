import pandas as pd
import numpy as np
import numpy.lib.scimath as scimath
import sys
from os.path import exists


'''
Reads results produced by solve.py or collected by reap.sh into a pandas dataframe
Use as:
python3 $KORE_DIR/tools/get_data.py filename 

filename : prefix used when collecting data via reap.sh (leave empty when reading results directly from solve.py)
'''

def load_results(filename):
    df = pd.DataFrame()
    if exists(filename + '.par'):
        par = pd.read_csv(filename + '.par', sep=' ', names=["Ek", "m", "symm", "ricb", "bci", "bco", "projection",
                                                             "forcing", "Af_cmb", "wf", "magnetic", "Em", "Le2", "N",
                                                             "lmax", "solve_time", "ncpus", "tol", "thermal", "Prandtl",
                                                             "Ra", "compositional", "Schmidt", "Ra_comp", "Af_icb",
                                                             "rc", "h", "mbci", "c", "c1", "mu", "B0_norm",
                                                             "time_scale"])
        df = pd.concat([df, par], axis=1)
    if exists(filename + '.flo'):
        flo = pd.read_csv(filename + '.flo', sep=' ', names=["KP", "KT", "Dint", "rDkin", "iDkin", "rpow", "ipow",
                                                             "Dint1", "Dint2", "Dint3", "rvtorq_cmb", "ivtorq_cmb",
                                                             "rvtorq_icb", "ivtorq_icb"])
        df = pd.concat([df, flo], axis=1)
    if exists(filename + '.mag'):
        mag = pd.read_csv(filename + '.mag', sep=' ', names=["MP", "MT", "ohm_disP", "ohm_disT", "Dohm1", "Dohm2",
                                                             "Dohm3", "rmtorq_cmb", "imtorq_cmb"])
        df = pd.concat([df, mag], axis=1)
    if exists(filename + '.tmp'):
        tmp = pd.read_csv(filename + '.tmp', sep=' ', names=['Dbuoy', 'TE', 'Dtemp', 'Dadv'])
        df = pd.concat([df, tmp], axis=1)
    if exists(filename + '.cmp'):
        cmp = pd.read_csv(filename + '.cmp', sep=' ', names=['Dbuoy_comp', 'CE', 'Dcomp', 'Dadv_comp'])
        df = pd.concat([df, cmp], axis=1)
    if exists(filename + '.eig'):
        err = pd.read_csv(filename + '.eig', sep=' ', names=['resid1', 'resid2'])
        df = pd.concat([df, err], axis=1)
    if exists(filename + '.eig'):
        eig = pd.read_csv(filename + '.eig', sep=' ', names=['rtau', 'itau'])
        df = pd.concat([df, eig], axis=1)
    return df


def load_result():
    df = pd.DataFrame()
    if exists('params.dat'):
        par = pd.read_csv('params.dat', sep=' ', names=["Ek", "m", "symm", "ricb", "bci", "bco", "projection",
                                                        "forcing", "Af_cmb", "wf", "magnetic", "Em", "Le2", "N", "lmax",
                                                        "solve_time", "ncpus", "tol", "thermal", "Prandtl", "Ra",
                                                        "compositional", "Schmidt", "Ra_comp", "Af_icb", "rc", "h",
                                                        "mbci", "c", "c1", "mu", "B0_norm", "tA"])
        df = pd.concat([df, par], axis=1)
    if exists('flow.dat'):
        flo = pd.read_csv('flow.dat', sep=' ', names=["KP", "KT", "Dint", "rDkin", "iDkin", "rpow", "ipow", "Dint1",
                                                      "Dint2", "Dint3", "rvtorq_cmb", "ivtorq_cmb", "rvtorq_icb",
                                                      "ivtorq_icb"])
        df = pd.concat([df, flo], axis=1)
    if exists('magnetic.dat'):
        mag = pd.read_csv('magnetic.dat', sep=' ', names=["MP", "MT", "DohmP", "DohmT", "Dohm1", "Dohm2",
                                                             "Dohm3", "rmtorq", "imtorq"])
        df = pd.concat([df, mag], axis=1)
    if exists('thermal.dat'):
        tmp = pd.read_csv('thermal.dat', sep=' ', names=['Dbuoy', 'TE', 'Dtemp', 'Dadv'])
        df = pd.concat([df, tmp], axis=1)
    if exists('compositional.dat'):
        cmp = pd.read_csv('compositional.dat', sep=' ', names=['Dbuoy_comp', 'CE', 'Dcomp', 'Dadv_comp'])
        df = pd.concat([df, cmp], axis=1)
    if exists('error.dat'):
        err = pd.read_csv('error.dat', sep=' ', names=['resid1', 'resid2'])
        df = pd.concat([df, err], axis=1)
    if exists('eigenvalues.dat'):
        eig = pd.read_csv('eigenvalues.dat', sep=' ', names=['rtau', 'itau'])
        df = pd.concat([df, eig], axis=1)
    return df


# --- INITIALIZATION ---
# load results from the solver(s)
if len(sys.argv) == 2:
    filename = sys.argv[1]
    df = load_results(filename)
else:
    df = load_result()

# --- COMPUTATION ---
# scaling factors
if (df['time_scale'] == 1).all():
    df['scale_factor'] = 1/df['Ek']
elif (df['time_scale'] == 2).all():
    df['scale_factor'] = 1/df['Le']
else:
    df['scale_factor'] = np.ones(len(df))

# kinetic energy parameters
df['KE'] = df['KP'] + df['KT']

df['t2p'] = df['KT']/df['KP']
df['p2t'] = df['KP']/df['KT']

df['Dkin'] = df['rDkin']*df['Ek']*df['scale_factor']
df['Dint'] = df['Dint']*df['Ek']*df['scale_factor']

if (df['Dint1'] != 0).all() or (df['Dint2'] != 0).all() or (df['Dint3'] != 0).all():
    df['Dint_bulk1'] = df['Dint'] - df['Dint1'] - df['Dint2']
    df['Dint_bulk2'] = df['Dint'] - df['Dint1'] - df['Dint3']
del df['Dint1']
del df['Dint2']
del df['Dint3']

df['vtorq_cmb'] = df['rvtorq_cmb'] + 1j*df['ivtorq_cmb']
df['vtorq_icb'] = df['rvtorq_icb'] + 1j*df['ivtorq_icb']

# forcing parameters
if (df['forcing'] == 0).all():
    del df['Af_icb']
    del df['Af_cmb']
    df['freq'] = df['itau']
    df['damp'] = df['rtau']
    df['pss'] = np.zeros(len(df))
    df['pvf'] = np.zeros(len(df))
else:
    df['freq'] = df['wf']
    if (df['forcing'] == 1).all():
        df['pss'] = np.zeros(len(df))
        df['pvf'] = df['rpow']
    elif (df['forcing'] == 7).all():
        df['pvf'] = 0
        df['pss'] = df['rpow']
    elif (df['forcing'] == 8).all():
        df['pss'] = np.zeros(len(df))
        df['pvf'] = df['rpow']
    elif (df['forcing'] == 9).all():
        df['pvf'] = np.zeros(len(df))
        df['pss'] = df['rpow']

del df['wf']
del df['rtau']
del df['itau']

# magnetic parameters
if (df['magnetic'] == 0).all():
    del df['Em']
    del df['Le2']
    del df['mu']
    del df['B0_norm']
    df['Dohm'] = np.zeros(len(df))
else:
    df['Le'] = np.sqrt(df['Le2'])
    df['Pm'] = df['Ek']/df['Em']
    df['Lam'] = df['Le2']/df['Em']

    df['ME'] = df['MP'] + df['MT']
    df['mtorq'] = df['rmtorq'] + 1j*df['imtorq']
    df['Dohm'] = (df['DohmP'] + df['DohmT'])*df['Le2']*df['Em']*df['scale_factor']

    df['o2v'] = df['Dohm']/df['Dint']

    if (df['tA'] == 1).all():
        df['Dkin'] = df['Dkin'] / df['Le']
        df['Dint'] = df['Dint'] / df['Le']

    if (df['Dohm1'] != 0).all() or (df['Dohm2'] != 0).all() or (df['Dohm3'] != 0).all():
        df['Dohm_bulk1'] = df['Dohm'] - df['Dohm1'] - df['Dohm2']
        df['Dohm_bulk2'] = df['Dohm'] - df['Dohm1'] - df['Dohm3']
        df['o2v1'] = df['Dohm_bulk1']/df['Dint_bulk1']
        df['o2v2'] = df['Dohm_bulk2']/df['Dint_bulk2']
    del df['Dohm1']
    del df['Dohm2']
    del df['Dohm3']

# thermal parameters
if (df['thermal'] == 0).all():
    del df['Prandtl']
    del df['Ra']
    del df['rc']
    del df['h']
    df['Dtemp'] = np.zeros(len(df))
else:
    df['Brunt'] = scimath.sqrt(-df['Ra']/df['Prandtl'])*df['Ek']
    df['Dbuoy'] = df['Dbuoy']*df['scale_factor']*(df['Ek']**2)*df['Ra']/df['Prandtl']
    df['Dtemp'] = df['Dtemp']*df['scale_factor']*df['Ek']/df['Prandtl']

# compositional parameters
if (df['compositional'] == 0).all():
    del df['Schmidt']
    del df['Ra_comp']
    df['Dbuoy_comp'] = np.zeros(len(df))
else:
    df['Brunt_comp'] = scimath.sqrt(-df['Ra_comp']/df['Schmidt'])*df['Ek']
    df['Dbuoy_comp'] = df['Dbuoy_comp']*df['scale_factor']*(df['Ek']**2)*df['Ra_comp']/df['Schmidt']
    df['Dcomp'] = df['Dcomp']*df['scale_factor']*df['Ek']/df['Schmidt']

# additional parameters
df['Dtot'] = df['Dint'] + df['Dohm'] + df['Dbuoy'] + df['Dbuoy_comp']
df['Q'] = df['KE']/df['Dtot']

# save results to comma-separated-value file
if len(sys.argv) == 2:
    df.to_csv(filename + '.csv')
else:
    df.to_csv('results.csv')
