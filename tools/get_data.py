import pandas as pd
import sys
from os.path import exists


'''
Reads results produced by spin_doctor.py or collected by reap.sh into a pandas dataframe, and saves it as csv
Use as:
python3 $KORE_DIR/tools/get_data.py filename

filename : prefix used when collecting data via reap.sh (leave empty when reading the .dat files of the current folder)
'''

# FIX (2026-11-02): the column names, in the order spin_doctor.py writes them. params.dat had 58 names for its 67
# columns and flow.dat 19 for 20: pandas then made the extra leading columns the index and shifted every label (the
# column "KE" held KP). The tables are read by read_table, which labels as many columns as a file has.
PARAMS = ['hydro', 'magnetic', 'thermal', 'compositional', 'Ek', 'm', 'symm', 'ricb', 'bci', 'bco', 'forcing',
          'forcing_frequency', 'forcing_amplitude_cmb', 'forcing_amplitude_icb', 'projection', 'B0type', 'beta_actual',
          'B0_l', 'innercore_mag_bc', 'c_icb', 'c1_icb', 'mantle_mag_bc', 'c_cmb', 'c1_cmb', 'mu', 'Em', 'Le2',
          'B0_norm', 'Etherm', 'heating', 'BV2', 'rc', 'h', 'rsy', 'bci_thermal', 'bco_thermal', 'Ecomp',
          'compositional_background', 'BV2_comp', 'rcc', 'hc', 'rsyc', 'bci_compositional', 'bco_compositional',
          'OmgTau', 'ncpus', 'N', 'lmax', 'runtime', 'mu_i2o', 'sigma_i2o', 'aux1', 'aux2', 'rotdyn', 'MoIZ_M',
          'MoIZ_IC', 'OmgtauIC', 'gTorque', 'bgflow', 'bgflow_Ro', 'B0_tor', 'B0_tor_l', 'cnorm_tor', 'r_surface',
          'N_mantle', 'sigma_mantle_cmb', 'c1_mantle', 'kappa_cmb',
          'topo_hrms', 'topo_lambda0', 'topo_H', 'topo_N_cmb', 'topo_omega']  # NEW (2026-11-04)
FLOW = ['KE', 'KP', 'KT', 'Dkin', 'Dint', 'Wlor', 'Wthm', 'Wcmp', 'resid0', 'residu', 'vtorq_real', 'vtorq_imag',
        'vtorq_ic_real', 'vtorq_ic_imag', 'elldom', 'angmom_real', 'angmom_imag', 'residL', 'geo', 'Wbg',
        'Dtopo', 'dsigma_topo']  # NEW (2026-11-04)
MAGNETIC = ['ME', 'Mdfs', 'Indu', 'residb', 'mtorq_real', 'mtorq_imag', 'mtorq_ic_real', 'mtorq_ic_imag', 'brms',
            'eps_twa_cmb', 'eps_twa_icb']  # NEW (2026-11-04)
ROTDYN = ['omg_mantle', 'omg_ic', 'misalignmt', 'gravtorq', 'vtorq', 'mtorq', 'vtorq_ic', 'mtorq_ic']
THERMAL = ['TE', 'Wadv_thm', 'Dthm', 'residt']
COMPOSITIONAL = ['CE', 'Wadv_cmp', 'Dcmp', 'residc']
EIGENVALUES = ['rtau', 'itau']

# the suffix reap.sh gives each table, the file it is collected from, and its column names
TABLES = [('.par', 'params.dat',        PARAMS),
          ('.flo', 'flow.dat',          FLOW),
          ('.mag', 'magnetic.dat',      MAGNETIC),
          ('.rot', 'rotdyn.dat',        ROTDYN),
          ('.tmp', 'thermal.dat',       THERMAL),
          ('.cmp', 'compositional.dat', COMPOSITIONAL),
          ('.eig', 'eigenvalues.dat',   EIGENVALUES)]


def read_table(fname, names):
    '''
    NEW (2026-11-02): reads one table and labels its columns with names, as many as it has. Files of an older
    spin_doctor have fewer columns (new ones were always added at the end); columns beyond the names get col<k>.
    '''
    df = pd.read_csv(fname, sep=r'\s+', header=None)
    n = df.shape[1]
    df.columns = names[:n] + ['col%d' % k for k in range(len(names) + 1, n + 1)]
    return df


def load_results(filename):
    '''
    The tables collected by reap.sh with the prefix filename
    '''
    tables = [ read_table(filename + suffix, names) for suffix, _, names in TABLES if exists(filename + suffix) ]
    return pd.concat(tables, axis=1) if tables else pd.DataFrame()


def load_result():
    '''
    The .dat files of the current folder
    '''
    tables = [ read_table(dat, names) for _, dat, names in TABLES if exists(dat) ]
    return pd.concat(tables, axis=1) if tables else pd.DataFrame()


# --- INITIALIZATION ---
# load results from the solver(s)
if len(sys.argv) == 2:
    filename = sys.argv[1]
    df = load_results(filename)
else:
    df = load_result()

# save results to comma-separated-value file
if len(sys.argv) == 2:
    df.to_csv(filename + '.csv')
else:
    df.to_csv('results.csv')
