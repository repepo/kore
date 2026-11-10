#!/usr/bin/env python3
'''
Reads the tables that spin_doctor.py writes, one row per solution, into one pandas DataFrame with named columns, and
saves it as a csv file. Use as

    python3 $KORE_DIR/tools/get_data.py [prefix]

prefix : the prefix given to tools/reap.sh, which collected the tables of a series of run folders into prefix.par,
         prefix.flo, ...; writes prefix.csv. Without it, the .dat files of the current folder; writes results.csv.

From python: sys.path.insert(0, KORE_DIR + '/tools'); import get_data; df = get_data.load_results(prefix), or
get_data.load_result() for the current folder.

NEW (2026-11-08, from conductive_IC): the columns are named as spin_doctor.py names them (docs/output.md describes them),
as many as each file has: a file of an older kore has fewer, and columns beyond the names are called col<k> (counted from
1). The text after '#' that ends some rows of params.dat (the expressions of a 'user' background) is left out. The columns
that two files share (resid1 and Wcmp of thermal.dat and composition.dat, the torques of rotdyn.dat) are kept once, from
the first file (flow.dat).
'''
import sys
from os.path import exists, join
import pandas as pd

# the columns of each table, in the order spin_doctor.py writes them (param_table, flow_cols, thermal_cols,
# composition_cols, magnetic_cols, rot_cols); tests/tools/tools_check.py compares them with spin_doctor.py
PARAMS = ['thermal', 'm', 'symm', 'ricb', 'bci', 'bco', 'forcing', 'forcing_frequency', 'forcing_amplitude_cmb',
          'forcing_amplitude_icb', 'Gaspard', 'Beyonce', 'ViscosD', 'ThermaD', 'ncpus', 'N', 'lmax', 'runtime', 'r_out',
          'aux1', 'aux2', 'aux3', 'aux4', 'aux5', 'visc0', 'hvisc', 'rvisc', 'rpower_u', 'rhopower_u', 'rpower_v',
          'rhopower_v', 'rpower_pp', 'rhopower_pp', 'therm0', 'visc_pow', 'therm_pow', 'diff_rot', 'diff_rot_ampl',
          'thermal_wind', 'diff_rot_type', 'dSdtheta'] \
       + ['diff_rot_term_amp_%d' % (2*k) for k in range(9)] \
       + ['thermal_wind_amplitude', 'thermal_wind_radial', 'diff_rot_frame', 'diff_rot_nmax', 'diff_rot_rdeg',
          'diff_rot_generic'] \
       + ['diff_rot_solar_%d' % k for k in range(6)] \
       + ['structure', 'solar_density_deg', 'solar_density_window_lo', 'solar_density_window_hi', 'solar_rz',
          'solar_rz_dSdr', 'solar_rz_x1', 'solar_rz_x2', 'solar_rz_d', 'solar_rz_core', 'solar_rz_xos', 'solar_rz_dos',
          'diff_rot_window', 'diff_rot_hold', 'solar_rz_delta_cz', 'solar_rz_delta_sf', 'drop_tol', 'diff_rot_param',
          'stratification', 'diffusivity', 'magnetic', 'B0', 'B0_l', 'cnorm', 'Hendrik', 'MagnetD', 'innercore', 'N_cic',
          'mu_i2o', 'sigma_i2o', 'B0_tor', 'B0_tor_l', 'cnorm_tor', 'B0_wound', 'B0_wound_eta', 'B0_conf_r1',
          'B0_conf_r2', 'B0_conf_p', 'B0_param', 'B0_tor_param', 'B0_ferraro_nmax', 'mag_couplings', 'tol_B0', 'B0_ic',
          'ferraro_policy', 'centrifugal', 'centrifugal_shape', 'centrifugal_q', 'centrifugal_f', 'centrifugal_param',
          'hydro', 'mantle', 'c_cmb', 'c1_cmb', 'muw_cmb', 'c_icb', 'c1_icb', 'muw_icb', 'mu', 'B0_cmb', 'r_surface',
          'N_mantle', 'sigma_mantle_cmb', 'sigma_mantle_param', 'c1_mantle', 'rotdyn', 'MoIZ_M', 'MoIZ_IC', 'gTorque',
          'OmgtauIC', 'vtrq', 'mtrq', 'kappa_cmb', 'topo_drag', 'topo_hrms', 'topo_lambda0', 'topo_H', 'topo_N_cmb',
          'topo_omega', 'compositional', 'comp_stratification', 'comp_stratification_param', 'BeyonceC', 'CompoD',
          'comp0', 'comp_pow', 'bci_compositional', 'bco_compositional', 'comp_two_zone_rc', 'comp_two_zone_h',
          'two_zone_rc', 'two_zone_h', 'bgflow', 'bgflow_Ro']
FLOW = ['KE', 'KP', 'KT', 'Dkin', 'ldom', 'lwidth', 'lconv', 'Ensvel', 'Enscor', 'Ensvif', 'Ensbuo', 'cuvismax_r',
        'cuvismax_l', 'cuvismax', 'resens', 'resid2', 'resid1', 'Wdr', 'Wdr_im', 'Ensdr', 'Wp_icb', 'Wp_cmb', 'Wp_im',
        'Wv_icb', 'Wv_cmb', 'Dvisc', 'residD', 'irreg0', 'angmom_re', 'angmom_im', 'vtorq_re', 'vtorq_im', 'vtorq_ic_re',
        'vtorq_ic_im', 'mtorq_re', 'mtorq_im', 'mtorq_ic_re', 'mtorq_ic_im', 'residL', 'Dtopo', 'dsig', 'Wf', 'Wf_im',
        'Wcmp', 'Wcmp_im', 'Wbg', 'Wbg_im', 'Ensbg', 'Enslor', 'geo']
THERMAL = ['TE', 'Wadv_thm', 'Dthm', 'resid3', 'Wthm', 'resid1', 'Wdr_thm', 'Wbg_thm']
COMPOSITION = ['CE', 'Wadv_cmp', 'Dcmp', 'resid4', 'Wcmp', 'resid1', 'Wdr_cmp', 'Wbg_cmp']
MAGNETIC = ['ME', 'Mdfs', 'Indu', 'residb', 'Wlor', 'Wlor_im', 'MEic', 'MEext', 'Ohm_f', 'Ohm_ic', 'residE',
            'MEc',                                            # with centrifugal = 1 or a conducting wall
            'E_w', 'Ohm_w', 'eps_twa_o', 'eps_twa_i']         # with a conducting wall (thin, or the resolved mantle)
ROTDYN = [name + part for name in ('omega_M', 'omega_IC', 'phi', 'Kphi', 'vtorq', 'mtorq', 'vtorq_ic', 'mtorq_ic')
          for part in ('_re', '_im')]
EIGENVALUES = ['Re_lambda', 'Im_lambda']

# the suffix reap.sh gives each table, the file it is collected from, and its column names
TABLES = [('.par', 'params.dat',      PARAMS),
          ('.flo', 'flow.dat',        FLOW),
          ('.thm', 'thermal.dat',     THERMAL),
          ('.cmp', 'composition.dat', COMPOSITION),
          ('.mag', 'magnetic.dat',    MAGNETIC),
          ('.rot', 'rotdyn.dat',      ROTDYN),
          ('.eig', 'eigenvalues.dat', EIGENVALUES)]


def read_table(fname, names):
    '''one table, its columns named by names, as many as it has (col<k> beyond them), with the numbers of the file exactly'''
    df = pd.read_csv(fname, sep=r'\s+', header=None, comment='#', float_precision='round_trip')
    n = df.shape[1]
    df.columns = names[:n] + ['col%d' % k for k in range(len(names) + 1, n + 1)]
    return df


def _join(tables):
    '''the tables side by side, each shared column kept once (the first)'''
    if not tables:
        return pd.DataFrame()
    df = pd.concat(tables, axis=1)
    return df.loc[:, ~df.columns.duplicated()]


def load_results(prefix):
    '''the tables that reap.sh collected with the prefix prefix'''
    return _join([read_table(prefix + suffix, names) for suffix, _, names in TABLES if exists(prefix + suffix)])


def load_result(folder='.'):
    '''the .dat files of a run folder'''
    return _join([read_table(join(folder, dat), names) for _, dat, names in TABLES if exists(join(folder, dat))])


if __name__ == '__main__':
    if len(sys.argv) == 2:
        df, out = load_results(sys.argv[1]), sys.argv[1] + '.csv'
    else:
        df, out = load_result(), 'results.csv'
    if df.empty:
        sys.exit('get_data.py: no table found (%s)' % ('%s.par, ...' % sys.argv[1] if len(sys.argv) == 2 else 'params.dat, ...'))
    df.to_csv(out)
    print('%d solutions, %d columns written to %s' % (df.shape[0], df.shape[1], out))
