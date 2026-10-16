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
        par = pd.read_csv(filename + '.par', sep=' ', names=['hydro',
                                                             'magnetic',
                                                             'thermal',
                                                             'compositional',
                                                             'Ek',
                                                             'm',
                                                             'symm',
                                                             'ricb',
                                                             'bci',
                                                             'bco',
                                                             'forcing',
                                                             'forcing_frequency',
                                                             'forcing_amplitude_cmb',
                                                             'forcing_amplitude_icb',
                                                             'projection',
                                                             'B0type',
                                                             'beta_actual',
                                                             'B0_l',
                                                             'innercore_mag_bc',
                                                             'c_icb',
                                                             'c1_icb',
                                                             'mantle_mag_bc',
                                                             'c_cmb',
                                                             'c1_cmb',
                                                             'mu',
                                                             'Em',
                                                             'Le2',
                                                             'B0_norm',
                                                             'Etherm',
                                                             'heating',
                                                             'BV2',
                                                             'rc',
                                                             'h',
                                                             'rsy',
                                                             'bci_thermal',
                                                             'bco_thermal',
                                                             'Ecomp',
                                                             'compositional_background',
                                                             'BV2_comp',
                                                             'rcc',
                                                             'hc',
                                                             'rsyc',
                                                             'bci_compositional',
                                                             'bco_compositional',
                                                             'OmgTau',
                                                             'ncpus',
                                                             'N',
                                                             'lmax',
                                                             'runtime',
                                                             'mu_i2o',
                                                             'sigma_i2o',
                                                             'aux1',
                                                             'aux2',
                                                             'rotdyn',
                                                             'MoIZ_M',
                                                             'MoIZ_IC',
                                                             'OmgtauIC',
                                                             'gTorque'])
        df = pd.concat([df, par], axis=1)
    if exists(filename + '.flo'):
        flo = pd.read_csv(filename + '.flo', sep=' ', names=["KE",
                                                             "KP",
                                                             "KT",
                                                             "Dkin",
                                                             "Dint",
                                                             "Wlor",
                                                             "Wthm",
                                                             "Wcmp",
                                                             "resid0",
                                                             "residu",
                                                             "vtorq_real",
                                                             "vtorq_imag",
                                                             "vtorq_ic_real",
                                                             "vtorq_ic_imag",
                                                             "elldom",
                                                             "angmom_real",
                                                             "angmom_imag",
                                                             "residL",
                                                             "geo"])
        df = pd.concat([df, flo], axis=1)
    if exists(filename + '.mag'):
        mag = pd.read_csv(filename + '.mag', sep=' ', names=["ME",
                                                             "Mdfs",
                                                             "Indu",
                                                             "residb",
                                                             "mtorq_real",
                                                             "mtorq_imag",
                                                             "mtorq_ic_real",
                                                             "mtorq_ic_imag",
                                                             "brms"])
        df = pd.concat([df, mag], axis=1)
    if exists(filename + '.rot'):
        rot = pd.read_csv(filename + '.rot', sep=' ', names=["omg_mantle",
                                                             "omg_ic",
                                                             "misalignmt",
                                                             "gravtorq",
                                                             "vtorq",
                                                             "mtorq",
                                                             "vtorq_ic",
                                                             "mtorq_ic"])
        df = pd.concat([df, rot], axis=1)
    if exists(filename + '.tmp'):
        tmp = pd.read_csv(filename + '.tmp', sep=' ', names=["TE",
                                                             "Wadv_thm",
                                                             "Dthm",
                                                             "residt"])
        df = pd.concat([df, tmp], axis=1)
    if exists(filename + '.cmp'):
        cmp = pd.read_csv(filename + '.cmp', sep=' ', names=["CE",
                                                             "Wadv_cmp",
                                                             "Dcmp"])
        df = pd.concat([df, cmp], axis=1)
    if exists(filename + '.eig'):
        eig = pd.read_csv(filename + '.eig', sep=' ', names=["rtau",
                                                             "itau"])
        df = pd.concat([df, eig], axis=1)
    return df


def load_result():
    df = pd.DataFrame()
    if exists('params.dat'):
        par = pd.read_csv('params.dat', sep=' ', names=['hydro',
                                                             'magnetic',
                                                             'thermal',
                                                             'compositional',
                                                             'Ek',
                                                             'm',
                                                             'symm',
                                                             'ricb',
                                                             'bci',
                                                             'bco',
                                                             'forcing',
                                                             'forcing_frequency',
                                                             'forcing_amplitude_cmb',
                                                             'forcing_amplitude_icb',
                                                             'projection',
                                                             'B0type',
                                                             'beta_actual',
                                                             'B0_l',
                                                             'innercore_mag_bc',
                                                             'c_icb',
                                                             'c1_icb',
                                                             'mantle_mag_bc',
                                                             'c_cmb',
                                                             'c1_cmb',
                                                             'mu',
                                                             'Em',
                                                             'Le2',
                                                             'B0_norm',
                                                             'Etherm',
                                                             'heating',
                                                             'BV2',
                                                             'rc',
                                                             'h',
                                                             'rsy',
                                                             'bci_thermal',
                                                             'bco_thermal',
                                                             'Ecomp',
                                                             'compositional_background',
                                                             'BV2_comp',
                                                             'rcc',
                                                             'hc',
                                                             'rsyc',
                                                             'bci_compositional',
                                                             'bco_compositional',
                                                             'OmgTau',
                                                             'ncpus',
                                                             'N',
                                                             'lmax',
                                                             'runtime',
                                                             'mu_i2o',
                                                             'sigma_i2o',
                                                             'aux1',
                                                             'aux2',
                                                             'rotdyn',
                                                             'MoIZ_M',
                                                             'MoIZ_IC',
                                                             'OmgtauIC',
                                                             'gTorque'])
        df = pd.concat([df, par], axis=1)
    if exists('flow.dat'):
        flo = pd.read_csv('flow.dat', sep=' ', names=["KE",
                                                      "KP",
                                                      "KT",
                                                      "Dkin",
                                                      "Dint",
                                                      "Wlor",
                                                      "Wthm",
                                                      "Wcmp",
                                                      "resid0",
                                                      "residu",
                                                      "vtorq_real",
                                                      "vtorq_imag",
                                                      "vtorq_ic_real",
                                                      "vtorq_ic_imag",
                                                      "elldom",
                                                      "angmom_real",
                                                      "angmom_imag",
                                                      "residL",
                                                      "geo"])
        df = pd.concat([df, flo], axis=1)
    if exists('magnetic.dat'):
        mag = pd.read_csv('magnetic.dat', sep=' ', names=["ME",
                                                             "Mdfs",
                                                             "Indu",
                                                             "residb",
                                                             "mtorq_real",
                                                             "mtorq_imag",
                                                             "mtorq_ic_real",
                                                             "mtorq_ic_imag",
                                                             "brms"])
        df = pd.concat([df, mag], axis=1)
    if exists('rotdyn.dat'):
        rot = pd.read_csv('rotdyn.dat', sep=' ', names=["omg_mantle",
                                                             "omg_ic",
                                                             "misalignmt",
                                                             "gravtorq",
                                                             "vtorq",
                                                             "mtorq",
                                                             "vtorq_ic",
                                                             "mtorq_ic"])
        df = pd.concat([df, rot], axis=1)
    if exists('thermal.dat'):
        tmp = pd.read_csv('thermal.dat', sep=' ', names=["TE",
                                                         "Wadv_thm",
                                                         "Dthm",
                                                         "residt"])
        df = pd.concat([df, tmp], axis=1)
    if exists('compositional.dat'):
        cmp = pd.read_csv('compositional.dat', sep=' ', names=["CE",
                                                             "Wadv_cmp",
                                                             "Dcmp"])
        df = pd.concat([df, cmp], axis=1)
    if exists('eigenvalues.dat'):
        eig = pd.read_csv('eigenvalues.dat', sep=' ', names=["rtau",
                                                             "itau"])
        df = pd.concat([df, eig], axis=1)
    return df


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

