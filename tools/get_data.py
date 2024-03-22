import pandas as pd
import numpy as np
import numpy.lib.scimath as scimath
import sys
from os.path import exists


'''
Reads results produced by solve.py or collected by reap.sh into a pandas dataframe
Use as:
python3 path/to/get_data.py filename 

filename : prefix used when collecting data via reap.sh (leave empty when reading results directly from solve.py)
'''

def load_results(filename):
    df = pd.DataFrame()
    if exists(filename + '.par'):
        par = pd.read_csv(filename + '.par', sep=' ', names=["hydro", "magnetic", "thermal", "compositional", "Ek", "m",
                                                             "symm", "ricb", "bc_icb", "bc_cmb", "forcing",
                                                             "forcing_freq", "forcing_amp_cmb", "forcing_amp_icb",
                                                             "projection", "B0_type", "FDM_beta", "FDM_l", "mag_bc_icb",
                                                             "c_icb", "c1_icb", "mag_bc_cmb", "c_cmb", "c1_cmb", "mu_cmb",
                                                             "Em", "Le2", "B0_norm", "Et", "T0_type", "thm_BV2",
                                                             "thm_rc", "thm_h", "thm_rsymm", "thm_bc_icb", "thm_bc_cmb",
                                                             "Ec", "C0_type", "cmp_BV2", "cmp_rc", "cmp_h", "cmp_rsymm",
                                                             "cmp_bc_icb", "cmp_bc_cmb", "omgtau", "ncpus", "N", "lmax",
                                                             "time", "mu_icb", "sigma_icb", "aux1", "aux2"])
        df = pd.concat([df, par], axis=1)
    if exists(filename + '.flo'):
        flo = pd.read_csv(filename + '.flo', sep=' ', names=["KE", "KP", "KT", "Dkin", "Dint", "Wlor", "Wthm", "Wcmp",
                                                             "resid0", "resid1", "real_vtorq_cmb", "imag_vtorq_cmb",
                                                             "real_vtorq_icb", "imag_vtorq_icb", "pressure", "aux3"])
        df = df.join(flo)
    if exists(filename + '.mag'):
        mag = pd.read_csv(filename + '.mag', sep=' ', names=["ME", "Dmag", "Wind", "resid2", "real_mtorq_cmb",
                                                             "imag_mtorq_cmb", "real_mtorq_icb", "imag_mtorq_icb"])
        df = df.join(mag)
    if exists(filename + '.tmp'):
        tmp = pd.read_csv(filename + '.tmp', sep=' ', names=["TE", "Wadv_thm", "Dthm", "resid3"])
        df = df.join(tmp)
    if exists(filename + '.cmp'):
        cmp = pd.read_csv(filename + '.cmp', sep=' ', names=["CE", "Wadv_cmp", "Dcmp"])
        df = df.join(cmp)
    if exists(filename + '.eig'):
        eig = pd.read_csv(filename + '.eig', sep=' ', names=["rtau", "itau", "norm"])
        df = df.join(eig)
    return df


def load_result():
    df = pd.DataFrame()
    if exists('params.dat'):
        par = pd.read_csv('params.dat', sep=' ', names=["hydro", "magnetic", "thermal", "compositional",
                                                        "Ek", "m",  "symm", "ricb", "bc_icb", "bc_cmb", "forcing",
                                                        "forcing_freq", "forcing_amp_cmb", "forcing_amp_icb",
                                                        "projection", "B0_type", "FDM_beta", "FDM_l", "mag_bc_icb",
                                                        "c_icb", "c1_icb", "mag_bc_cmb", "c_cmb", "c1_cmb", "mu", "Em",
                                                        "Le2", "B0_norm", "Et", "T0_type", "thm_BV2", "thm_rc", "thm_h",
                                                        "thm_rsymm", "thm_bc_icb", "thm_bc_cmb", "Ec", "C0_type",
                                                        "cmp_BV2", "cmp_rc", "cmp_h", "cmp_rsymm", "cmp_bc_icb",
                                                        "cmp_bc_cmb", "omgtau", "ncpus", "N", "lmax", "time", "mu_icb",
                                                        "sigma_icb", "aux1", "aux2"])
        df = pd.concat([df, par], axis=1)
    if exists('flow.dat'):
        flo = pd.read_csv('flow.dat', sep=' ', names=["KE", "KP", "KT", "Dkin", "Dint", "Wlor", "Wthm",
                                                      "Wcmp", "resid0", "resid1", "real_vtorq_cmb", "imag_vtorq_cmb",
                                                      "real_vtorq_icb", "imag_vtorq_icb", "pressure", "aux3"])
        df = pd.concat([df, flo], axis=1)
    if exists('magnetic.dat'):
        mag = pd.read_csv('magnetic.dat', sep=' ', names=["ME", "Dmag", "Wind", "resid2",
                                                          "real_mtorq_cmb", "imag_mtorq_cmb", "real_mtorq_icb",
                                                          "imag_mtorq_icb"])
        df = pd.concat([df, mag], axis=1)
    if exists('thermal.dat'):
        tmp = pd.read_csv('thermal.dat', sep=' ', names=["TE", "Wadv_thm", "Dthm", "resid3"])
        df = pd.concat([df, tmp], axis=1)
    if exists('compositional.dat'):
        cmp = pd.read_csv('compositional.dat', sep=' ', names=["CE", "Wadv_cmp", "Dcmp"])
        df = pd.concat([df, cmp], axis=1)
    if exists('eigenvalues.dat'):
        eig = pd.read_csv('eigenvalues.dat', sep=' ', names=["rtau", "itau", "norm"])
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
# forcing parameters
if (df["forcing"] == 0).all():
    del df["forcing_amp_icb"]
    del df["forcing_amp_cmb"]
    del df["projection"]
    df["freq"] = df["itau"]
    df["damp"] = df["rtau"]
    del df["rtau"]
    del df["itau"]
elif (df["forcing"] != 0).all():
    df["freq"] = df["forcing_freq"]
    df["damp"] = np.zeros(len(df))

del df["forcing_freq"]

# magnetic parameters
if (df['magnetic'] == 0).all():
    del df["B0_type"]
    del df["FDM_beta"]
    del df["FDM_l"]
    del df["mag_bc_icb"]
    del df["c_icb"]
    del df["c1_icb"]
    del df["mag_bc_cmb"]
    del df["c_cmb"]
    del df["c1_cmb"]
    del df["mu"]
    del df["Em"]
    del df["Le2"]
    del df["B0_norm"]
else:
    if (df["B0_type"] != 5).all():
        del df["FDM_beta"]
        del df["FDM_l"]
    if (df["mag_bc_icb"] == 0).all():
        del df["c_icb"]
        del df["c1_icb"]
    if (df["mag_bc_cmb"] == 0).all():
        del df["c_cmb"]
        del df["c1_cmb"]
        if (df["mag_bc_icb"] == 0).all():
            del df["mu"]
    df["Le"] = np.sqrt(df["Le2"])
    if (df["Em"] != 0).all():
        df["Pm"] = df["Ek"]/df["Em"]
        df["Lam"] = df["Le2"]/df["Em"]

# thermal parameters
if (df["thermal"] == 0).all():
    del df["Et"]
    del df["T0_type"]
    del df["thm_BV2"]
    del df["thm_rc"]
    del df["thm_h"]
    del df["thm_rsymm"]
    del df["thm_bc_icb"]
    del df["thm_bc_cmb"]
else:
    if (df["T0_type"] <= 1).all():
        del df["thm_rc"]
        del df["thm_h"]
        del df["thm_rsymm"]
    if (df["Et"] != 0).all():
        df["Prandtl"] = df["Ek"]/df["Et"]
        df["thm_Ra"] = -df["thm_BV2"] * df["Prandtl"] / (df["Ek"])**2

# compositional parameters
if (df["compositional"] == 0).all():
    del df["Ec"]
    del df["C0_type"]
    del df["cmp_BV2"]
    del df["cmp_rc"]
    del df["cmp_h"]
    del df["cmp_rsymm"]
    del df["cmp_bc_icb"]
    del df["cmp_bc_cmb"]
else:
    if (df["C0_type"] <= 1).all():
        del df["cmp_rc"]
        del df["cmp_h"]
        del df["cmp_rsymm"]
    if (df["Ec"] != 0).all():
        df["Schmidt"] = df["Ek"]/df["Ec"]
        df["cmp_Ra"] = -df["cmp_BV2"] * df["Schmidt"] / (df["Ek"])**2

# save results to comma-separated-value file
if len(sys.argv) == 2:
    df.to_csv(filename + '.csv')
else:
    df.to_csv('results.csv')