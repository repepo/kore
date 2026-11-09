'''
NEW (2026-11-03): the file of the radial operators, submatrices.npz, which submatrices.py writes and operators.py reads.

WHY. submatrices.py used to write one Matrix Market text file per operator (<label>.mtx), and operators.py loaded every
*.mtx in the current directory. Text is about 2.4 times larger than binary and slower to read (0.5 s against 0.04 s for
the 268 operators of an N = 192 magnetic run), and a .mtx left over from an earlier run could stand in, unnoticed, for an
operator this run did not build. Now one binary file holds every operator of a run together with the parameters they were
built from, and operators.py loads that file only and stops when the run's parameters differ.

HOW. For each label, the CSR arrays of its matrix ('<label>.data', '.indices', '.indptr', '.shape'); 'params' holds, as
JSON text, the repr of every parameter except those in OPERATOR_INDEPENDENT (which change no operator, so a run may change
them between submatrices.py and assemble.py). A parameter added to params_default.py is compared unless it is added to
that list. The file is written under a temporary name and renamed, so a submatrices.py that stops leaves no partial file.
Only numpy and scipy are imported (no parameters.py), so outside tools can read the file with read().
'''
import os
import sys
import json
import numpy as np
import scipy.sparse as ss

FILE = 'submatrices.npz'

# parameters that change no radial operator: the solver and its target, the parallel layout, the forcing (assemble.py,
# reynolds_stress.py; also every rsf_* name), the boundary conditions (rows of assemble.py), post-processing, the inputs of
# set_scales (their results Gaspard, ViscosD, ThermaD, Beyonce, Hendrik, MagnetD are compared), the Ferraro policy (it only
# stops a run), the .mtx export and hydro (NEW 2026-11-06: submatrices.py builds the flow's operators either way)
OPERATOR_INDEPENDENT = ('ncpus', 'petsc_opts', 'prescale', 'which_eigenpairs', 'nev', 'tol', 'maxit', 'rtau', 'itau',
                        'forcing', 'forcing_frequency', 'forcing_amplitude_cmb', 'forcing_amplitude_icb',
                        'bci', 'bco', 'bci_thermal', 'bco_thermal', 'mu_i2o',
                        'rpower_pp', 'rhopower_pp', 'hvisc', 'rvisc',
                        'timescale', 'Ek', 'Em', 'Etherm', 'Le', 'g',
                        'ferraro_policy', 'write_mtx', 'hydro',
                        'c1_cmb', 'c_icb', 'c1_icb', 'muw_icb',   # NEW (2026-11-06): thin walls, rows of assemble.py only
                        'kappa_cmb', 'topo_drag', 'topo_hrms', 'topo_lambda0', 'topo_H', 'topo_N', 'topo_omega', 'topo_K',
                        # NEW (2026-11-06): the drag at the outer wall (topo.py) and the rotating solids (rotdyn.py): rows of
                        # assemble.py only
                        'rotdyn', 'MoIZ_M', 'MoIZ_IC', 'gTorque', 'OmgtauIC', 'vtrq', 'mtrq',
                        'wobble', 'MoIX_M', 'dynflat_M',   # NEW (2026-11-09): the wobble of the mantle (wobble.py)
                        'forced_walls',   # NEW (2026-11-06): the solids under a radial boundary flow, rows of assemble.py
                        'Ecomp', 'bci_compositional', 'bco_compositional',   # NEW (2026-11-08): the composition's (CompoD is
                        # compared), its walls
                        'track_target', 'field_format')   # NEW (2026-11-08): the solver's target, the solution files


def params_of(par):
    '''{name: repr(value)} of the parameters that can change an operator (all but OPERATOR_INDEPENDENT and rsf_*); NEW
    (2026-11-08) with a background flow from a file (bgflow = 1, bgflow_profiles empty), the sha256 of that file as well, under
    the name 'bgflow_file sha256' (its content changes the operators, not only its name)'''
    out = {}
    for name, value in sorted(vars(par).items()):
        if name in OPERATOR_INDEPENDENT or name.startswith('rsf_'):
            continue
        out[name] = repr(value.tolist() if isinstance(value, np.ndarray) else value)
    if getattr(par, 'bgflow', 0) and not getattr(par, 'bgflow_profiles', None):
        out['bgflow_file sha256'] = _sha256(par.bgflow_file)
    return out


def _sha256(fname):
    '''the sha256 of a file as hex text, '(absent)' if there is none'''
    import hashlib
    if not os.path.isfile(fname):
        return '(absent)'
    with open(fname, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def write(mats, par, fname=FILE):
    '''all the operators {label: sparse matrix} and the parameters of par in one .npz (written whole or not at all)'''
    arrs = {'params': np.array(json.dumps(params_of(par)))}
    for label, m in mats.items():
        m = ss.csr_matrix(m)
        arrs[label + '.data'], arrs[label + '.indices'], arrs[label + '.indptr'] = m.data, m.indices, m.indptr
        arrs[label + '.shape'] = np.array(m.shape)
    tmp = fname[:-4] + '_partial.npz'
    np.savez(tmp, **arrs)
    os.replace(tmp, fname)


def read(fname=FILE):
    '''({label: CSR matrix}, {name: repr(value)}) from a file written by write()'''
    with np.load(fname) as z:
        params = json.loads(str(z['params']))
        labels = sorted({k.rsplit('.', 1)[0] for k in z.files if k != 'params'})
        mats = {lab: ss.csr_matrix((z[lab + '.data'], z[lab + '.indices'], z[lab + '.indptr']), shape=tuple(z[lab + '.shape']))
                for lab in labels}
    return mats, params


def check(built, par, fname=FILE):
    '''stop if the parameters the operators were built from (built, from read()) differ from those of par'''
    now = params_of(par)
    # NEW (2026-11-06): a name listed in OPERATOR_INDEPENDENT after the file was written is not compared either
    diff = [n for n in sorted(set(built) | set(now)) if built.get(n) != now.get(n)
            and n not in OPERATOR_INDEPENDENT and not n.startswith('rsf_')]
    if diff:
        lines = ['  %s: %s when built, %s now' % (n, built.get(n, '(absent)'), now.get(n, '(absent)')) for n in diff[:10]]
        more = ['  ... and %d more' % (len(diff) - 10)] if len(diff) > 10 else []
        sys.exit('\n'.join(['%s was built with other parameters; run submatrices.py again:' % fname] + lines + more))
