'''
NEW (2026-11-08, from conductive_IC): the solution files of kore, one solution per column, in one of two formats
(par.field_format): 'npy', one binary file per field, name.npy (complex; the default), or 'text', the real and imaginary
parts in real_name.field and imag_name.field (19 significant digits, so that both formats hold the same numbers to the
last bit). The fields are 'flow', 'thermal', 'magnetic', 'magnetic_ic', 'magnetic_mantle', 'composition' and 'rotdyn'
(solve.py, solve_ind.py). This module reads no parameters, so that the tests and the tools can read the files of any run
folder; kore's scripts use it through utils.save_field, utils.load_field and utils.field_exists.
'''
import os
import numpy as np

FORMATS = ('npy', 'text')


def files(name, fmt, folder='.'):
    '''the solution files of field name in the format fmt ('npy' or 'text'), in folder'''
    names = [name + '.npy'] if fmt == 'npy' else ['real_' + name + '.field', 'imag_' + name + '.field']
    return [os.path.join(folder, f) for f in names]


def exists(name, folder='.'):
    '''whether the solution files of field name are in folder, in either format'''
    return os.path.isfile(files(name, 'npy', folder)[0]) or all(os.path.isfile(f) for f in files(name, 'text', folder))


def save(name, blk, fmt, folder='.'):
    '''
    Writes the solutions blk (one per column) of field name in the format fmt, and deletes the files of the other format,
    so that load never reads a stale one.
    '''
    if fmt not in FORMATS:
        raise ValueError('field_format %r: %s' % (fmt, ' or '.join(repr(f) for f in FORMATS)))
    if fmt == 'npy':
        np.save(files(name, 'npy', folder)[0], np.asarray(blk))
    else:
        re_, im_ = files(name, 'text', folder)
        np.savetxt(re_, np.real(blk))
        np.savetxt(im_, np.imag(blk))
    for f in files(name, 'text' if fmt == 'npy' else 'npy', folder):
        if os.path.isfile(f):
            os.remove(f)


def load(name, cols=None, folder='.'):
    '''
    The solutions of field name in folder as a complex array (rows, solutions), from name.npy or else from the text files;
    only the solution cols if it is an int (a 1-d array), or the solutions cols if it is a list. None if neither format's
    files are there.
    '''
    npy, (re_, im_) = files(name, 'npy', folder)[0], files(name, 'text', folder)
    if os.path.isfile(npy):
        x = np.load(npy, mmap_mode='r')
        x = x.reshape((x.shape[0], -1)) if x.ndim < 2 else x
        return np.array(x if cols is None else x[:, cols], dtype=complex)
    if not (os.path.isfile(re_) and os.path.isfile(im_)):
        return None
    x = np.loadtxt(re_, usecols=cols, ndmin=2) + 1j*np.loadtxt(im_, usecols=cols, ndmin=2)
    return x.ravel() if cols is not None and np.ndim(cols) == 0 else x
