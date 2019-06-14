import numpy as np
import h5py
from scipy.io import loadmat,savemat
import pathlib
import lzma
import io

def _load_csv(fn):
    return np.loadtxt(fn, delimiter=',')

def _load_hdf5(fn):
    return h5py.File(fn, 'r')

def _load_hdf5_xz(fn):
    input_file = lzma.open(fn, 'r')
    return h5py.File(input_file, 'r')

def _load_xz(fn):
    p = pathlib.PosixPath(fn)
    memfile = io.BytesIO(lzma.open(fn, 'r').read())
    nest_suffix = p.with_suffix('').suffix
    if nest_suffix not in _SUFFIX_TO_LOADER:
        raise NotImplementedError("Parser for {} file is not implemented".format(p.suffix))
    return _SUFFIX_TO_LOADER[nest_suffix](memfile)

_SUFFIX_TO_LOADER = {
        '.npz': np.load,
        '.txt': np.loadtxt,
        '.csv': _load_csv,
        '.mat': loadmat,
        '.hdf5': _load_hdf5,
        '.xz': _load_xz
        }

def load(fn, key=None):
    p = pathlib.PosixPath(fn)
    if p.suffix not in _SUFFIX_TO_LOADER:
        raise NotImplementedError("Parser for {} file is not implemented".format(p.suffix))
    d = _SUFFIX_TO_LOADER[p.suffix](fn)
    if p.suffix == '.txt':
        return d
    if key is not None:
        return d[key]
    return d

'''
hdf5_safefile:
    The only way to ensure its safety is to overwite.
'''
def hdf5_safefile(fn):
    return h5py.File(fn, 'w')

def hdf5_overwrite(f, path, ds):
    if path in f:
        del f[path]
    if np.isscalar(ds) or np.ndim(ds) == 0:
        # Scalar datasets don't support chunk/filter options
        f.create_dataset(path, data=ds)
    else:
        f.create_dataset(path, data=ds, compression='lzf')

def hdf5_open(f, path, shape, dtype, **kwds):
    if path in f:
        del f[path]
    return f.create_dataset(path, shape=shape, dtype=dtype, **kwds)

def savetxt(fn, a):
    np.savetxt(fn, a, fmt='%.17g')

def npz_cat(fns):
    dlist = {}
    for fn in fns:
        d = np.load(fn)
        for k,v in d.items(): # Python 3 syntax
            if k not in dlist:
                dlist[k] = [v]
            else:
                dlist[k].append(v)
    dcat = {}
    for k,v in dlist.items():
        dcat[k] = np.concatenate(dlist[k])
    return dcat
