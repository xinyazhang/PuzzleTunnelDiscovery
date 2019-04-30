import numpy as np
import h5py
from scipy.io import loadmat,savemat
import pathlib

def _load_csv(fn):
    return np.loadtxt(fn, delimiter=',')

def _load_hdf5(fn):
    return h5py.File(fn, 'r')

_SUFFIX_TO_LOADER = {
        '.npz': np.load,
        '.txt': np.loadtxt,
        '.csv': _load_csv,
        '.mat': loadmat,
        '.hdf5': _load_hdf5
        }

def load(fn):
    p = pathlib.PosixPath(fn)
    return _SUFFIX_TO_LOADER[p.suffix](fn)

def hdf5_overwrite(f, path, ds):
    if path in f:
        del f[path]
    if np.isscalar(ds) or np.ndim(ds) == 0:
        # Scalar datasets don't support chunk/filter options
        f.create_dataset(path, data=ds)
    else:
        f.create_dataset(path, data=ds, compression='lzf')

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
