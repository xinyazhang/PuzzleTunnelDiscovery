import numpy as np
from scipy.io import loadmat,savemat
import pathlib
import lzma
import io

def _load_csv(fn):
    return np.loadtxt(fn, delimiter=',')

def _load_hdf5(fn):
    import h5py
    return h5py.File(fn, 'r')

def _load_xz(fn):
    p = pathlib.PosixPath(fn)
    memfile = io.BytesIO(lzma.open(str(fn), 'r').read())
    nest_suffix = p.with_suffix('').suffix
    if nest_suffix not in _SUFFIX_TO_LOADER:
        raise NotImplementedError("Parser for {} file is not implemented".format(p.suffix))
    return _SUFFIX_TO_LOADER[nest_suffix](memfile)

def _loadmat(fn):
    return loadmat(fn, verify_compressed_data_integrity=False)

_SUFFIX_TO_LOADER = {
        '.npz': np.load,
        '.txt': np.loadtxt,
        '.csv': _load_csv,
        '.mat': _loadmat,
        '.hdf5': _load_hdf5,
        '.xz': _load_xz
        }

def load(fn, key=None):
    p = pathlib.PosixPath(fn)
    if p.suffix not in _SUFFIX_TO_LOADER:
        raise NotImplementedError("Parser for {} file is not implemented".format(p.suffix))
    try:
        d = _SUFFIX_TO_LOADER[p.suffix](fn)
    except Exception as e:
        print("error in loading {}".format(fn))
        print(e)
        exit()
    if p.suffix == '.txt':
        return d
    if key is not None:
        return d[key]
    return d

"""
load_safeshape:
    fn: file name
    key: key in the filename

Return
    the shape of the np.ndarray stored in the file[key].
    [None] if the file does not exist or the key is not in the file.
"""
def load_safeshape(fn, key):
    p = pathlib.PosixPath(fn)
    if not p.is_file():
        return [None]
    # print(f'loadding {fn}')
    d = load(fn)
    if key not in d:
        return [None]
    return d[key].shape

'''
hdf5_safefile:
    The only way to ensure its safety is to overwite.
'''
def hdf5_safefile(fn):
    import h5py
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
