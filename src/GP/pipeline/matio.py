import numpy as np
import h5py
from scipy.io import loadmat,savemat

def load(fn):
    if fn.endswith('.npz'):
        return np.load(fn)
    elif fn.endswith('.txt'):
        return np.loadtxt(fn)
    elif fn.endswith('.csv'):
        return np.loadtxt(fn, delimiter=',')
    elif fn.endswith('.mat'):
        return loadmat(fn)
    elif fn.endswith('.hdf5'):
        return h5py.File(fn, 'r')

def hdf5_overwrite(f, path, ds):
    if path in f:
        del f[path]
    f.create_dataset(path, data=ds, compression='lzf')
