#!/usr/bin/env python3

'''
Note:
    This application does NOT (and ARE NOT ABLE TO) check the same PDS is used.
    User is responsible to guarantee this.
'''

import sys, os
sys.path.append(os.getcwd())

import argparse
import numpy as np
from scipy.io import loadmat
import h5py
import scipy.sparse as sparse
from progressbar import progressbar, ProgressBar
from pipeline import matio
from psutil import virtual_memory

_OPENSPACE_FLAG = 1

def _total_memory():
    return virtual_memory().total

def collect_ITE(files, buf, ITE, low, high, pbar=None, QF=None, roots_to_open=None):
    batch = high - low
    '''
    Load data into buffer
    '''
    N = len(files)
    for i,fn in enumerate(files):
        row_data = matio.load(fn)['C'].todense()[:, low:high]
        buf[i, 0:batch] = row_data
        if pbar is not None:
            pbar += batch
    col_sum = np.sum(buf, axis=0)
    for i in range(batch):
        local_col = batch - 1 - i
        global_col = local_col + low
        col_data = buf[:, local_col]
        if col_sum[local_col] >= 2:
            star_from_pds = col_data.nonzero()[0]
            edge_from = star_from_pds[:-1]
            edge_to = star_from_pds[1:]
            ITE[edge_from, edge_to] = global_col + 1
            pbar += N
        if QF is not None and (QF[global_col] & _OPENSPACE_FLAG != 0):
            rows = col_data.nonzero()[0].tolist()
            roots_to_open.update(rows)

def print_edge(args):
    if args.pdsflags is not None:
        QF = np.load(args.pdsflags)['QF']
    else:
        QF = None
    f = h5py.File(args.out, mode='a')
    N = len(args.files)                     # N: number of roots
    K = matio.load(args.files[0])['C'].shape[1]  # K: PDS size

    #inter_tree_dtype = np.uint32 if K < np.iinfo(np.uint32).max else np.uint64
    inter_tree_dtype = np.int64
    inter_tree_max = np.iinfo(inter_tree_dtype).max
    memory_limit = _total_memory() * 0.4
    print("N {}".format(N))
    print("memory_limit {}".format(memory_limit))
    pds_limit_per_run = min(K, int(memory_limit / N))
    print("pds_limit_per_run {}".format(pds_limit_per_run))
    ITE = inter_tree_edge = np.zeros((N, N), dtype=inter_tree_dtype)
    per_run_buffer = np.zeros((N, pds_limit_per_run), dtype=np.int8)
    sep = list(range(K, 0, -pds_limit_per_run))
    sep.append(0)
    pbar = ProgressBar(max_value=K*N*2)
    roots_to_open = set()
    print("Chunk the forest from {} to {}".format((N,K), per_run_buffer.shape))
    '''
    iterate through batches of PDS columns
    '''
    for high,low in zip(sep[:-1], sep[1:]):
        if high == low:
            continue
        collect_ITE(args.files, per_run_buffer, ITE, low, high, pbar,
                    QF=QF, roots_to_open=roots_to_open)
    del per_run_buffer

    edge_from, edge_to = ITE.nonzero()
    edge = np.transpose(np.array([edge_from, edge_to, ITE[edge_from, edge_to]], dtype=ITE.dtype))
    matio.hdf5_overwrite(f, 'E', edge)
    if QF is not None:
        roots_to_open_list = np.array(list(roots_to_open), dtype=ITE.dtype)
        matio.hdf5_overwrite(f, 'OpenTree', roots_to_open_list)

    f.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', help='ssc-*.mat file', nargs='+')
    parser.add_argument('--out', help='output edge file in .hdf5', required=True)
    parser.add_argument('--pdsflags', help='File that stores PDS Flags, usually in the same npz file that also stores PDS', default=None)
    args = parser.parse_args()
    if not args.out.endswith('.hdf5'):
        print("--out requires hdf5 extension")
        return
    print_edge(args)

if __name__ == '__main__':
    main()
