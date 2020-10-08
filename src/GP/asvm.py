#!/usr/bin/env python2
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

'''
ASsembly Visibility Matrix
'''

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import argparse
from scipy.io import savemat
from scipy import sparse
from progressbar import progressbar

def fn_gen(fvm_dir, block_size):
    index = 0
    while True:
        fn = '{}/index-{}-under-bs-{}.npz'.format(fvm_dir, index, block_size)
        index += 1
        if os.path.exists(fn):
            yield fn
        else:
            return

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', help='Directory of segmented visibility matrix', nargs=None, type=str)
    parser.add_argument('block_size', help='Block Size', nargs=None, type=int)
    parser.add_argument('output', help='File output of assembled matrix', nargs=None, type=str)
    args = parser.parse_args()
    fvm_dir = args.dir
    block_size = args.block_size
    out = args.output
    vmfrags = []
    vmlocators = []
    q0end = 0
    q1end = 0
    for fn in progressbar(fn_gen(fvm_dir, block_size)):
        d = np.load(fn)
        vmfrags.append(d['VMFrag'])
        vmlocators.append(d['Locator'])
        # print("Load {}".format(fn))
        q0end = max(q0end, vmlocators[-1][1])
        q1end = max(q1end, vmlocators[-1][3])
    giant = np.full((q0end, q1end), -1, dtype=np.int8)
    for i,(m,loc) in enumerate(zip(vmfrags, vmlocators)):
        [q0start, q0end, q1start, q1end] = loc
        # print("Store Block {}".format(i))
        giant[q0start:q0end, q1start:q1end] = m
    if np.min(giant) < 0:
        print("Caveat: the assembled visibility matrix has undefined coefficients")
    if out.endswith('.mat'):
        savemat(out, dict(VM=sparse.csr_matrix(giant)), do_compression=True)
    else:
        np.savez(out, VM=giant)

if __name__ == '__main__':
    main()
