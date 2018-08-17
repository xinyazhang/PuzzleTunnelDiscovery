#!/usr/bin/env python2

'''
ASsembly Visibility Matrix
'''

import os
import sys
sys.path.append(os.getcwd())

import numpy as np

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
    fvm_dir = 'vm-path' # Fragmented Visibility Matrix DIRectory
    block_size = 2048
    out = 'giant-vm-path.npz'
    vmfrags = []
    vmlocators = []
    q0end = 0
    q1end = 0
    for fn in fn_gen(fvm_dir, block_size):
        d = np.load(fn)
        vmfrags.append(d['VMFrag'])
        vmlocators.append(d['Locator'])
        print("Load {}".format(fn))
        q0end = max(q0end, vmlocators[-1][1])
        q1end = max(q1end, vmlocators[-1][3])
    giant = np.full((q0end, q1end), -1, dtype=np.int8)
    for i,(m,loc) in enumerate(zip(vmfrags, vmlocators)):
        [q0start, q0end, q1start, q1end] = loc
        print("Store Block {}".format(i))
        giant[q0start:q0end, q1start:q1end] = m
    np.savez(out, VM=giant)

if __name__ == '__main__':
    main()
