#!/usr/bin/env python2

import os
import sys
sys.path.append(os.getcwd())

import pyosr
import aniconf12 as aniconf
import numpy as np

def index_to_ranges(V0, V1, block_size, index):
    block_per_row, remainder = divmod(V1.shape[0], block_size)
    block_per_row += 1 if remainder != 0 else 0
    q0start = index / block_per_row
    q0end = q0start + 1
    q1start = (index % block_per_row) * block_size
    q1end = min(V1.shape[0], q1start + block_size)
    return q0start, q0end, q1start, q1end, block_per_row * V0.shape[0]

def visibilty_matrix_calculator(V0, V1, q0start, q0end, q1start, q1end, out_dir, index=None, block_size=None):
    r = pyosr.UnitWorld() # pyosr.Renderer is not avaliable in HTCondor
    r.loadModelFromFile(aniconf.env_fn)
    r.loadRobotFromFile(aniconf.rob_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)

    VM = r.calculate_visibility_matrix2(V0[q0start:q0end], False,
                                        V1[q1start:q1end], False,
                                        0.0125 * 4 / 8)
    if out_dir == '-':
        print(VM)
    else:
        if index is None or block_size is None:
            fn = '{}/q0-{}T{}-q1-{}T{}.npz'.format(out_dir, q0start, q0end, q1start, q1end)
        else:
            fn = '{}/index-{}-under-bs-{}.npz'.format(out_dir, index, block_size) # Second naming scheme
        np.savez(fn, VMFrag=VM, Locator=[q0start, q0end, q1start, q1end])

def usage():
    print('''
Find least visible samples
    ''')
    print('''
Usage:
condor-visibility-matrix.py <q0 start> <q0 end> <q1 start> <q1 end> <output directory>
or
condor-visibility-matrix.py <block size> <index>
'''
    )

def main():
    gtfn = 'blend-low.gt.npz'
    pathfn = '../res/alpha/alpha-1.2.org.w-first.npz'
    gtdic = np.load(gtfn)
    pathdic = np.load(pathfn)
    V0 = pathdic['VS']
    V1 = gtdic['V']
    # print('len {}'.format(len(sys.argv)))

    if len(sys.argv) == 3:
        print(index_to_ranges(V0, V1, int(sys.argv[1]), int(sys.argv[2])))
    elif len(sys.argv) == 4:
        block_size = int(sys.argv[1])
        index = int(sys.argv[2])
        q0start, q0end, q1start, q1end, index_max = index_to_ranges(V0, V1, block_size, index)
        visibilty_matrix_calculator(V0,
                                    V1,
                                    q0start, q0end, q1start, q1end,
                                    sys.argv[3],
                                    index=index,
                                    block_size=block_size)
    elif len(sys.argv) == 6:
        visibilty_matrix_calculator(V0,
                                    V1,
                                    int(sys.argv[1]), int(sys.argv[2]),
                                    int(sys.argv[3]), int(sys.argv[4]),
                                    sys.argv[5])
    else:
        usage()
        exit()

if __name__ == '__main__':
    main()
