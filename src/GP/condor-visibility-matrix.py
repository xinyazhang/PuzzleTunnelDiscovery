#!/usr/bin/env python2
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import os
import sys
sys.path.append(os.getcwd())

import pyosr
import aniconf12 as aniconf
import numpy as np

from condor_vm import *

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
        visibilty_matrix_calculator(aniconf,
                                    V0, V1,
                                    q0start, q0end, q1start, q1end,
                                    sys.argv[3],
                                    index=index,
                                    block_size=block_size)
    elif len(sys.argv) == 6:
        visibilty_matrix_calculator(aniconf,
                                    V0, V1,
                                    int(sys.argv[1]), int(sys.argv[2]),
                                    int(sys.argv[3]), int(sys.argv[4]),
                                    sys.argv[5])
    else:
        usage()
        exit()

if __name__ == '__main__':
    main()
