#!/usr/bin/env python2

import os
import sys
sys.path.append(os.getcwd())

import pyosr
import aniconf12 as aniconf
import numpy as np

def visibilty_matrix_calculator(gtfn, pathfn, q0start, q0end, q1start, q1end, out_dir):
    r = pyosr.UnitWorld() # pyosr.Renderer is not avaliable in HTCondor
    r.loadModelFromFile(aniconf.env_fn)
    r.loadRobotFromFile(aniconf.rob_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)

    gtdic = np.load(gtfn)
    pathdic = np.load(pathfn)

    V0 = pathdic['VS']
    V1 = gtdic['V']
    VM = r.calculate_visibility_matrix2(V0[q0start:q0end], False,
                                        V1[q1start:q1end], False,
                                        0.0125 * 4 / 8)
    if out_dir == '-':
        print(VM)
    else:
        fn = '{}/q0-{}-q1-{}.npz'.format(out_dir, q0start, q1start)
        np.savez(fn, VMFrag=VM)

def usage():
    print('''
Find least visible samples
    ''')
    print('''
Usage: condor-tunnel-segment-finder.py <q0 start> <q0 end> <q1 start> <q1 end> <output directory>
'''
    )

if __name__ == '__main__':
    if len(sys.argv) < 6:
        usage()
        exit()
    #tunnel_finder(sys.argv[1], sys.argv[2], sys.argv[3])
    visibilty_matrix_calculator('blend-low.gt.npz',
                                '../res/alpha/alpha-1.2.org.w-first.npz',
                                int(sys.argv[1]), int(sys.argv[2]),
                                int(sys.argv[3]), int(sys.argv[4]),
                                sys.argv[5])
