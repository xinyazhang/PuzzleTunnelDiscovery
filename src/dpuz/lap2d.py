#!/usr/bin/env python3

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import argparse
from collections import namedtuple
import pycutec2

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # p.add_argument('geo', help='Geometry NPZ file')
    p.add_argument('--bb_min', help='Bounding Box (min)', nargs=2, type=float, default=[-1,-1])
    p.add_argument('--bb_max', help='Bounding Box (max)', nargs=2, type=float, default=[ 1, 1])
    p.add_argument('--res', help='Resolution', nargs=2, default=[1024,1024], type=int)
    p.add_argument('--out', help='Output file for the mesh', required=True)

    args = p.parse_args()
    for attr in ['bb_min', 'bb_max', 'res']:
        setattr(args, attr, np.array(getattr(args, attr)))

    return args

def main():
    args = parse_args()
    V, F = pycutec2.build_mesh_2d(args.bb_min, args.bb_max, args.res)
    if args.out.endswith('.obj'):
        pycutec2.save_obj_1(V, F, args.out)

if __name__ == '__main__':
    main()
