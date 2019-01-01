#!/usr/bin/env python2

import os
import sys
sys.path.append(os.getcwd())

import pyosr
import aniconf12
import numpy as np
import dualconf_tiny

from condor_vm import *
import argparse

def usage():
    print('''
Computer visibility matrix in segments
    ''')
    print('''
Usage:
condor-visibility-matrix2.py <block size> <index>
'''
    )

_name2module = { 'aniconf12' : aniconf12,
                 'dual_tiny' : dualconf_tiny }

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    info_parser = subparsers.add_parser("info")
    calc_parser = subparsers.add_parser("calc", help='Calculate the visibility matrix between path (axis 0) and prm (axis 1)')
    info_parser.add_argument('prm', help='Sample from PRM', nargs=None, type=str)
    info_parser.add_argument('path', help='Sample from PRM', nargs=None, type=str)
    info_parser.add_argument('block_size', help='Block Size', nargs=None, type=int)
    info_parser.add_argument('task_id', help='Task Index', nargs='?', default=-1, type=int)
    calc_parser.add_argument('prm', help='Sample from PRM', nargs=None)
    calc_parser.add_argument('path', help='Sample from PRM', nargs=None)
    calc_parser.add_argument('puzzlename', help='Sample from PRM', choices=['aniconf12', 'dual_tiny'])
    calc_parser.add_argument('block_size', help='Block Size', nargs=None, type=int)
    calc_parser.add_argument('task_id', help='Task Index', nargs=None, type=int)
    calc_parser.add_argument('out', help='Output Directory', nargs=None, type=str)
    return parser

def parse():
    parser = get_parser()
    return parser.parse_args()

def _get_V0V1(args):
    gtdic = np.load(args.prm)
    pathdic = np.load(args.path)
    V0 = pathdic['VS']
    V1 = gtdic['V']
    return V0, V1

def info(args):
    V0, V1 = _get_V0V1(args)
    print("Path vertices {}".format(len(V0)))
    print("PRM vertices {}".format(len(V1)))
    if args.task_id < 0:
        print("Total number of tasks: {}".format(index_to_ranges(V0, V1, args.block_size, 0)[-1]))
    else:
        tup = index_to_ranges(V0, V1, args.block_size, args.task_id)
        print("Calculate path[{}:{}] vs prm[{}:{}]. Total number of tasks: {}".format(tup[0], tup[1], tup[2], tup[3], tup[4]))

def calc(args):
    args.model = _name2module[args.puzzlename]
    print(args.model.env_fn)
    V0, V1 = _get_V0V1(args)
    q0start, q0end, q1start, q1end, index_max = index_to_ranges(V0, V1, args.block_size, args.task_id)
    visibilty_matrix_calculator(args.model,
                                V0, V1,
                                q0start, q0end, q1start, q1end,
                                args.out,
                                index=args.task_id,
                                block_size=args.block_size)

def main():
    args = parse()
    globals()[args.command](args)

if __name__ == '__main__':
    main()
