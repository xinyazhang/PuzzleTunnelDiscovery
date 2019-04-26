#!/usr/bin/env python3

import argparse
from six.moves import configparser
import numpy as np
from scipy.io import savemat,loadmat
import scipy.sparse as sparse
import sys
import os
from os.path import abspath, dirname, join, isdir
import itertools

sys.path.insert(0, os.getcwd())
import pyse3ompl as plan
import parse_ompl
from parse_ompl import read_xyz

PLANNER_PRM = plan.PLANNER_PRM
PLANNER_RDT = plan.PLANNER_RDT

def create_driver(puzzle, planner_id, sampler_id, cdres=None, saminj='', rdt_k=1):
    cfg, config = parse_ompl.parse_simple(puzzle)
    driver = plan.OmplDriver()
    driver.set_planner(planner_id, sampler_id, aminj, rdt_k)
    driver.set_model_file(plan.MODEL_PART_ENV, cfg.env_fn)
    driver.set_model_file(plan.MODEL_PART_ROB, cfg.rob_fn)
    for i,prefix in zip([plan.INIT_STATE, plan.GOAL_STATE], ['start', 'goal']):
        tr = read_xyz(config, 'problem', prefix)
        rot_axis = read_xyz(config, 'problem', prefix + '.axis')
        rot_angle  = config.getfloat('problem', prefix + '.theta')
        driver.set_state(i, tr, rot_axis, rot_angle)
    lo = read_xyz(config, 'problem', 'volume.min')
    hi = read_xyz(config, 'problem', 'volume.max')
    driver.set_bb(lo, hi)
    if cdres is not None:
        cdres = config.getfloat('problem', 'collision_resolution')
    driver.set_cdres(cdres)

    return driver

def _load_states_from_dic(dic):
    if 'file' not in dic:
        return None, 0, 0, None
    is_fn = dic['file']
    is_key = dic['key'] if 'key' in dic else None
    is_offset = int(dic['offset'])
    is_size = int(dic['size']) if 'size' in dic else 1
    is_out = dic['out']
    d = np.load(is_fn)
    if is_key is None:
        is_key = list(d.keys())[0]
    A = d[is_key]
    total_size = A.shape[0]
    if is_size == -1:
        is_size = total_size
    return A, is_size, is_offset, is_out

def solve(args):
    driver = create_driver(args)
    ccd = (args.cdres <= 0.0)
    if args.samset2:
        current = int(args.samset2[0])
        total = int(args.samset2[1])
        prefix = args.samset2[2]
        args.samset = '{}{}.npz'.format(prefix, current % total)
    if args.samset:
        d = np.load(args.samset)
        Q = d['Q']
        driver.set_sample_set(Q)
        if 'QF' in d:
            QF = d['QF']
            driver.set_sample_set_flags(QF)
        record_compact_tree = True
    else:
        record_compact_tree = False
    print("solve args {}".format(args))
    if args.replace_istate is not None or args.replace_gstate is not None:
        A, is_size, is_offset, is_out = _load_states_from_dic(args.istate_dic)
        B, gs_size, gs_offset, gs_out = _load_states_from_dic(args.gstate_dic)
        # FIXME: better error handling for input
        assert is_size == gs_size or is_size == 0 or gs_size == 0
        if is_size == 0:
            out_path = gs_out
        else:
            out_path = is_out
        if isdir(out_path):
            out_dir = out_path
        else:
            out_dir = dirname(out_path)
        print("is_size, gs_size {} {}".format(is_size,gs_size))
        print("A, B {} {}".format(A.shape if A is not None else None, B.shape if B is not None else None))
        for i in range(max(is_size,gs_size)):
            index = is_offset + i
            gindex = gs_offset + i
            if is_size > 0:
                if index > A.shape[0]:
                    print("istate offset {} out of range {}".format(index, A.shape[0]))
                    break
                driver.substitute_state(plan.INIT_STATE, A[index])
            if gs_size > 0:
                if gindex > B.shape[0]:
                    print("gstate offset {} out of range {}".format(index, B.shape[0]))
                    break
                driver.substitute_state(plan.GOAL_STATE, B[gindex])
            if args.samset2:
                index = current
            ssc_fn = '{}/ssc-{}.mat'.format(out_dir, index)
            tree_fn = '{}/compact_tree-{}.mat'.format(out_path, index)
            if args.samset and args.skip_existing:
                if os.path.exists(ssc_fn) and os.path.exists(tree_fn):
                    print("skipping exising file {} and {}".format(ssc_fn, tree_fn))
                    continue
            _,_ = driver.solve(args.days, return_ve = False, sbudget=args.sbudget, record_compact_tree=record_compact_tree, continuous_motion_validator=ccd)
            '''
            if isdir(out_path):
                savemat('{}/tree-{}.mat'.format(out_path, index), dict(V=V, E=E), do_compression=True)
            else:
                savemat(out_path, dict(V=V, E=E), do_compression=True)
            '''
            if args.samset:
                savemat(ssc_fn, dict(C=driver.get_sample_set_connectivity()), do_compression=True)
            if record_compact_tree:
                CNVI, CNV, CE = driver.get_compact_graph()
                savemat(tree_fn, dict(CNVI=CNVI, CNV=CNV, CE=CE), do_compression=True)
    else:
        driver.solve(args.days, args.out, sbudget=args.sbudget)


def merge_forest(args):
    # Create PRM planner
    args.planner = plan.PLANNER_PRM
    args.sampler = 0 # Uniform sampler
    args.saminj = ''
    args.rdt_k = 0
    driver = create_driver(args)
    def matgen(paths):
        for p in paths:
            if os.path.isfile(p):
                yield p
            else:
                for fn in os.listdir(p):
                    if fn.endswith('.mat'):
                        yield join(p, fn)
    for fn in matgen(args.trees):
        d = loadmat(fn)
        driver.add_existing_graph(d['V'], d['E'])
    V,E = driver.solve(args.days, return_ve=True, sbudget=args.sbudget)
    if args.out is not None:
        savemat(args.out, dict(V=V, E=E), do_compression=True)

def presample(args):
    args.planner = plan.PLANNER_PRM
    args.saminj = ''
    args.rdt_k = 0
    driver = create_driver(args)
    Q = driver.presample(args.nsamples)
    np.savez(args.out, Q=Q)

def merge_pdsc(args):
    fnl = []
    for i in itertools.count(start=0, step=1):
        fn = '{}/ssc-{}.mat'.format(args.dir, i)
        if not os.path.isfile(fn):
            break
        fnl.append(fn)
    cl = []
    for fn in fnl:
        cl.append(loadmat(fn)['C'])
    # merged connectivity matrix
    mc = sparse.vstack(cl)
    savemat(args.out, dict(MC=mc), do_compression=True)

def main():
    main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = main_parser.add_subparsers(dest='command')
    # Subcommand 'solve'
    parser = subparsers.add_parser("solve", help='Solve a puzzle with the specified planner.')
    parser.add_argument('puzzle', help='Configure file generated by OMPL GUI')
    parser.add_argument('planner', help='Choose a planner', choices=range(18), type=int)
    parser.add_argument('days', help='Time limit in day(s)', type=float)
    parser.add_argument('--sbudget', help='Number of samples limit, in additional to time limit', type=int ,default=-1)
    parser.add_argument('--out', help='Output complete planning data', default='')
    parser.add_argument('--sampler', help='Valid state sampler', type=int, default=0)
    parser.add_argument('--saminj', help='Sample injection file', type=str, default='')
    parser.add_argument('--samset', help='Predefined sample set', type=str, default='')
    parser.add_argument('--samset2', help='Predefined sample set, in current total prefix', type=str, nargs=3, default=[])
    parser.add_argument('--skip_existing', help='Quit if the output file already exists', action='store_true')
    parser.add_argument('--rdt_k', help='K Nearest in RDT algorithm', type=int, default=1)
    parser.add_argument('--cdres', help='Collision detection resolution, set zero/negative to enable continuouse collision detection', type=float, default=0.005)
    parser.add_argument('--replace_istate', help='''
Replace Initial State. Syntax: file=<path to npz>,key=<npz key>,offset=<number>,size=<number>,out=<dir>,
in which key=, size= are optional.
key is default to the first array in NPZ,
and size is default to 1''',
            type=str, default=None)
    parser.add_argument('--replace_gstate', help='''Same syntax with replace_istate, but replaces goal state''', type=str, default=None)
    # Subcommand 'merge_forest'
    parser = subparsers.add_parser("merge_forest", help='Merge the planning data from multiple planners')
    parser.add_argument('puzzle', help='Configure file generated by OMPL GUI')
    parser.add_argument('days', help='Time limit in day(s)', type=float)
    parser.add_argument('trees', help='''Tree/Graph files/directories created by individual planner.
If a directory is provided, all .mat files in this directory will be loaded''', nargs='+')
    parser.add_argument('--out', help='Output file of the merged graph, in .npz format', default=None)
    parser.add_argument('--cdres', help='Collision detection resolution', type=float, default=0.005)
    # Subcommand 'presample'
    parser = subparsers.add_parser("presample", help='Presample a set of samples.')
    parser.add_argument('puzzle', help='Configure file generated by OMPL GUI')
    parser.add_argument('nsamples', help='Total Number of samples', type=int)
    parser.add_argument('out', help='Output file for samples')
    parser.add_argument('--sampler', help='Valid state sampler', type=int, default=0)
    parser.add_argument('--cdres', help='Collision detection resolution', type=float, default=0.005)
    # Subcommand 'merge_pdsc'
    parser = subparsers.add_parser("merge_pdsc", help='Merge connectivity matrix created from PreDefined set of samples.')
    parser.add_argument('pdsf', help='Pre-Defined Sample set (PDS) File')
    parser.add_argument('dir', help='''Directory that stores the connectivity sparse matrices''')
    parser.add_argument('out', help='''output file''')
    args = main_parser.parse_args()
    # Empty dic 
    args.istate_dic = {}
    args.gstate_dic = {}
    if hasattr(args, 'replace_istate') and args.replace_istate is not None:
        dic = dict(item.split("=") for item in args.replace_istate.split(","))
        print(dic)
        assert ('file' in dic) and ('offset' in dic) and ('out' in dic), '--replace_istate need file, offset and out'
        if 'size' in dic and int(dic['size']) > 1:
            assert 'out' in dic, 'size= requires out= for storage'
        args.istate_dic = dic
    if hasattr(args, 'replace_gstate') and args.replace_gstate is not None:
        dic = dict(item.split("=") for item in args.replace_gstate.split(","))
        print(dic)
        assert ('file' in dic) and ('offset' in dic) and ('out' in dic), '--replace_gstate need file, offset and out'
        if 'size' in dic and int(dic['size']) > 1:
            assert 'out' in dic, 'size= requires out= for storage'
        args.gstate_dic = dic

    globals()[args.command](args)

if __name__ == '__main__':
    main()
