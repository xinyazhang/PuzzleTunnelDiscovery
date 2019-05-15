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

from . import util
sys.path.insert(0, os.getcwd())
try:
    import pyse3ompl as plan
    PLANNER_PRM = plan.PLANNER_PRM
    PLANNER_RDT = plan.PLANNER_RDT
except ImportError as e:
    util.warn(str(e))
    util.warn("[WARNING] CANNOT IMPORT pyse3ompl. This node is incapable of RDT forest planning")
from . import parse_ompl
PDS_FLAG_TERMINATE = 1

def create_driver(args):
    puzzle = args.puzzle
    planner_id = args.planner_id
    sampler_id = args.sampler_id

    cdres = args.cdres if hasattr(args, 'cdres') else None
    saminj = args.saminj if hasattr(args, 'saminj') else ''
    rdt_k = args.rdt_k if hasattr(args, 'rdt_k') else 1
    cfg, config = parse_ompl.parse_simple(puzzle)

    driver = plan.OmplDriver()
    driver.set_planner(planner_id, sampler_id, saminj, rdt_k)
    driver.set_model_file(plan.MODEL_PART_ENV, cfg.env_fn)
    driver.set_model_file(plan.MODEL_PART_ROB, cfg.rob_fn)
    for i,q_tup in zip([plan.INIT_STATE, plan.GOAL_STATE], [cfg.iq_tup, cfg.gq_tup]):
        driver.set_state(i, q_tup.tr, q_tup.rot_axis, q_tup.rot_angle)
    lo = parse_ompl.read_xyz(config, 'problem', 'volume.min')
    hi = parse_ompl.read_xyz(config, 'problem', 'volume.max')
    if hasattr(args, 'bvresize'):
        lo -= args.bvresize
        hi += args.bvresize
    driver.set_bb(lo, hi)
    if cdres is None:
        cdres = config.getfloat('problem', 'collision_resolution')
    driver.set_cdres(cdres)
    if hasattr(args, 'solver_option_vector') and args.solver_option_vector:
        driver.set_option_vector(args.solver_option_vector)

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
            return_ve = args.bloom_out is not None
            V,_ = driver.solve(args.days, return_ve=return_ve, sbudget=args.sbudget, record_compact_tree=record_compact_tree, continuous_motion_validator=ccd)
            '''
            if isdir(out_path):
                savemat('{}/tree-{}.mat'.format(out_path, index), dict(V=V, E=E), do_compression=True)
            else:
                savemat(out_path, dict(V=V, E=E), do_compression=True)
            '''
            if args.samset:
                savemat(ssc_fn, dict(C=driver.get_sample_set_connectivity()), do_compression=True)
                util.log("saving ssc matrix {}".format(ssc_fn))
            if record_compact_tree:
                CNVI, CNV, CE = driver.get_compact_graph()
                savemat(tree_fn, dict(CNVI=CNVI, CNV=CNV, CE=CE), do_compression=True)
                util.log("saving compact tree {}".format(tree_fn))
            if args.bloom_out:
                np.savez(args.bloom_out, BLOOM=V)
                util.log("saving bloom results to {}".format(args.bloom_out))
    else:
        return_ve = args.bloom_out is not None
        V, _ = driver.solve(args.days, args.out, sbudget=args.sbudget, return_ve=return_ve)
        if args.trajectory_out:
            is_complete = (driver.latest_solution_status == plan.EXACT_SOLUTION)
            np.savez(args.trajectory_out, OMPL_TRAJECTORY=driver.latest_solution, FLAG_IS_COMPLETE=is_complete)


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
