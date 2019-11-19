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
from progressbar import progressbar

from . import util
from . import matio
sys.path.insert(0, os.getcwd())
try:
    import pyse3ompl as plan
    PLANNER_PRM = plan.PLANNER_PRM
    PLANNER_RDT = plan.PLANNER_RDT
except ImportError as e:
    util.warn(str(e))
    util.warn(str(sys.version))
    util.warn(str(sys.version_info))
    util.warn("[WARNING] CANNOT IMPORT pyse3ompl. This node is incapable of RDT forest planning")
from . import parse_ompl
PDS_FLAG_TERMINATE = 1

_lsv = util.lsv

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
        return None, 1, 0, None
    is_fn = dic['file']
    is_key = dic['key'] if 'key' in dic else 'KEYQ_OMPL'
    is_offset = int(dic['offset'])
    is_size = int(dic['size']) if 'size' in dic else 1
    is_out = dic['out']
    d = np.load(is_fn)
    '''
    if is_key is None:
        is_key = list(d.keys())[0] # <- DON'T IT'S RISKY
    '''
    A = d[is_key]
    total_size = A.shape[0]
    if is_size == -1:
        is_size = total_size
    return A, is_size, is_offset, is_out

def _pair_generator(istate_dic, gstate_dic):
    A, is_size, is_offset, is_out = _load_states_from_dic(istate_dic)
    B, gs_size, gs_offset, gs_out = _load_states_from_dic(gstate_dic)
    out_path = is_out if is_out else gs_out
    out_dir = out_path if isdir(out_path) else dirname(out_path)
    for is_index in range(is_offset, is_offset+is_size):
        istate = np.zeros(shape=(0)) if A is None else A[is_index]
        pds_tree_index = is_index
        for gs_index in range(gs_offset, gs_offset+gs_size):
            gstate = np.zeros(shape=(0)) if B is None else B[gs_index]
            yield istate, gstate, is_index, gs_index, pds_tree_index, out_dir

def add_performance_numbers_to_dic(driver, dic):
    pn = driver.latest_performance_numbers
    dic['PF_LOG_PLAN_T'] = pn.planning_time
    dic['PF_LOG_MCHECK_N'] = pn.motion_check
    dic['PF_LOG_MCHECK_T'] = pn.motion_check_time
    dic['PF_LOG_DCHECK_N'] = pn.motion_discrete_state_check
    dic['PF_LOG_KNN_QUERY_T'] = pn.knn_query_time
    dic['PF_LOG_KNN_DELETE_T'] = pn.knn_delete_time

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
        if args.use_blooming_tree:
            assert 'QB' in d
            assert 'QE' in d
            assert 'QEB' in d
            driver.set_sample_set_edges(QB=d['QB'], QE=d['QE'], QEB=d['QEB'])
    else:
        record_compact_tree = False
    print("solve args {}".format(args))

    if args.replace_istate is None and args.replace_gstate is None:
        return_ve = args.bloom_out is not None
        V, _ = driver.solve(args.days, args.out, sbudget=args.sbudget, return_ve=return_ve)
        if args.trajectory_out:
            is_complete = (driver.latest_solution_status == plan.EXACT_SOLUTION)
            np.savez(args.trajectory_out, OMPL_TRAJECTORY=driver.latest_solution, FLAG_IS_COMPLETE=is_complete)
        return

    h5traj = None
    complete_tuple = []
    complete_list = []
    for istate, gstate, is_index, gs_index, pds_tree_index, pds_out_dir in _pair_generator(args.istate_dic, args.gstate_dic):
        driver.substitute_state(plan.INIT_STATE, istate)
        driver.substitute_state(plan.GOAL_STATE, gstate)
        index = pds_tree_index
        if args.samset2:
            index = current
        ssc_fn = '{}/ssc-{}.mat'.format(pds_out_dir, index)
        tree_fn = '{}/compact_tree-{}.mat'.format(pds_out_dir, index)
        if args.samset and args.skip_existing:
            if os.path.exists(ssc_fn) and os.path.exists(tree_fn):
                print("skipping exising file {} and {}".format(ssc_fn, tree_fn))
                continue
        return_ve = bool(args.bloom_out is not None or args.trajectory_out)
        V,E = driver.solve(args.days, return_ve=return_ve, sbudget=args.sbudget, record_compact_tree=record_compact_tree, continuous_motion_validator=ccd)
        '''
        if isdir(out_path):
            savemat('{}/tree-{}.mat'.format(out_path, index), dict(V=V, E=E), do_compression=True)
        else:
            savemat(out_path, dict(V=V, E=E), do_compression=True)
        '''
        if args.trajectory_out:
            if h5traj is None:
                h5traj = matio.hdf5_safefile(args.trajectory_out)
            is_complete = (driver.latest_solution_status == plan.EXACT_SOLUTION)
            matio.hdf5_overwrite(h5traj, f'{gs_index}/OMPL_TRAJECTORY', driver.latest_solution)
            matio.hdf5_overwrite(h5traj, f'{gs_index}/FLAG_IS_COMPLETE', is_complete)
            complete_tuple.append([is_index, gs_index, int(is_complete)])
            if is_complete:
                complete_list.append(gs_index)
        if args.samset:
            ssc_data = driver.get_sample_set_connectivity()
            savemat(ssc_fn, dict(C=ssc_data), do_compression=True)
            # np.savez_compressed(ssc_fn+'.npz', C=ssc_data)
            util.log("saving ssc matrix {}, shape {}".format(ssc_fn, ssc_data.shape))
        if record_compact_tree:
            CNVI, CNV, CE = driver.get_compact_graph()
            savemat(tree_fn, dict(CNVI=CNVI, CNV=CNV, CE=CE), do_compression=True)
            util.log("saving compact tree {}".format(tree_fn))
        if args.bloom_out:
            '''
            _, _, CE = driver.get_compact_graph()
            np.savez(args.bloom_out, BLOOM=V, BLOOM_EDGE=CE)
            '''
            dic = { 'BLOOM': V,
                    'BLOOM_EDGE': np.array(E.nonzero()),
                    'IS_INDICES': driver.get_graph_istate_indices(),
                    'GS_INDICES': driver.get_graph_gstate_indices()
                  }
            add_performance_numbers_to_dic(driver, dic)
            np.savez(args.bloom_out, **dic)
            util.log("saving bloom results to {}".format(args.bloom_out))
    if h5traj is not None:
        matio.hdf5_overwrite(h5traj, 'COMPLETE_TUPLE', complete_tuple)
        matio.hdf5_overwrite(h5traj, 'COMPLETE_LIST', complete_list)
        h5traj.close()

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

'''
merge_blooming_forest:
    Merge the forest from blooming algorithm
'''
def merge_blooming_forest(args):
    print('running merge_blooming_forest with {}'.format(vars(args)))
    puzzle = args.puzzle
    args.planner_id = plan.PLANNER_RDT
    args.sampler_id = 0 # Uniform sampler
    args.saminj = ''
    args.rdt_k = 0
    driver = create_driver(args)
    bloom_files = _lsv(indir=args.bloom_dir, prefix='bloom-from_', suffix='.npz')
    for bf in progressbar(bloom_files):
        d = matio.load(bf)
        V = d['BLOOM']
        nv = V.shape[0]
        E = sparse.csr_matrix((nv, nv), dtype=np.uint8)
        driver.add_existing_graph(V, E)
    inter_tree_edges = driver.merge_existing_graph(args.knn,
                                                   verbose=True,
                                                   version=args.algo_version,
                                                   subset=args.subset)
    if args.out is not None:
        dic = { 'INTER_BLOOMING_TREE_EDGES': inter_tree_edges }
        add_performance_numbers_to_dic(driver, dic)
        np.savez(args.out, **dic)
    if args.algo_version >= 2:
        return
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(bloom_files))])
    G.add_edges_from(inter_tree_edges[:,[0,2]])
    try:
        ids = nx.shortest_path(G, 0, 1)
        print('Solved with KNN, tree level path: {}'.format(ids))
    except nx.exception.NetworkXNoPath:
        print('Failed to solve with KNN')

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
