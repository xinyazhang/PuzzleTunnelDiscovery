#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import subprocess
import pathlib
import numpy as np
import copy
import multiprocessing
from imageio import imwrite as imsave
from imageio import imread
from progressbar import progressbar, ProgressBar
import shutil

from . import util
from . import disjoint_set
from . import choice_formatter
try:
    from . import se3solver
except ImportError as e:
    util.warn(str(e))
    util.warn("[WARNING] CANNOT IMPORT se3solver. Some function will be disabled and the pipeline is broken")
from . import partt
from . import condor
from . import matio
from . import atlas
from . import texture_format
from . import parse_ompl
from .solve import (remote_assemble_pds,
        setup_parser as original_setup_parser
)

ALGORITHM_VERSION_PHASE2_WITH_BLOOMING_TREE = 5

class TmpDriverArgs(object):
    pass

def _rel_bloom_scratch(ws, puzzle_name, trial):
    return join(util.SOLVER_SCRATCH, puzzle_name, util.PDS_SUBDIR, f'bloom-{ws.current_trial}')

def _puzzle_pds(ws, puzzle_name, trial):
    pds_dir = ws.local_ws(util.SOLVER_SCRATCH,
                          puzzle_name,
                          util.PDS_SUBDIR)
    os.makedirs(pds_dir, exist_ok=True)
    fn ='{}.npz'.format(_trial_id(ws, trial))
    return join(pds_dir, fn)

def _trial_str(current_trial):
    return f'pairwise_knn-{current_trial}'
'''
knn_forest:
    Connect forest with prm like algorithm
'''
def pairwise_knn(args, ws):
    ALGO_VERSION = 2
    trial_str = _trial_str(args.current_trial)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        _, config = parse_ompl.parse_simple(puzzle_fn)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        if args.only_wait:
            condor.local_wait(scratch_dir)
            continue
        key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
        if args.task_id is None:
            keys = matio.load(key_fn)
            condor_job_args = ['./facade.py',
                    'solve1',
                    '--stage', 'pairwise_knn',
                    '--current_trial', str(args.current_trial),
                    '--puzzle_name', puzzle_name,
                    '--task_id', '$(Process)',
                    ws.local_ws()]
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=rel_scratch_dir,
                                arguments=condor_job_args,
                                instances=keys['KEYQ_OMPL'].shape[0],
                                wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
        else:
            solver_args = TmpDriverArgs()
            solver_args.puzzle = puzzle_fn
            rel_bloom = _rel_bloom_scratch(ws, puzzle_name, ws.current_trial)
            solver_args.bloom_dir = ws.local_ws(rel_bloom)
            solver_args.out = join(scratch_dir, 'pairwise_knn_edges-{}.npz'.format(args.task_id))
            solver_args.knn = 8 # default
            solver_args.algo_version = ALGO_VERSION # algorithm version
            solver_args.subset = np.array([args.task_id], dtype=np.int)
            se3solver.merge_blooming_forest(solver_args)
    if args.task_id is not None:
        return
    if args.no_wait:
        return
    if not args.only_wait:
        only_wait_args = copy.deepcopy(args)
        only_wait_args.only_wait = True
        pairwise_knn(only_wait_args, ws)

def _ibte_fn(ws, puzzle_name, trial):
    return ws.local_ws(util.SOLVER_SCRATCH, puzzle_name, _trial_str(ws.current_trial),
                       'inter_blooming_tree_edges.npz')

def assemble_knn(args, ws):
    trial_str = _trial_str(args.current_trial)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
        keys = matio.load(key_fn)
        NTree = keys['KEYQ_OMPL'].shape[0]
        ITE_array = []
        for i in range(NTree):
            fn = join(scratch_dir, f'pairwise_knn_edges-{i}.npz')
            ITE_array.append(matio.load(fn)['INTER_BLOOMING_TREE_EDGES'])
        ITE = util.safe_concatente(ITE_array, axis=0)
        np.savez_compressed(ws.local_ws(rel_scratch_dir, 'inter_blooming_tree_edges.npz'),
                            INTER_BLOOMING_TREE_EDGES=ITE)

VIRTUAL_OPEN_SPACE_NODE = 1j
OPENSPACE_FLAG = 1

def _extract_bound(B, total, i):
    f = B[i]
    t = total if f == B.shape[0] else B[i+1]
    return f, t

def tree_level_path(Q, QB, QE, QEB, QF, from_fi, from_vi, to_fi, to_vi):
    import networkx as nx
    assert from_fi == to_fi, f'from_fi {from_fi} does not match to_fi {to_fi}'
    q_from, q_to = _extract_bound(QB, Q.shape[0], from_fi)
    from_gvi = q_from + from_vi
    to_gvi = q_from + to_vi
    qe_from, qe_to = _extract_bound(QEB, QE.shape[0], from_fi)
    G = nx.Graph()
    G.add_nodes_from([i for i in range(q_from, q_to)] + [VIRTUAL_OPEN_SPACE_NODE])
    G.add_edges_from(QE[qe_from:qe_to])
    if to_vi == VIRTUAL_OPEN_SPACE_NODE:
        to_gvi = VIRTUAL_OPEN_SPACE_NODE
        for root, flag in enumerate(QF):
            if flag & OPENSPACE_FLAG:
                virtual_edges.append((root + q_from, VIRTUAL_OPEN_SPACE_NODE))
        G.add_edges_from(virtual_edges)
    ids = nx.shortest_path(G, from_gvi, to_gvi)
    if to_vi == VIRTUAL_OPEN_SPACE_NODE:
        ids = ids[:-1]
    return Q[ids, :]

def connect_knn(args, ws):
    trial_str = _trial_str(ws.current_trial)
    algoprefix = 'pairwise_knn-'
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        ITE = matio.load(_ibte_fn(ws, puzzle_name))['INTER_BLOOMING_TREE_EDGES']
        pds_fn = _puzzle_pds(ws, puzzle_name, ws.current_trial)
        d = matio.load(pds_fn)
        QF = d['QF']

        import networkx as nx
        # Forest level path
        G = nx.Graph()
        G.add_nodes_from([i for i in range(QF.shape[0])] + [VIRTUAL_OPEN_SPACE_NODE])
        G.add_edges_from(inter_tree_edges[:,[0,2]])
        virtual_edges = []
        for root, flag in enumerate(QF):
            if flag & OPENSPACE_FLAG:
                virtual_edges.append((root, VIRTUAL_OPEN_SPACE_NODE))
        G.add_edges_from(virtual_edges)
        try:
            ids = nx.shortest_path(G, 0, VIRTUAL_OPEN_SPACE_NODE)
            util.log('Forest-level shortest path {}'.format(ids))
        except nx.exception.NetworkXNoPath:
            util.warn(f'[connect_knn] Cannot find path for puzzle {puzzle_name}')
            continue
        ITE_meta = {}
        for index, ite in progressbar(enumerate(ITE)):
            from_fi, from_vi, to_fi, to_vi = ite
            if from_fi not in ITE_meta:
                ITE_meta[from_fi] = {}
            if to_fi not in ITE_meta[from_fi]:
                ITE_meta[from_fi][to_fi] = [(from_vi, to_vi)]
            else:
                ITE_meta[from_fi][to_fi].append((from_vi, to_vi))
        Q = d['Q']
        QB = d['QB']
        QE = d['QE']
        QEB = d['QEB']
        ompl_q = []
        bloom0_fn = ws.local_ws(_rel_bloom_scratch(ws, puzzle_name, ws.current_trial), f'bloom-from_0.npz')
        prev_fi = 0
        prev_vi = matio.load(bloom0_fn)['IS_INDICES'][0]
        try:
            for from_fi, to_fi in zip(ids, ids[1:]):
                from_vi, to_vi = ITE_meta[from_fi][to_fi][0]
                ompl_q += tree_level_path(Q, QB, QE, QEB, QF, prev_fi, prev_vi, from_fi, from_vi)
                gvi = QB[to_fi + to_vi]
                ompl_q.append(Q[gvi,:])
                prev_fi = to_fi
                prev_vi = to_vi
            ompl_q += tree_level_path(Q, QB, QE, QEB, QF, prev_fi, prev_vi, prev_fi, VIRTUAL_OPEN_SPACE_NODE)
        except nx.exception.NetworkXNoPath:
            assert False, "Should not happen. Found forest-level path but no tree-level path."
            continue
        path_out = ws.local_ws(rel_scratch_dir, algoprefix + 'path.txt')
        matio.savetxt(path_out, ompl_q)

        uw = util.create_unit_world(puzzle_fn)
        unit_q = uw.translate_ompl_to_unit(ompl_q)
        sol_out = ws.solution_file(puzzle_name, type_name=algoprefix+'unit')
        matio.savetxt(sol_out, unit_q)

function_dict = {
        'pairwise_knn': pairwise_knn,
        'assemble_knn': assemble_knn,
        'connect_knn': connect_knn,
}

def setup_parser(subparsers, module_name='solve'):
    original_setup_parser(subparsers, module_name='solve1', function_dict=function_dict)

def run(args):
    if args.stage in function_dict:
        ws = util.Workspace(args.dir)
        for current_trial in util.rangestring_to_list(args.current_trial):
            ws.current_trial = current_trial
            function_dict[args.stage](args, ws)
    else:
        print("Unknown solve pipeline stage {}".format(args.stage))

#
# Automatic functions start here
#
def _remote_command(ws, cmd, auto_retry=True, alter_host='', extra_args=''):
    if not alter_host:
        alter_host = ws.condor_host
    ws.remote_command(alter_host,
                      ws.condor_exec(),
                      ws.condor_ws(),
                      'solve1', cmd, auto_retry=auto_retry,
                      with_trial=True,
                      extra_args=extra_args)

def _remote_command_distributed(ws, cmd, extra_args=''):
    for host,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        _remote_command(ws, cmd,
                        alter_host=host,
                        extra_args=extra_args+' --puzzle_name {} --no_wait'.format(puzzle_name))
    for host,_,puzzle_name in ws.condor_host_vs_test_puzzle_generator():
        _remote_command(ws, cmd,
                        alter_host=host,
                        extra_args=extra_args+' --puzzle_name {} --only_wait'.format(puzzle_name))

def _remote_command_auto(ws, cmd, extra_args=''):
    if ws.condor_extra_hosts:
        _remote_command_distributed(ws, cmd, extra_args=extra_args)
    else:
        _remote_command(ws, cmd, extra_args=extra_args)

def remote_pairwise_knn(ws):
    _remote_command_auto(ws, 'pairwise_knn')

def remote_assemble_knn(ws):
    _remote_command_auto(ws, 'assemble_knn')

def remote_connect_knn(ws):
    _remote_command_auto(ws, 'connect_knn')

def collect_stages(variant=0):
    if variant in [5.1]:
        ret = [
                ('assemble_pds', remote_assemble_pds),
                ('pairwise_knn', remote_pairwise_knn),
                ('assemble_knn', remote_assemble_knn),
                ('connect_knn', remote_connect_knn),
              ]
    else:
        assert False, f'Solve1 Pipeline Variant {variant} has not been implemented'
    return ret

def autorun(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)

