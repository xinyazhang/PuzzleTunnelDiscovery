#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import copy
import argparse
import numpy as np
import networkx as nx
from progressbar import progressbar

from . import util
from . import condor
from . import parse_ompl
from . import matio

def get_ec_limit(args, ws, puzzle_name):
    if args.ec_limit is not None:
        return args.ec_limit
    return ws.config.getint('Solver', 'PDSBloom') * 2

def query(args, ws, reference_trial, current_trial):
    trial_str = 'ref-{}_trial-{}'.format(reference_trial, current_trial)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_scratch_dir = join(util.BASELINE_SCRATCH,
                               puzzle_name,
                               'pwrdtc',
                               trial_str)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
        keys = matio.load(key_fn, key='KEYQ_OMPL')
        ninstances = keys.shape[0]
        G = nx.Graph()
        G.add_nodes_from([i for i in range(ninstances)]) # Root nodes
        CM = np.zeros((ninstances, ninstances), dtype=np.int8)
        for i in progressbar(range(ninstances)):
            d = matio.load(f'{scratch_dir}/traj_{i}.hdf5')
            if 'COMPLETE_TUPLE' in d:
                tup = d['COMPLETE_TUPLE'][...]
                CM[tup[:,0], tup[:,1]] = tup[:,2]
            else:
                for j in range(ninstances):
                    complete = bool(d[f'{j}/FLAG_IS_COMPLETE'][...])
                    if complete:
                        CM[i,j] = 1
        edges = np.transpose(np.nonzero(CM))
        G.add_edges_from(edges)
        np.savez_compressed(join(scratch_dir, 'cache_conn_matrix.npz'), CM=CM)
        out_fn = join(scratch_dir, 'forest_path.npz')
        try:
            path = nx.shortest_path(G, 0, 1)
            np.savez(out_fn, FOREST_PATH=path)
        except nx.exception.NetworkXNoPath:
            if isfile(out_fn):
                os.remove(out_fn)

def plan(args, ws, reference_trial, current_trial):
    trial_str = 'ref-{}_trial-{}'.format(reference_trial, current_trial)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        _, config = parse_ompl.parse_simple(puzzle_fn)
        rel_scratch_dir = join(util.BASELINE_SCRATCH,
                               puzzle_name,
                               'pwrdtc',
                               trial_str)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        if args.only_wait:
            condor.local_wait(scratch_dir)
            continue
        key_fn = ws.screened_keyconf_prediction_file(puzzle_name)
        keys = matio.load(key_fn, key='KEYQ_OMPL')
        ninstances = keys.shape[0]
        condor_job_args = ['se3solver.py',
                'solve',
                '--cdres', config.getfloat('problem', 'collision_resolution', fallback=0.0001),
                '--replace_istate',
                'file={},key=KEYQ_OMPL,offset=$$([$(Process)]),size=1,out={}'.format(key_fn, scratch_dir),
                '--replace_gstate',
                'file={},key=KEYQ_OMPL,offset=0,size=-1,out={}'.format(key_fn, scratch_dir),
                '--trajectory_out',
                '{}/traj_$(Process).hdf5'.format(scratch_dir),
                puzzle_fn,
                args.planner_id,
                args.time_limit]
        # Note: --ec_limit is OptionVector passed to OMPL planners and
        #       must be put at the end of arguments
        ec_limit = get_ec_limit(args, ws, puzzle_name)
        condor_job_args += ['--ec_limit', str(ec_limit)]
        util.log('[baseline][{}] ec_limit {}'.format(puzzle_name, ec_limit))
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir_rel=rel_scratch_dir,
                            arguments=condor_job_args,
                            instances=ninstances,
                            wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    if args.no_wait:
        return
    if not args.only_wait:
        only_wait_args = copy.deepcopy(args)
        only_wait_args.only_wait = True
        plan(only_wait_args, ws, reference_trial=reference_trial, current_trial=current_trial)

def setup_parser(subparsers):
    p = subparsers.add_parser('baseline_pwrdtc', help='Pairwise RDT-Connect', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--no_wait', action='store_true')
    p.add_argument('--query', help='Check the connectivity', action='store_true')
    p.add_argument('--current_trial', help='Trial to run the baseline under a specific list of reference trials. Could be a list.', type=str, default='0')
    p.add_argument('--reference_trial', help='Trial that runs the RDT forest', type=str, required=True)
    p.add_argument('--nrepeats', help='Number of repeats', type=int, default=10)
    p.add_argument('--time_limit', help='Time Limit in day(s)', type=float, default=1.0)
    p.add_argument('--ec_limit', help='Override edge limit defined as PDSBloom in config',
                   type=int, default=None)
    p.add_argument('--puzzle_name', help='Only for one puzzle', default='')
    p.add_argument('dir', help='Workspace directory')

def run(args):
    args.planner_id = util.RDT_CONNECT_ALGORITHM_ID
    ws = util.Workspace(args.dir)
    ref_trial_list = util.rangestring_to_list(args.reference_trial)
    for reference_trial in ref_trial_list:
        ws.current_trial = reference_trial
        if args.current_trial is None:
            trial_list = [reference_trial]
        else:
            trial_list = util.rangestring_to_list(args.current_trial)
        for trial in trial_list:
            if args.query:
                query(args, ws, reference_trial=reference_trial, current_trial=trial)
            else:
                plan(args, ws, reference_trial=reference_trial, current_trial=trial)
