#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import copy
import argparse
import numpy as np

from . import util
from . import condor
from . import parse_ompl
from . import matio

def mean_ec(args, ws, puzzle_name):
    ref_trial_list = util.rangestring_to_list(args.reference_trials)
    all_roots = []
    all_pds = []
    for trial in ref_trial_list:
        ws.current_trial = trial
        pds_fn = ws.local_ws(util.SOLVER_SCRATCH,
                             puzzle_name,
                             util.PDS_SUBDIR,
                             '{}.npz'.format(trial))
        if not os.path.exists(pds_fn):
            continue
        puzzle_pds = matio.load(pds_fn)['Q'].shape[0]
        kq_fn = ws.keyconf_prediction_file(puzzle_name)
        puzzle_roots = matio.load(kq_fn)['KEYQ_OMPL'].shape[0]
        all_roots.append(puzzle_roots)
        all_pds.append(puzzle_pds)
    mean_roots = np.mean(np.array(all_roots))
    mean_pds = np.mean(np.array(all_pds))
    return int(mean_roots * mean_pds)

def plan(args, ws):
    trial_str = 'trial-{}'.format(args.current_trial)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        _, config = parse_ompl.parse_simple(puzzle_fn)
        rel_scratch_dir = join(util.BASELINE_SCRATCH,
                               puzzle_name,
                               'planner-{}'.format(args.planner_id),
                               trial_str)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        if args.only_wait:
            condor.local_wait(scratch_dir)
            continue
        condor_job_args = ['se3solver.py',
                'solve',
                '--cdres', config.getfloat('problem', 'collision_resolution', fallback=0.0001),
                '--trajectory_out', '{}/traj_$(Process).npz'.format(scratch_dir),
                puzzle_fn,
                args.planner_id,
                args.time_limit]
        if args.reference_trials:
            ec_limit = mean_ec(args, ws, puzzle_name)
            condor_job_args += ['--ec_limit', str(ec_limit)]
            util.log('[baseline][{}] ec_limit {}'.format(puzzle_name, ec_limit))
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir_rel=rel_scratch_dir,
                            arguments=condor_job_args,
                            instances=args.nrepeats,
                            wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    if args.no_wait:
        return
    if not args.only_wait:
        only_wait_args = copy.deepcopy(args)
        only_wait_args.only_wait = True
        plan(only_wait_args, ws)

def setup_parser(subparsers):
    p = subparsers.add_parser('baseline', help='Solve all testing puzzle with baseline algorithms', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--no_wait', action='store_true')
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=int, required=True)
    p.add_argument('--reference_trials', help='Use existing trials as reference to set --', type=str, default=None)
    p.add_argument('--nrepeats', help='Number of repeats', type=int, default=100)
    p.add_argument('--planner_id', help='Planner ID', type=int, default=util.RDT_FOREST_ALGORITHM_ID)
    p.add_argument('--time_limit', help='Time Limit in day(s)', type=float, default=1.0)
    p.add_argument('dir', help='Workspace directory')

def run(args):
    if args.reference_trials:
        assert args.planner_id == util.RDT_FOREST_ALGORITHM_ID, 'Only RDT implemented --bloom_limit'
    ws = util.Workspace(args.dir)
    plan(args, ws)
