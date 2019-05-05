#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import copy

from . import util
from . import condor

def plan(args, ws):
    trial_str = 'trial-{}'.format(args.current_trial)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
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
                puzzle_fn,
                util.RDT_FOREST_ALGORITHM_ID,
                0.1]
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir_rel=rel_scratch_dir,
                            arguments=condor_job_args,
                            instances=args.nrepeats,
                            wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    if not args.only_wait:
        only_wait_args = copy.deepcopy(args)
        only_wait_args.only_wait = True
        plan(only_wait_args, ws)

def setup_parser(subparsers):
    p = subparsers.add_parser('baseline', help='Final step to solve the puzzle')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=int, required=True)
    p.add_argument('--nrepeats', help='Number of repeats', type=int, default=100)
    p.add_argument('--planner_id', help='Planner ID', type=int, default=util.RDT_FOREST_ALGORITHM_ID)
    p.add_argument('--time_limit', help='Time Limit in day(s)', type=float, default=0.1)
    p.add_argument('dir', help='Workspace directory')

def run(args):
    ws = util.Workspace(args.dir)
    plan(args, ws)
