#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join, isdir, isfile
import subprocess
import pathlib
import numpy as np
import copy

import util
import se3solver
import condor
import matio

#
# Functions to process workspaces locally
#

def _trial_id(ws, trial):
    max_trial = ws.config.getint('Solver', 'Trials', fallback=1)
    util.padded(trial, max(trial, max_trial))

def _puzzle_pds(ws, puzzle_name, trial):
    fn = ws.local_ws(util.TESTING_DIR,
                     puzzle_name,
                     util.PDS_SUBDIR,
                     '{}.npz'.format(_trial_id(ws, trial)))
    return fn

def test_puzzle_generator(ws):
    for ent in os.listdir(ws.local_ws(util.TESTING_DIR)):
        puzzle_fn = ws.local_ws(util.TESTING_DIR, ent, 'puzzle.cfg')
        if not isdir(ent) or not isfile(puzzle_fn):
            util.log("Cannot find puzzle file {}, continue to next dir".format(puzzle_fn))
            continue
        yield puzzle_fn, ent

def sample_pds(args, ws):
    max_trial = ws.config.getint('Solver', 'Trials', fallback=1)
    nsamples = ws.config.getint('Solver', 'PDSSize')
    for puzzle_fn, puzzle_name in test_puzzle_generator(ws):
        driver = se3solver.create_driver(puzzle=puzzle_fn,
                planner_id=se3solver.PLANNER_RDT,
                sampler_id=0)
        uw = ws.create_unit_world(puzzle_fn)
        for i in range(max_trial):
            Q = driver.presample(nsamples)
            uQ = ws.translate_ompl_to_unit(Q)
            n = Q.shape[0]
            QF = np.zeros((n, 1), dtype=np.uint32)
            for uq in uQ:
                if uw.is_disentangled(uq):
                    QF[j] = PDS_FLAG_TERMINATE
            fn = _puzzle_pds(ws, puzzle_name, i)
            np.savez(fn, Q=Q, QF=QF)

def forest_rdt(args, ws):
    trial_str = 'trial-{}'.format(_trial_id(args.current_trial))
    for puzzle_fn, puzzle_name in test_puzzle_generator(ws):
        scratch_dir = ws.local_ws(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        if args.only_wait:
            condor.local_wait(scratch_dir)
            continue
        args = ['facade.py',
                'solve',
                '--stage', 'forest_rdt'
                '--current_trial', args.current_trial,
                '--task_id', '$(Process)']
        keys = matio.load(ws.local_ws(util.TESTING_DIR, puzzle_name, util.KEY_PREDICTION))
        condor.local_submit(ws,
                            '/usr/bin/python3',
                            iodir=scratch_dir,
                            arguments=args,
                            instances=keys['KEYQ_OMPL'].shape[0],
                            wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    only_wait_args = copy.deepcopy(args)
    only_wait_args.only_wait = True
    forest_rdt(only_wait_args, ws)

function_dict = {
        'sample_pds' : sample_pds,
        'forest_rdt' : forest_rdt,
        'forest_edges' : forest_rdt,
        'connect_forest' : connect_forest,
}

def setup_parser(subparsers):
    p = subparsers.add('solve', help='Final step to solve the puzzle')
    p.add_argument('--stage', choices=list(function_dict.keys()), default='')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--task_id', help='Feed $(Process) from HTCondor', type=int, default=None)
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=int, default=0)

def run(args):
    if args.stage in function_dict:
        ws = util.Workspace(args.dir)
        function_dict[args.stage](args, ws)
    else:
        print("Unknown solve pipeline stage {}".format(args.stage))

#
# Automatic functions start here
#
def _remote_command(ws, cmd, auto_retry=True):
    ws.remote_command(ws.condor_host,
                      ws.condor_exec,
                      'preprocess_surface', cmd, auto_retry=auto_retry)

def remote_sample_pds(ws):
    _remote_command(ws, 'sample_pds')

def remote_forest_rdt(ws):
    _remote_command(ws, 'forest_rdt')

def remote_forest_edges(ws):
    _remote_command(ws, 'forest_edges')

def remote_connect_forest(ws):
    _remote_command(ws, 'connect_forest')

def collect_stages():
    return [ ('upload_keyconf_to_condor', lambda ws: ws.deploy_to_condor(util.TESTING_DIR + '/')),
             ('sample_pds', remote_sample_pds),
             ('forest_rdt', remote_forest_rdt),
             ('forest_edges', remote_forest_edges),
             ('connect_forest', remote_connect_forest),
           ]

def autorun(args):
    ws = util.Workspace(args.dir)
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)
