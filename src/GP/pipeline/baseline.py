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
from .file_locations import FileLocations, KEY_PRED_SCHEMES

"""
Candiate baseline planners:
    RRTConnect (0)
    RRT (1)
    RDT (15)
    RDT-Connect (17)
    PRM (7)
    BITstar (8)
    EST (6)
    LBKPIECE1 (3)
    BKPIECE1 (2)
    KPIECE1 (4)
    PDST (9)
    SBL (5)
    STRIDE (18)
    FMT (19)
    ~~LightningRetrieveRepair (20)~~ <- can't be used due to mandatory experience db

Plan:
    compare all schemes on duet-g2 and alpha-1.0
"""

def mean_ec_from_output(args, ws, puzzle_name):
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

def mean_ec_from_pflog(args, ws, puzzle_name):
    ref_trial_list = util.rangestring_to_list(args.reference_trials)
    total_ec = 0
    total_ec_time = 0.0
    valid_trials = []
    for trial in ref_trial_list:
        ws.current_trial = trial
        try:
            cur_ec = 0
            cur_ec_time = 0
            fl = FileLocations(args, ws, puzzle_name, ALGO_VERSION=6)
            for i, bloom_fn in fl.bloom_fn_gen:
                # util.log(f'loading {bloom_fn}')
                d = matio.load(bloom_fn)
                cur_ec += int(d['PF_LOG_MCHECK_N'])
                cur_ec_time += float(d['PF_LOG_MCHECK_T'])
            for i, knn_fn in fl.knn_fn_gen:
                # util.log(f'loading {knn_fn}')
                d = matio.load(knn_fn)
                cur_ec += int(d['PF_LOG_MCHECK_N'])
                cur_ec_time += float(d['PF_LOG_MCHECK_T'])
            valid_trials.append(trial)
            total_ec += cur_ec
            total_ec_time += cur_ec_time
        except Exception as e:
            util.log(f'[{puzzle_name}][trial {trial}] is incomplete, some file is missing')
    return int(total_ec/len(valid_trials)), total_ec, total_ec_time

def mean_ec(args, ws, puzzle_name):
    return mean_ec_from_pflog(args, ws, puzzle_name)

def run_baseline(args, ws):
    trial_str = 'trial-{}'.format(args.current_trial)
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        if args.reference_trials:
            ec_budget, ec_total, ec_time_sum = mean_ec(args, ws, puzzle_name)
        if args.dry:
            assert args.reference_trials
            util.ack(f'Puzzle {puzzle_name} trial {args.reference_trials} EC Budget {ec_budget} EC total {ec_total} EC time {ec_time_sum} Ave EC per sec {ec_total / (ec_time_sum/1000.0)}')
            continue
        _, config = parse_ompl.parse_simple(puzzle_fn)
        fl = FileLocations(args, ws, puzzle_name, ALGO_VERSION=6)
        for planner_id in args.planner_id:
            rel_scratch_dir = join(util.BASELINE_SCRATCH,
                                   puzzle_name,
                                   'planner-{}'.format(planner_id),
                                   f'{trial_str}.{args.scheme}')
            scratch_dir = ws.local_ws(rel_scratch_dir)
            if args.only_wait:
                condor.local_wait(scratch_dir)
                continue
            condor_job_args = ['se3solver.py',
                    'solve',
                    '--cdres', config.getfloat('problem', 'collision_resolution', fallback=0.0001),
                    '--trajectory_out', f'{scratch_dir}/traj_$(Process).npz']
            if args.reference_trials:
                condor_job_args += ['--ec_budget', str(ec_budget)]
                util.log('[baseline][{}] ec_budget {}'.format(puzzle_name, ec_budget))
                ex_args = []
                if args.use_roots_from_reference_trials:
                    ref_trial_list = util.rangestring_to_list(args.reference_trials)
                    for trial in ref_trial_list:
                        fn = fl.get_cmb_screened_key_fn(trial)
                        # util.log(f'Checking {fn}')
                        if os.path.isfile(fn):
                            ex_args += [fn]
                if ex_args:
                    condor_job_args += ['--use_roots_from'] + ex_args + ['--']
            condor_job_args += [puzzle_fn,
                    planner_id,
                    args.time_limit]
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=rel_scratch_dir,
                                arguments=condor_job_args,
                                instances=args.nrepeats,
                                wait=False,
                                dryrun=args.no_submit) # do NOT wait here, we have to submit EVERY puzzle at once
    if args.dry or args.no_submit:
        return
    if args.no_wait:
        return
    if not args.only_wait:
        only_wait_args = copy.deepcopy(args)
        only_wait_args.only_wait = True
        run_baseline(only_wait_args, ws)

def setup_parser(subparsers):
    p = subparsers.add_parser('baseline', help='Solve all testing puzzle with baseline algorithms', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--no_wait', action='store_true')
    p.add_argument('--no_submit', action='store_true')
    p.add_argument('--use_all_planners', action='store_true')
    p.add_argument('--reference_trials', help='Use existing trials as reference to set --', type=str, default=None)
    p.add_argument('--use_roots_from_reference_trials', help='Use roots generated from reference trials', action='store_true')
    p.add_argument('--nrepeats', help='Number of repeats', type=int, default=100)
    p.add_argument('--remote_hosts', help='Run the baseline remotely', nargs='*', type=str, default=None)
    p.add_argument('--remote_host_shift', help='shift the list of remote hosts, for load balance', type=int, default=None)
    p.add_argument('--planner_id', help='Planner ID', nargs='*', type=int,
                   default=[util.RDT_FOREST_ALGORITHM_ID])
    p.add_argument('--time_limit', help='Time Limit in day(s)', type=float, default=1.0)
    p.add_argument('--scheme', help='Choose key prediction scheme', choices=KEY_PRED_SCHEMES, default='cmb')
    p.add_argument('--dry', help='Only print ec_budget and exit', action='store_true')
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=int, required=True)
    p.add_argument('--override_config', help='Override the options. Syntax: SECTION.OPTION=VALUE Separated with semicolon (;)', type=str, default=None)
    p.add_argument('dirs', help='Workspace directory', nargs='+')

def run(args):
    '''
    if args.reference_trials:
        assert args.planner_id == util.RDT_FOREST_ALGORITHM_ID, 'Only RDT implemented --bloom_limit'
    '''
    if args.use_all_planners:
        assert args.planner_id == [util.RDT_FOREST_ALGORITHM_ID], '--use_all_planners should not be used with --planner_id. It overrides the latter'
        import pyse3ompl as plan
        args.planner_id = [
                plan.PLANNER_RRT_CONNECT,
                plan.PLANNER_RRT,
                plan.PLANNER_RDT,
                plan.PLANNER_RDT_CONNECT,
                plan.PLANNER_PRM,
                plan.PLANNER_BITstar,
                plan.PLANNER_EST,
                plan.PLANNER_LBKPIECE1,
                plan.PLANNER_BKPIECE1,
                plan.PLANNER_KPIECE1,
                plan.PLANNER_PDST,
                plan.PLANNER_SBL,
                plan.PLANNER_STRIDE,
                plan.PLANNER_FMT,
                ]
    NHOST = len(args.remote_hosts) if args.remote_hosts is not None else 1
    host_index = NHOST//2 if args.remote_host_shift is None else args.remote_host_shift
    for d in args.dirs:
        args.dir = d
        ws = util.create_workspace_from_args(args)
        print(args)
        if args.remote_hosts is None:
            run_baseline(args, ws)
            return
        for planner in args.planner_id:
            host = args.remote_hosts[host_index % NHOST]
            host_index += 1
            extra_args = ''
            if args.reference_trials:
                extra_args += f'--reference_trials {args.reference_trials} '
            extra_args += f'--nrepeats {args.nrepeats} '
            extra_args += f'--planner_id {planner} '
            extra_args += f'--time_limit {args.time_limit} '
            extra_args += f'--scheme {args.scheme} '
            extra_args += f'--no_wait '
            if args.use_roots_from_reference_trials:
                import pyse3ompl as plan
                assert args.planner_id == [plan.PLANNER_PRM]
                assert args.reference_trials is not None
                extra_args += f'--use_roots_from_reference_trials '
            # print(host)
            ws.remote_command(host=host, exec_path=ws.condor_exec(), ws_path=ws.condor_ws(),
                              pipeline_part='baseline', cmd=None,
                              in_tmux=False, with_trial=True,
                              extra_args=extra_args)
