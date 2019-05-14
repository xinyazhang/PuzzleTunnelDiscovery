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
from . import choice_formatter
try:
    from . import se3solver
except ImportError as e:
    util.warn(str(e))
    util.warn("[WARNING] CANNOT IMPORT se3solver. Some function will be disabled and the pipeline is broken")
from . import condor
from . import matio
from . import atlas
from . import texture_format
from . import parse_ompl

#
# Functions to process workspaces locally
#

def _trial_id(ws, trial):
    max_trial = ws.config.getint('Solver', 'Trials', fallback=1)
    # util.log('[_trial_id] trial {} max_trial {}'.format(trial, max_trial))
    return util.padded(trial, max(trial, max_trial))

def _puzzle_pds(ws, puzzle_name, trial):
    pds_dir = ws.local_ws(util.SOLVER_SCRATCH,
                          puzzle_name,
                          util.PDS_SUBDIR)
    os.makedirs(pds_dir, exist_ok=True)
    fn ='{}.npz'.format(_trial_id(ws, trial))
    return join(pds_dir, fn)

def _rel_bloom_scratch(ws, puzzle_fn, trial):
    return join(util.SOLVER_SCRATCH, puzzle_fn, util.PDS_SUBDIR, 'bloom-'+_trial_id(ws, trial))

class TmpDriverArgs(object):
    pass

def _sample_pds_old(args, ws):
    if 'pipeline.se3solver' not in sys.modules:
        raise RuntimeError("se3solver is not loaded")
    max_trial = ws.config.getint('Solver', 'Trials', fallback=1)
    nsamples = ws.config.getint('Solver', 'PDSSize')
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        util.log('[sample_pds] sampling puzzle {}'.format(puzzle_name))
        driver_args = TmpDriverArgs()
        driver_args.puzzle = puzzle_fn
        driver_args.planner_id = se3solver.PLANNER_PRM
        driver_args.sampler_id = 0
        driver = se3solver.create_driver(driver_args)
        uw = util.create_unit_world(puzzle_fn)
        for i in range(max_trial):
            # util.log('[sample_pds] trial id {}'.format(_trial_id(ws, 0)))
            Q = driver.presample(nsamples)
            uQ = uw.translate_ompl_to_unit(Q)
            n = Q.shape[0]
            QF = np.zeros((n, 1), dtype=np.uint32)
            for j, uq in enumerate(uQ):
                if uw.is_disentangled(uq):
                    QF[j] = se3solver.PDS_FLAG_TERMINATE
            fn = _puzzle_pds(ws, puzzle_name, i)
            np.savez(fn, Q=Q, QF=QF)
            util.log('[sample_pds] samples stored at {}'.format(fn))

# Bloom from roots
def sample_pds(args, ws):
    if not args.only_wait:
        for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
            rel_bloom = _rel_bloom_scratch(ws, puzzle_name, ws.current_trial)
            util.log('[sample_pds]  rel_bloom {}'.format(rel_bloom))
            scratch_dir = ws.local_ws(rel_bloom)
            key_fn = ws.keyconf_prediction_file(puzzle_name)
            if not os.path.exists(key_fn):
                util.warn("[sample_pds] forest root file {} does not exist")
                probe_trial = ws.current_trial - 1
                while probe_trial >= 0:
                    key_fn_0 = ws.keyconf_prediction_file(puzzle_name, trial_override=probe_trial)
                    if os.path.exists(key_fn_0):
                        break
                    probe_trial -= 1
                shutil.copy(key_fn_0, key_fn)
                util.warn("[sample_pds] copy {} to {} as forest root file".format(key_fn_0, key_fn))
            condor_job_args = ['se3solver.py',
                    'solve',
                    '--replace_istate',
                    'file={},offset=$$([$(Process)]),size=1,out={}'.format(key_fn, scratch_dir),
                    '--bloom_out',
                    join(scratch_dir, 'bloom-from_$(Process).npz'),
                    puzzle_fn,
                    util.RDT_FOREST_ALGORITHM_ID,
                    1.0,
                    '--bloom_limit',
                    ws.config.getint('Solver', 'PDSBloom')
                              ]
            keys = matio.load(key_fn)
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=rel_bloom,
                                arguments=condor_job_args,
                                instances=keys['KEYQ_OMPL'].shape[0],
                                wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        rel_bloom = _rel_bloom_scratch(ws, puzzle_name, ws.current_trial)
        scratch_dir = ws.local_ws(rel_bloom)
        condor.local_wait(scratch_dir)
        pds_fn = _puzzle_pds(ws, puzzle_name, ws.current_trial)
        Q_list = []
        fn_list = sorted(pathlib.Path(scratch_dir).glob("bloom-from_*.npz"))
        for fn in progressbar(fn_list):
            d = matio.load(fn)
            s = d['BLOOM'].shape
            if s[0] == 0:
                continue
            assert s[1] == 7, "{}'s shape is {}".format(fn, s)
            Q_list.append(d['BLOOM'])
        Q = np.concatenate(Q_list, axis=0)
        uw = util.create_unit_world(puzzle_fn)
        uQ = uw.translate_ompl_to_unit(Q)
        QF = np.zeros((Q.shape[0], 1), dtype=np.uint32)
        for j, uq in enumerate(uQ):
            if uw.is_disentangled(uq):
                QF[j] = se3solver.PDS_FLAG_TERMINATE
        np.savez(pds_fn, Q=Q, QF=QF)
        util.log('[sample_pds] samples stored at {}'.format(pds_fn))

def forest_rdt(args, ws):
    trial_str = 'trial-{}'.format(_trial_id(ws, ws.current_trial))
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        scratch_dir = ws.local_ws(rel_scratch_dir)
        if args.only_wait:
            condor.local_wait(scratch_dir)
            continue
        # se3solver.py solve
        # /u/zxy/applique/puzzle-geometry/puzzles/claw/claw.cfg 15 1.00
        # --samset mkobs-claw/pds/1m.0.npz --replace_istate
        # file=mkobs-claw/key.ompl.withig.npz,offset=$$([$(Process)*1]),size=1,out=mkobs-claw/trial-1/pdsrdt/
        #key_fn = ws.local_ws(util.TESTING_DIR, puzzle_name, util.KEY_PREDICTION)

        # Note: no need to probe the key configuration, which has been done in sample_pds and 
        #       ws.keyconf_prediction_file should always point to a valid keyconf
        key_fn = ws.keyconf_prediction_file(puzzle_name)
        condor_job_args = ['se3solver.py',
                'solve',
                '--samset', _puzzle_pds(ws, puzzle_name, ws.current_trial),
                '--replace_istate',
                'file={},offset=$$([$(Process)]),size=1,out={}'.format(key_fn, scratch_dir),
                puzzle_fn,
                util.RDT_FOREST_ALGORITHM_ID,
                1.0]
        keys = matio.load(key_fn)
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir_rel=rel_scratch_dir,
                            arguments=condor_job_args,
                            instances=keys['KEYQ_OMPL'].shape[0],
                            wait=False) # do NOT wait here, we have to submit EVERY puzzle at once
    if not args.only_wait:
        only_wait_args = copy.deepcopy(args)
        only_wait_args.only_wait = True
        forest_rdt(only_wait_args, ws)

def forest_edges(args, ws):
#pdsrdt-g9ae-1-edges.hdf5:
#    ./pds_edge.py --pdsflags dual/pdsrdt-g9ae-1/pds-4m.0.npz --out pdsrdt-g9ae-1-edges.hdf5 `ls -v dual/pdsrdt-g9ae-1/pdsrdt/ssc-*.mat`
    trial_str = 'trial-{}'.format(_trial_id(ws, ws.current_trial))
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        rel_scratch_dir = os.path.join(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        shell_script = './pds_edge.py --pdsflags '
        shell_script += _puzzle_pds(ws, puzzle_name, ws.current_trial)
        shell_script += ' --out {}'.format(ws.local_ws(rel_scratch_dir, 'edges.hdf5'))
        shell_script += ' `ls -v {}/ssc-*.mat`'.format(ws.local_ws(rel_scratch_dir))
        util.shell(['bash', '-c', shell_script])

def connect_forest(args, ws):
# pdsrdt-g9ae-1-path-1.txt: pdsrdt-g9ae-1-edges-1.hdf5
#   ./forest_dijkstra.py --indir dual/pdsrdt-g9ae-1/pdsrdt-1/ \
#               --forest_edge pdsrdt-g9ae-1-edges-1.hdf5 \
#               --rootf dual/pdsrdt-g9ae-1/postscreen-1_withig.npz \
#               --pdsf dual/pdsrdt-g9ae-1/pds-4m.0.npz \
#               --out pdsrdt-g9ae-1-path-1.txt
#
    trial_str = 'trial-{}'.format(_trial_id(ws, ws.current_trial))
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        # key_fn = ws.local_ws(util.TESTING_DIR, puzzle_name, util.KEY_PREDICTION)
        key_fn = ws.keyconf_prediction_file(puzzle_name)
        rel_scratch_dir = join(util.SOLVER_SCRATCH, puzzle_name, trial_str)
        rel_edges = join(rel_scratch_dir, 'edges.hdf5')
        path_out = ws.local_ws(rel_scratch_dir, 'path.txt')
        shell_script = './forest_dijkstra.py'
        shell_script += ' --indir '
        shell_script += ws.local_ws(rel_scratch_dir)
        shell_script += ' --forest_edge '
        shell_script += ws.local_ws(rel_edges)
        shell_script += ' --rootf '
        shell_script += key_fn
        shell_script += ' --pdsf '
        shell_script += _puzzle_pds(ws, puzzle_name, ws.current_trial)
        shell_script += ' --out {}'.format(path_out)
        ret = util.shell(['bash', '-c', shell_script])
        if ret != 0:
            util.fatal("[solve] FALIED TO SOLVE PUZZLE {}".format(puzzle_name))
            continue
        util.ack("Saving OMPL solution of {} to {}".format(puzzle_name, path_out))

        ompl_q = matio.load(path_out)
        uw = util.create_unit_world(puzzle_fn)
        unit_q = uw.translate_ompl_to_unit(ompl_q)
        sol_out = ws.solution_file(puzzle_name, type_name='unit')
        matio.savetxt(sol_out, unit_q)
        util.ack("Saving UNIT solution of {} to {}".format(puzzle_name, sol_out))

function_dict = {
        'sample_pds' : sample_pds,
        'forest_rdt' : forest_rdt,
        'forest_edges' : forest_edges,
        'connect_forest' : connect_forest,
}

def setup_parser(subparsers):
    p = subparsers.add_parser('solve', help='Final step to solve the puzzle',
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('--stage',
                   choices=list(function_dict.keys()),
                   help='R|Possible stages:\n'+'\n'.join(list(function_dict.keys())),
                   default='',
                   metavar='')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--task_id', help='Feed $(Process) from HTCondor', type=int, default=None)
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=int, default=0)
    p.add_argument('dir', help='Workspace directory')

def run(args):
    if args.stage in function_dict:
        ws = util.Workspace(args.dir)
        ws.current_trial = args.current_trial
        function_dict[args.stage](args, ws)
    else:
        print("Unknown solve pipeline stage {}".format(args.stage))

#
# Automatic functions start here
#
def _remote_command(ws, cmd, auto_retry=True):
    ws.remote_command(ws.condor_host,
                      ws.condor_exec(),
                      ws.condor_ws(),
                      'solve', cmd, auto_retry=auto_retry,
                      with_trial=True)

def remote_sample_pds(ws):
    _remote_command(ws, 'sample_pds')

def remote_forest_rdt(ws):
    _remote_command(ws, 'forest_rdt')

def remote_forest_edges(ws):
    _remote_command(ws, 'forest_edges')

def remote_connect_forest(ws):
    _remote_command(ws, 'connect_forest')

def collect_stages():
    ret = [('sample_pds', remote_sample_pds),
            ('forest_rdt', remote_forest_rdt),
            ('forest_edges', remote_forest_edges),
            ('connect_forest', remote_connect_forest),
          ]
    return ret

def autorun(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)
