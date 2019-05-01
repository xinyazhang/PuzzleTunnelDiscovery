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
from scipy.misc import imsave, imread
from progressbar import progressbar

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
import pyosr

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

def _predict_atlas2prim(tup):
    ws_dir, puzzle_fn, puzzle_name = tup
    ws = util.Workspace(ws_dir)
    r = util.create_offscreen_renderer(puzzle_fn, ws.chart_resolution)
    r.uv_feedback = True
    r.avi = False
    for geo_type,flags in zip(['rob', 'env'], [pyosr.Renderer.NO_SCENE_RENDERING, pyosr.Renderer.NO_ROBOT_RENDERING]):
        r.render_mvrgbd(pyosr.Renderer.UV_MAPPINNG_RENDERING|flags)
        atlas2prim = np.copy(r.mvpid.reshape((r.pbufferWidth, r.pbufferHeight)))
        #imsave(geo_type+'-a2p-nt.png', atlas2prim) # This is for debugging
        atlas2prim = texture_format.framebuffer_to_file(atlas2prim)
        atlas2uv = np.copy(r.mvuv.reshape((r.pbufferWidth, r.pbufferHeight, 2)))
        atlas2uv = texture_format.framebuffer_to_file(atlas2uv)
        np.savez(ws.local_ws(util.TESTING_DIR, puzzle_name, geo_type+'-a2p.npz'),
                 PRIM=atlas2prim,
                 UV=atlas2uv)
        imsave(ws.local_ws(util.TESTING_DIR, puzzle_name, geo_type+'-a2p.png'), atlas2prim) # This is for debugging

def _predict_worker(tup):
    ws_dir, puzzle_fn, puzzle_name = tup
    ws = util.Workspace(ws_dir)
    uw = util.create_unit_world(puzzle_fn)
    rob_sampler = atlas.AtlasSampler(ws.local_ws(util.TESTING_DIR, puzzle_name, 'rob-a2p.npz'),
                                     ws.local_ws(util.TESTING_DIR, puzzle_name, 'rob-atex.npz'),
                                     'rob', uw.GEO_ROB)
    env_sampler = atlas.AtlasSampler(ws.local_ws(util.TESTING_DIR, puzzle_name, 'env-a2p.npz'),
                                     ws.local_ws(util.TESTING_DIR, puzzle_name, 'env-atex.npz'),
                                     'env', uw.GEO_ENV)
    key_conf = []
    nrot = ws.config.getint('Prediction', 'NumberOfRotations')
    margin = ws.config.getfloat('Prediction', 'Margin')
    batch_size = ws.config.getint('Prediction', 'SurfacePairsToSample')
    for i in progressbar(range(batch_size)):
        tup1 = rob_sampler.sample(uw)
        tup2 = env_sampler.sample(uw)
        qs_raw = uw.enum_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1],
                                            margin,
                                            denominator=nrot)
        qs = [q for q in qs_raw if not uw.is_disentangled(q)] # Trim disentangled state
        for q in qs:
            key_conf.append(q)
    cfg, config = parse_ompl.parse_simple(puzzle_fn)
    iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
    gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
    util.log("[predict_keyconf(worker)] sampled {} keyconf from puzzle {}".format(len(key_conf), puzzle_name))
    qs_ompl = uw.translate_unit_to_ompl(key_conf)
    ompl_q = np.concatenate((iq, gq, qs_ompl), axis=0)
    key_fn = ws.local_ws(util.TESTING_DIR, puzzle_name, util.KEY_PREDICTION)
    util.log("[predict_keyconf(worker)] save to key file {}".format(key_fn))
    np.savez(key_fn, KEYQ_OMPL=ompl_q)

def predict_keyconf(args, ws):
    task_tup = []
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        task_tup.append((ws.dir, puzzle_fn, puzzle_name))
        util.log('[predict_keyconf] found puzzle {} at {}'.format(puzzle_name, puzzle_fn))
    if 'auto' == ws.config.get('Prediction', 'NumberOfPredictionProcesses'):
        ncpu = None
    else:
        ncpu = ws.config.getint('Prediction', 'NumberOfPredictionProcesses')
    pgpu = multiprocessing.Pool(1)
    pgpu.map(_predict_atlas2prim, task_tup)
    pcpu = multiprocessing.Pool(ncpu)
    pcpu.map(_predict_worker, task_tup)

class TmpDriverArgs(object):
    pass

def sample_pds(args, ws):
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

def forest_rdt(args, ws):
    trial_str = 'trial-{}'.format(_trial_id(ws, args.current_trial))
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
        key_fn = ws.local_ws(util.TESTING_DIR, puzzle_name, util.KEY_PREDICTION)
        condor_job_args = ['se3solver.py',
                'solve', puzzle_fn,
                util.RDT_FOREST_ALGORITHM_ID,
                ws.config.getfloat('Solver', 'TimeThreshold'),
                '--samset', _puzzle_pds(ws, puzzle_name, args.current_trial),
                '--replace_istate',
                'file={},offset=$$([$(Process)]),size=1,out={}'.format(key_fn, scratch_dir)
                          ]
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
    raise NotImplemented("forest_edges")

def connect_forest(args, ws):
    raise NotImplemented("connect_forest")

function_dict = {
        'predict_keyconf' : predict_keyconf,
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
    ret = [ ('predict_keyconf', lambda ws: predict_keyconf(None, ws)),
            ('upload_keyconf_to_condor', lambda ws: ws.deploy_to_condor(util.TESTING_DIR + '/')),
            ('sample_pds', remote_sample_pds),
            ('forest_rdt', remote_forest_rdt),
            ('forest_edges', remote_forest_edges),
            ('connect_forest', remote_connect_forest),
          ]
    return ret

def autorun(args):
    ws = util.Workspace(args.dir)
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)
