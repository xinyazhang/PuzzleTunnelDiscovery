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
    #for i in progressbar(range(batch_size)):
    with ProgressBar(max_value=batch_size) as bar:
        while True:
            tup1 = rob_sampler.sample(uw)
            tup2 = env_sampler.sample(uw)
            qs_raw = uw.enum_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1],
                                                margin,
                                                denominator=nrot,
                                                only_median=True)
            qs = [q for q in qs_raw if not uw.is_disentangled(q)] # Trim disentangled state
            for q in qs:
                key_conf.append(q)
                bar.update(min(batch_size, len(key_conf)))
            if len(key_conf) > batch_size:
                break
    cfg, config = parse_ompl.parse_simple(puzzle_fn)
    iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
    gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
    util.log("[predict_keyconf(worker)] sampled {} keyconf from puzzle {}".format(len(key_conf), puzzle_name))
    qs_ompl = uw.translate_unit_to_ompl(key_conf)
    ompl_q = np.concatenate((iq, gq, qs_ompl), axis=0)
    key_fn = ws.keyconf_prediction_file(puzzle_name)
    util.log("[predict_keyconf(worker)] save to key file {}".format(key_fn))
    unit_q = uw.translate_ompl_to_unit(ompl_q)
    np.savez(key_fn, KEYQ_OMPL=ompl_q, KEYQ_UNIT=unit_q)


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


function_dict = {
        'predict_keyconf' : predict_keyconf,
}


def setup_parser(subparsers):
    p = subparsers.add_parser('keyconf', help='Sample key configuration from surface distribution',
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('--stage',
                   choices=list(function_dict.keys()),
                   help='R|Possible stages:\n'+'\n'.join(list(function_dict.keys())),
                   default='',
                   metavar='')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--current_trial', help='Trial to solve the puzzle', type=int, default=0)
    p.add_argument('dir', help='Workspace directory')


def run(args):
    if args.stage in function_dict:
        ws = util.Workspace(args.dir)
        ws.current_trial = args.current_trial
        function_dict[args.stage](args, ws)
    else:
        print("Unknown keyconf pipeline stage {}".format(args.stage))


#
# Automatic functions start here
#
def collect_stages():
    ret = [ ('predict_keyconf', lambda ws: predict_keyconf(None, ws)),
            ('upload_keyconf_to_condor', lambda ws: ws.deploy_to_condor(util.TESTING_DIR + '/'))
          ]
    return ret

def autorun(args):
    ws = util.Workspace(args.dir)
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)
