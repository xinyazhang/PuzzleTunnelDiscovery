#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing
import copy

from . import solve
from . import choice_formatter
from . import util
from . import parse_ompl
from . import matio
try:
    import pygeokey
except ImportError as e:
    util.warn("[WARNING] CANNOT IMPORT pygeokey. This node is incapable of geometric based prediction")

class WorkerArgs(object):
    pass

def _get_task_args(ws, per_geometry):
    task_args = []
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        cfg, config = parse_ompl.parse_simple(puzzle_fn)
        wag = WorkerArgs()
        wag.dir = ws.dir
        wag.current_trial = ws.current_trial
        wag.puzzle_fn = puzzle_fn
        wag.puzzle_name = puzzle_name
        wag.env_fn = cfg.env_fn
        wag.rob_fn = cfg.rob_fn
        if per_geometry:
            wag.geo_type = 'env'
            wag.geo_fn = cfg.env_fn
            task_args.append(copy.deepcopy(wag))
            wag.geo_type = 'rob'
            wag.geo_fn = cfg.rob_fn
            task_args.append(copy.deepcopy(wag))
        else:
            task_args.append(copy.deepcopy(wag))
    return task_args

def _sample_key_point_worker(wag):
    ws = util.Workspace(wag.dir)
    ws.current_trial = wag.current_trial
    kpp = pygeokey.KeyPointProber(wag.geo_fn)
    natt = ws.config.getint('GeometriK', 'KeyPointAttempts')
    util.log("[sample_key_point] probing {} for {} attempts".format(wag.geo_fn, natt))
    pts = kpp.probe_key_points(natt)
    kps_fn = ws.keypoint_prediction_file(wag.puzzle_name, wag.geo_type)
    util.log("[sample_key_point] writing {} points to {}".format(pts.shape[0], kps_fn))
    np.savez(kps_fn, KEY_POINT_AMBIENT=pts)

def sample_key_point(args, ws):
    task_args = _get_task_args(ws, per_geometry=True)
    #for wag in task_args:
    #    _sample_key_point_worker(wag)
    pcpu = multiprocessing.Pool()
    pcpu.map(_sample_key_point_worker, task_args)

def _sample_key_conf_worker(wag):
    ws = util.Workspace(wag.dir)
    ws.current_trial = wag.current_trial
    kfn = ws.keyconf_prediction_file(wag.puzzle_name)
    util.log('[sample_key_conf] trial {}'.format(ws.current_trial))
    util.log('[sample_key_conf] sampling to {}'.format(kfn))
    env_kps_fn = ws.keypoint_prediction_file(wag.puzzle_name, 'env')
    rob_kps_fn = ws.keypoint_prediction_file(wag.puzzle_name, 'rob')
    env_kps = matio.load(env_kps_fn)['KEY_POINT_AMBIENT']
    rob_kps = matio.load(rob_kps_fn)['KEY_POINT_AMBIENT']
    ks = pygeokey.KeySampler(wag.env_fn, wag.rob_fn)
    kqs, _, _ = ks.get_all_key_configs(env_kps, rob_kps,
                                       ws.config.getint('GeometriK', 'KeyConfigRotations'))
    uw = util.create_unit_world(wag.puzzle_fn)
    cfg, config = parse_ompl.parse_simple(wag.puzzle_fn)
    iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
    gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
    ompl_q = uw.translate_vanilla_to_ompl(kqs)
    ompl_q = np.concatenate((iq, gq, ompl_q), axis=0)
    kfn = ws.keyconf_prediction_file(wag.puzzle_name)
    #np.savez(kfn, KEYQ_AMBIENT_NOIG=kqs, KEYQ_OMPL=ompl_q)
    np.savez(kfn, KEYQ_OMPL=ompl_q)
    util.log('[sample_key_conf] save key confs to {}'.format(kfn))

def sample_key_conf(args, ws):
    task_args = _get_task_args(ws, per_geometry=False)
    for wag in task_args:
         _sample_key_conf_worker(wag)
    # pcpu = multiprocessing.Pool()
    # pcpu.map(_sample_key_conf_worker, task_args)

def deploy_geometrik_to_condor(args, ws):
    ws.deploy_to_condor(util.WORKSPACE_SIGNATURE_FILE,
                        util.WORKSPACE_CONFIG_FILE,
                        util.CONDOR_TEMPLATE,
                        util.TESTING_DIR+'/')

function_dict = {
        'sample_key_point' : sample_key_point,
        'sample_key_conf' : sample_key_conf,
        'deploy_geometrik_to_condor' : deploy_geometrik_to_condor,
}

def setup_parser(subparsers):
    p = subparsers.add_parser('geometrik', help='Sample Key configuration from Geometric features',
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
        print("Unknown geometrik pipeline stage {}".format(args.stage))

def autorun(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)

def collect_stages():
    ret = [
            ('sample_key_point', lambda ws: sample_key_point(None, ws)),
            ('sample_key_conf', lambda ws: sample_key_conf(None, ws)),
            ('deploy_geometrik_to_condor', lambda ws: deploy_geometrik_to_condor(None, ws))
          ]
    return ret
