#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
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
        wag.refined_env_fn = cfg.refined_env_fn
        wag.refined_rob_fn = cfg.refined_rob_fn
        if per_geometry:
            wag.geo_type = 'env'
            wag.geo_fn = cfg.env_fn
            wag.refined_geo_fn = cfg.refined_env_fn
            task_args.append(copy.deepcopy(wag))
            wag.geo_type = 'rob'
            wag.geo_fn = cfg.rob_fn
            wag.refined_geo_fn = cfg.refined_rob_fn
            task_args.append(copy.deepcopy(wag))
        else:
            task_args.append(copy.deepcopy(wag))
    return task_args

def refine_mesh(args, ws):
    target_v = ws.config.getint('GeometriK', 'FineMeshV')
    task_args = _get_task_args(ws, per_geometry=True)
    for wag in task_args:
        if os.path.isfile(wag.refined_geo_fn):
            continue
        # util.shell(['/usr/bin/env'])
        util.shell(['./TetWild', '--level', '6', '--targeted-num-v', str(target_v), '--output-surface', wag.refined_geo_fn, wag.geo_fn])

def _sample_key_point_worker(wag):
    ws = util.Workspace(wag.dir)
    ws.current_trial = wag.current_trial
    kpp = pygeokey.KeyPointProber(wag.geo_fn)
    natt = ws.config.getint('GeometriK', 'KeyPointAttempts')
    util.log("[sample_key_point] probing {} for {} attempts".format(wag.geo_fn, natt))
    pts = kpp.probe_key_points(natt)
    kps_fn = ws.keypoint_prediction_file(wag.puzzle_name, wag.geo_type)
    util.log("[sample_key_point] writing {} points to {}".format(pts.shape[0], kps_fn))
    SAMPLE_NOTCH = ws.config.getboolean('GeometriK', 'EnableNotchDetection', fallback=False)
    if SAMPLE_NOTCH:
        util.log("[sample_key_point] Probing notches for {}".format(wag.refined_geo_fn))
        kpp2 = pygeokey.KeyPointProber(wag.refined_geo_fn)
        npts = kpp2.probe_notch_points()
        util.log("[sample_key_point] writing {} points and {} notches to {}".format(pts.shape[0], npts.shape[0], kps_fn))
        np.savez(kps_fn, KEY_POINT_AMBIENT=pts, NOTCH_POINT_AMBIENT=npts)
    else:
        np.savez(kps_fn, KEY_POINT_AMBIENT=pts)

def sample_key_point(args, ws):
    task_args = _get_task_args(ws, per_geometry=True)
    USE_MP = False
    if USE_MP:
        pcpu = multiprocessing.Pool()
        pcpu.map(_sample_key_point_worker, task_args)
    else:
        for wag in task_args:
            _sample_key_point_worker(wag)

def _sample_key_conf_worker(wag):
    ws = util.Workspace(wag.dir)
    ws.current_trial = wag.current_trial
    kfn = ws.keyconf_prediction_file(wag.puzzle_name, for_read=False)
    util.log('[sample_key_conf] trial {}'.format(ws.current_trial))
    util.log('[sample_key_conf] sampling to {}'.format(kfn))
    ks = pygeokey.KeySampler(wag.env_fn, wag.rob_fn)
    INTER_CLASS_PREDICTION = True # For debugging. This must be enabled in order to predict notch-tooth key confs
    if INTER_CLASS_PREDICTION:
        def _load_kps(geo_type):
            kps_fn = ws.keypoint_prediction_file(wag.puzzle_name, geo_type)
            d = matio.load(kps_fn)
            return util.access_keypoints(d)
        env_kps = _load_kps('env')
        rob_kps = _load_kps('rob')
        kqs, _, _ = ks.get_all_key_configs(env_kps, rob_kps,
                                           ws.config.getint('GeometriK', 'KeyConfigRotations'))
    else:
        def _load_kps(geo_type):
            kps_fn = ws.keypoint_prediction_file(wag.puzzle_name, geo_type)
            return matio.load(kps_fn)
        env_d = _load_kps('env')
        rob_d = _load_kps('rob')
        def _gen_kqs(key):
            if not key in env_d or env_d[key].shape[0] == 0:
                return None
            if not key in rob_d or rob_d[key].shape[0] == 0:
                return None
            ip1 = env_d[key]
            ip2 = rob_d[key]
            kqs, _, _ = ks.get_all_key_configs(ip1, ip2,
                                               ws.config.getint('GeometriK', 'KeyConfigRotations'))
            return kqs
        kqs1 = _gen_kqs('KEY_POINT_AMBIENT')
        kqs2 = _gen_kqs('NOTCH_POINT_AMBIENT')
        kqs = util.safe_concatente([kqs1, kqs2])
    uw = util.create_unit_world(wag.puzzle_fn)
    cfg, config = parse_ompl.parse_simple(wag.puzzle_fn)
    iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
    gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
    ompl_q = uw.translate_vanilla_to_ompl(kqs)
    ompl_q = np.concatenate((iq, gq, ompl_q), axis=0)
    #np.savez(kfn, KEYQ_AMBIENT_NOIG=kqs, KEYQ_OMPL=ompl_q)
    np.savez(kfn, KEYQ_OMPL=ompl_q)
    util.log('[sample_key_conf] save {} key confs to {}'.format(ompl_q.shape, kfn))

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
        'refine_mesh' : refine_mesh,
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

def _remote_command(ws, cmd, auto_retry=True):
    ws.remote_command(ws.condor_host,
                      ws.condor_exec(),
                      ws.condor_ws(),
                      'geometrik', cmd, with_trial=True, auto_retry=auto_retry)

def remote_refine_mesh(ws):
    _remote_command(ws, 'refine_mesh')

def remote_sample_key_point(ws):
    _remote_command(ws, 'sample_key_point')

def remote_sample_key_conf(ws):
    _remote_command(ws, 'sample_key_conf')

'''
# Deprecated
def autorun(args):
    ws = util.Workspace(args.dir)
    ws.current_trial = args.current_trial
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)
'''

def collect_stages():
    ret = [ ('deploy_to_condor',
              lambda ws: ws.deploy_to_condor(util.WORKSPACE_SIGNATURE_FILE,
                                             util.WORKSPACE_CONFIG_FILE,
                                             util.CONDOR_TEMPLATE,
                                             util.TESTING_DIR+'/')
            ),
            ('refine_mesh', remote_refine_mesh),
            ('sample_key_point', remote_sample_key_point),
            ('sample_key_conf', remote_sample_key_conf),
            #('deploy_geometrik_to_condor', lambda ws: deploy_geometrik_to_condor(None, ws))
          ]
    return ret
