#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing
import copy
from progressbar import ProgressBar

from . import choice_formatter
from . import util
from . import parse_ompl
from . import matio
from . import keyconf
from . import atlas
try:
    import pygeokey
except ImportError as e:
    util.warn("[WARNING] CANNOT IMPORT pygeokey. This node is incapable of geometric based prediction")

class WorkerArgs(object):
    pass

def _get_task_args(ws):
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
        wag.geo_type = 'rob'
        wag.geo_fn = cfg.rob_fn
        task_args.append(copy.deepcopy(wag))
    return task_args

def _sample_key_point_worker(wag):
    ws = util.Workspace(wag.dir)
    ws.current_trial = wag.current_trial
    kpp = pygeokey.KeyPointProber(wag.geo_fn)
    natt = ws.config.getint('RoboGeoK', 'KeyPointAttempts')
    pt_pairs = kpp.probe_key_points(natt)
    util.log("[sample_rob_key_point] {}: {} key point sampled".format(wag.geo_fn, pt_pairs.shape))
    util.log("[sample_rob_key_point] probing {} for {} attempts".format(wag.geo_fn, natt))
    '''
    pts = np.zeros((pt_pairs.shape[0] * 2, 7), dtype=np.float64)
    for pti, ptp in enumerate(pt_pairs):
        pts[2 * pti + 0, 0:3] = ptp[0:3]
        pts[2 * pti + 1, 0:3] = ptp[3:6]
    '''
    uw = util.create_unit_world(wag.puzzle_fn)
    unit_imp_1 = uw.translate_vanilla_pts_to_unit(uw.GEO_ROB, pt_pairs[:, 0:3])
    unit_imp_2 = uw.translate_vanilla_pts_to_unit(uw.GEO_ROB, pt_pairs[:, 3:6])
    kps_fn = ws.keypoint_prediction_file(wag.puzzle_name, wag.geo_type)
    util.log("[sample_rob_key_point] writing {} points to {}".format(pt_pairs.shape[0], kps_fn))
    matio.savetxt('visimp.tmp.txt', np.concatenate((unit_imp_1, unit_imp_2), axis=0))
    np.savez(kps_fn, UNIT_IMP_1=unit_imp_1, UNIT_IMP_2=unit_imp_2)

def sample_rob_key_point(args, ws):
    task_args = _get_task_args(ws)
    #for wag in task_args:
    #    _sample_key_point_worker(wag)
    pcpu = multiprocessing.Pool()
    pcpu.map(_sample_key_point_worker, task_args)

# It's so nasty that numpy does not have a normalized function for us
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def enum_key_conf(args, ws):
    task_args = _get_task_args(ws)
    margin = ws.config.getfloat('Prediction', 'Margin')
    nrot = ws.config.getint('RoboGeoK', 'KeyConfigRotations')
    for wag in task_args:
        d = np.load(ws.keypoint_prediction_file(wag.puzzle_name, wag.geo_type))
        rob_kp_1 = d['UNIT_IMP_1']
        rob_kp_2 = d['UNIT_IMP_2']
        nkp = rob_kp_1.shape[0]
        nsurface = ws.config.getint('RoboGeoK', 'EnvKeyPoints')
        # nsurface = 32
        uw = util.create_unit_world(wag.puzzle_fn)
        env_sampler = atlas.AtlasSampler(ws.local_ws(util.TESTING_DIR,
                                                     wag.puzzle_name,
                                                     'env-a2p.npz'),
                                         None,
                                         'env', uw.GEO_ENV)
        key_conf = []
        batch_size = 2*nkp*nsurface
        with ProgressBar(max_value=batch_size) as bar:
            for kp1, kp2 in zip(rob_kp_1, rob_kp_2):
                tup1 = (kp1, normalize(kp2 - kp1))
                for i in range(nsurface):
                    tup2 = env_sampler.sample(uw)
                    qs_raw = uw.enum_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1],
                                                        margin,
                                                        denominator=nrot,
                                                        only_median=True)
                    qs = [q for q in qs_raw if not uw.is_disentangled(q)] # Trim disentangled state
                    key_conf += qs
                    bar += 1
                tup1 = (kp2, normalize(kp1 - kp2))
                for i in range(nsurface):
                    tup2 = env_sampler.sample(uw)
                    qs_raw = uw.enum_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1],
                                                        margin,
                                                        denominator=nrot,
                                                        only_median=True)
                    qs = [q for q in qs_raw if not uw.is_disentangled(q)] # Trim disentangled state
                    key_conf += qs
                    bar += 1
        keyconf.export_keyconf(ws, uw, wag.puzzle_fn, wag.puzzle_name, key_conf)

def deploy_geometrik_to_condor(args, ws):
    ws.deploy_to_condor(util.WORKSPACE_SIGNATURE_FILE,
                        util.WORKSPACE_CONFIG_FILE,
                        util.CONDOR_TEMPLATE,
                        util.TESTING_DIR+'/')

function_dict = {
        'generate_atlas2prim' : keyconf.generate_atlas2prim,
        'sample_rob_key_point' : sample_rob_key_point,
        'enum_key_conf' : enum_key_conf,
        'deploy_geometrik_to_condor' : deploy_geometrik_to_condor,
}

def setup_parser(subparsers):
    p = subparsers.add_parser('robogeok', help='Sample Key configuration from Geometric features',
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
            ('generate_atlas2prim', lambda ws: keyconf.generate_atlas2prim(None, ws)),
            ('sample_rob_key_point', lambda ws: sample_rob_key_point(None, ws)),
            ('enum_key_conf', lambda ws: enum_key_conf(None, ws)),
            ('deploy_robogeok_to_condor', lambda ws: deploy_robogeok_to_condor(None, ws))
          ]
    return ret
