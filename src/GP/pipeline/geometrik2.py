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
from .geometrik import (get_task_args,
        refine_mesh,
        detect_geratio_feature_worker,
        detect_notch_feature_worker,
        remote_refine_mesh,
        setup_parser as gk1_setup_parser,
)
try:
    import pygeokey
except ImportError as e:
    util.warn("[WARNING] CANNOT IMPORT pygeokey. This node is incapable of geometric based prediction")

def FMT_to_file(ws, wag, FMT):
    return ws.local_ws(util.TESTING_DIR, wag.puzzle_name,
                       FMT.format(geo_type=wag.geo_type, trial=wag.current_trial))

def prefix_iq_and_gq(wag, ompl_q):
    cfg, config = parse_ompl.parse_simple(wag.puzzle_fn)
    iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
    gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
    return np.concatenate((iq, gq, ompl_q), axis=0)

def _DETECTOR_TEMPLATE(args, ws, per_geometry, FMT, worker_func, KEY):
    task_args = get_task_args(ws, args=args, per_geometry=True, FMT=FMT, pairing=True)
    ws = util.Workspace(wag.dir)
    for wag in task_args:
        ws.current_trial = wag.current_trial
        pts = worker_func(ws, wag)
        fn = FMT_to_file(ws, wag, FMT)
        dic = {KEY: pts}
        npz.savez(fn, **dic)

def predict_geratio_key_worker(ws, wag_pair):
    wag1, wag2 = wag_pair
    assert wag1.geo_type == 'env'
    assert wag2.geo_type == 'rob'
    ge_fn1 = FMT_to_file(ws, wag1, util.GERATIO_POINT_FMT)
    ge_fn2 = FMT_to_file(ws, wag2, util.GERATIO_POINT_FMT)
    ge1 = matio.load(ge_fn1)['KEY_POINT_AMBIENT']
    ge2 = matio.load(ge_fn2)['KEY_POINT_AMBIENT']
    util.log(f"{wag1.geo_type}_gepts {ge1.shape}")
    util.log(f"{wag2.geo_type}_gepts {ge2.shape}")
    nrot = ws.config.getint('GeometriK', 'KeyConfigRotations')
    kqs, keyid1, keyid2 = ks.get_all_key_configs(ge1, ge2, nrot)
    unit_q = uw.translate_vanilla_to_unit(kqs)
    ompl_q = uw.translate_unit_to_ompl(unit_q)
    ompl_q = prefix_iq_and_gq(wag, ompl_q)
    kfn = FMT_to_file(ws, wag1, util.GERATIO_KEY_FMT)
    np.savez(kfn, KEYQ_OMPL=ompl_q, ENV_KEYID=keyid1, ROB_KEYID=keyid2)

def predict_notch_key_worker(ws, wag_pair):
    wag1, wag2 = wag_pair
    assert wag1.geo_type == 'env'
    assert wag2.geo_type == 'rob'
    ge_fn1 = FMT_to_file(ws, wag1, util.GERATIO_POINT_FMT)
    ge_fn2 = FMT_to_file(ws, wag2, util.GERATIO_POINT_FMT)
    ge1 = matio.load(ge_fn1)['KEY_POINT_AMBIENT']
    ge2 = matio.load(ge_fn2)['KEY_POINT_AMBIENT']
    util.log(f"{wag1.geo_type}_gepts {ge1.shape}")
    util.log(f"{wag2.geo_type}_gepts {ge2.shape}")
    nt_fn1 = FMT_to_file(ws, wag1, util.NOTCH_POINT_FMT)
    nt_fn2 = FMT_to_file(ws, wag2, util.NOTCH_POINT_FMT)
    nt1 = matio.load(ge_fn1)['NOTCH_POINT_AMBIENT']
    nt2 = matio.load(ge_fn2)['NOTCH_POINT_AMBIENT']
    util.log(f"{wag1.geo_type}_ntpts {nt1.shape}")
    util.log(f"{wag2.geo_type}_ntpts {nt2.shape}")
    nrot = ws.config.getint('GeometriK', 'KeyConfigRotations')
    kqs1, keyid_ge1, keyid_nt2 = ks.get_all_key_configs(ge1, nt2, nrot)
    kqs2, keyid_ge2, keyid_nt1 = ks.get_all_key_configs(ge2, nt1, nrot)
    kqs = util.safe_concatente([kqs1, kqs2], axis=0)
    unit_q = uw.translate_vanilla_to_unit(kqs)
    ompl_q = uw.translate_unit_to_ompl(unit_q)
    ompl_q = prefix_iq_and_gq(wag, ompl_q)
    kfn = FMT_to_file(ws, wag1, util.NOTCH_KEY_FMT)
    # Remind ge1 and nt1 come from env, and *2 come from rob
    assert wag1.geo_type == 'env'
    assert wag2.geo_type == 'rob'
    np.savez(kfn, KEYQ_OMPL=ompl_q,
             ENV_GEKEYID=keyid_ge1,
             ROB_GEKEYID=keyid_ge2,
             ENV_NTKEYID=keyid_nt1,
             ROB_NTKEYID=keyid_nt2)

def detect_geratio_feature(args, ws):
    _DETECTOR_TEMPLATE(args, ws,
                       per_geometry=True,
                       FMT=util.GERATIO_POINT_FMT,
                       worker_func=detect_geratio_feature_worker,
                       KEY='KEY_POINT_AMBIENT')

def detect_notch_feature(args, ws):
    _DETECTOR_TEMPLATE(args, ws,
                       per_geometry=True,
                       FMT=util.NOTCH_POINT_FMT,
                       worker_func=detect_notch_feature_worker,
                       KEY='NOTCH_POINT_AMBIENT')

def predict_geratio_key_conf(args, ws):
    _DETECTOR_TEMPLATE(args, ws,
                       per_geometry=True,
                       FMT=util.NOTCH_POINT_FMT,
                       worker_func=predict_geratio_key_worker,
                       KEY='NOTCH_POINT_AMBIENT')

def predict_notch_key_conf(args, ws):
    _DETECTOR_TEMPLATE(args, ws,
                       per_geometry=True,
                       FMT=util.NOTCH_POINT_FMT,
                       worker_func=predict_notch_key_worker,
                       KEY='NOTCH_POINT_AMBIENT')

function_dict = {
        'refine_mesh' : refine_mesh,
        'detect_geratio_feature': detect_geratio_feature,
        'detect_notch_feature': detect_notch_feature,
        'predict_geratio_key_conf': predict_geratio_key_conf,
        'predict_notch_key_conf': predict_notch_key_conf,
}

def setup_parser(subparsers):
    gk1_setup_parser(subparsers, module_name='geometrik2')

def run(args):
    if args.stage in function_dict:
        ws = util.Workspace(args.dir)
        ws.current_trial = args.current_trial
        function_dict[args.stage](args, ws)
    else:
        print("Unknown geometrik2 pipeline stage {}".format(args.stage))

def _remote_command(ws, cmd, auto_retry=True):
    ws.remote_command(ws.condor_host,
                      ws.condor_exec(),
                      ws.condor_ws(),
                      'geometrik2', cmd, with_trial=True, auto_retry=auto_retry)

def remote_detect_geratio_feature(ws):
    _remote_command(ws, 'detect_geratio_feature')

def remote_detect_notch_feature(ws):
    _remote_command(ws, 'detect_notch_feature')

def remote_predict_geratio_key_conf(ws):
    _remote_command(ws, 'predict_geratio_key_conf')

def remote_predict_notch_key_conf(ws):
    _remote_command(ws, 'predict_notch_key_conf')

def collect_stages(variant=0):
    if variant in [6]:
        ret = [
                ('refine_mesh', remote_refine_mesh),
                ('detect_geratio_feature', remote_detect_geratio_feature),
                ('detect_notch_feature', remote_detect_notch_feature),
                ('predict_geratio_key_conf', remote_predict_geratio_key_conf),
                ('predict_notch_key_conf', remote_predict_notch_key_conf),
              ]
    else:
        assert False, '[geometrik2] Unknown variant {}'.format(variant)
    return ret
