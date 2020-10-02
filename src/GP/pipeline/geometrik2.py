#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
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

def _DETECTOR_TEMPLATE(args, ws, pairing, FMT, worker_func):
    task_args = get_task_args(ws, args=args, per_geometry=True, FMT=FMT, pairing=pairing)
    for wag in task_args:
        pts = worker_func(ws, wag)

def predict_geratio_key_worker(ws, wag_pair):
    wag1, wag2 = wag_pair
    ws.current_trial = wag1.current_trial
    assert wag1.current_trial == wag2.current_trial
    assert wag1.puzzle_fn == wag2.puzzle_fn
    assert wag1.geo_type == 'env'
    assert wag2.geo_type == 'rob'

    ge_fn1 = FMT_to_file(ws, wag1, util.GERATIO_POINT_FMT)
    ge_fn2 = FMT_to_file(ws, wag2, util.GERATIO_POINT_FMT)
    ge1 = matio.load(ge_fn1)['KEY_POINT_AMBIENT']
    ge2 = matio.load(ge_fn2)['KEY_POINT_AMBIENT']
    util.log(f"{wag1.geo_type}_gepts {ge1.shape}")
    util.log(f"{wag2.geo_type}_gepts {ge2.shape}")

    ks = pygeokey.KeySampler(wag1.geo_fn, wag2.geo_fn)
    nrot = ws.config.getint('GeometriK', 'KeyConfigRotations')
    kqs, keyid1, keyid2 = ks.get_all_key_configs(ge1, ge2, nrot)

    """
    Post processing
    1. Translate to OMPL configuration
    2. Prefix initial state and goal state
    """
    if kqs.shape[0] != 0:
        uw = util.create_unit_world(wag1.puzzle_fn)
        ompl_q = uw.translate_vanilla_to_ompl(kqs)
        ompl_q = prefix_iq_and_gq(wag1, ompl_q)
    else:
        ompl_q = kqs

    kfn = FMT_to_file(ws, wag1, util.GERATIO_KEY_FMT)
    np.savez(kfn, KEYQ_OMPL=ompl_q, ENV_KEYID=keyid1, ROB_KEYID=keyid2)
    return None

def predict_notch_key_worker(ws, wag_pair):
    wag1, wag2 = wag_pair
    ws.current_trial = wag1.current_trial
    assert wag1.current_trial == wag2.current_trial
    assert wag1.puzzle_fn == wag2.puzzle_fn
    assert wag1.geo_type == 'env'
    assert wag2.geo_type == 'rob'

    SAMPLE_NOTCH = ws.config.getboolean('GeometriK', 'EnableNotchDetection', fallback=True)
    if SAMPLE_NOTCH:
        ge_fn1 = FMT_to_file(ws, wag1, util.GERATIO_POINT_FMT)
        ge_fn2 = FMT_to_file(ws, wag2, util.GERATIO_POINT_FMT)
        ge1 = matio.load(ge_fn1)['KEY_POINT_AMBIENT']
        ge2 = matio.load(ge_fn2)['KEY_POINT_AMBIENT']
        util.log(f"{wag1.geo_type}_gepts {ge1.shape}")
        util.log(f"{wag2.geo_type}_gepts {ge2.shape}")
        nt_fn1 = FMT_to_file(ws, wag1, util.NOTCH_POINT_FMT)
        nt_fn2 = FMT_to_file(ws, wag2, util.NOTCH_POINT_FMT)
        nt1 = matio.load(nt_fn1)['NOTCH_POINT_AMBIENT']
        nt2 = matio.load(nt_fn2)['NOTCH_POINT_AMBIENT']
        util.log(f"{wag1.geo_type}_ntpts {nt1.shape}")
        util.log(f"{wag2.geo_type}_ntpts {nt2.shape}")

        ks = pygeokey.KeySampler(wag1.geo_fn, wag2.geo_fn)
        nrot = ws.config.getint('GeometriK', 'KeyConfigRotations')
        kqs1, keyid_ge1, keyid_nt2 = ks.get_all_key_configs(ge1, nt2, nrot)
        kqs2, keyid_nt1, keyid_ge2 = ks.get_all_key_configs(nt1, ge2, nrot)
        kqs = util.safe_concatente([kqs1, kqs2], axis=0)
    else:
        util.log("[predict_notch_key_worker] Notch detection is disabled in this workspace")
        keyid_ge1 = keyid_nt2 = keyid_nt1 = keyid_ge2 = kqs = np.array([])
    if kqs.shape[0] > 0:
        uw = util.create_unit_world(wag1.puzzle_fn)
        unit_q = uw.translate_vanilla_to_unit(kqs)
        ompl_q = uw.translate_unit_to_ompl(unit_q)
        ompl_q = prefix_iq_and_gq(wag1, ompl_q)
    else:
        ompl_q = kqs # empty
    kfn = FMT_to_file(ws, wag1, util.NOTCH_KEY_FMT)

    # Remind ge1 and nt1 come from env, and *2 come from rob
    assert wag1.geo_type == 'env'
    assert wag2.geo_type == 'rob'
    np.savez(kfn, KEYQ_OMPL=ompl_q,
             ENV_GEKEYID=keyid_ge1,
             ROB_GEKEYID=keyid_ge2,
             ENV_NTKEYID=keyid_nt1,
             ROB_NTKEYID=keyid_nt2)
    util.ack(f'[predict_geratio_key] save {ompl_q.shape} keys to {kfn}')
    return None

def _detect_geratio_feature_worker(ws, wag):
    ws.current_trial = wag.current_trial
    pts = detect_geratio_feature_worker(ws, wag)
    ge_fn = FMT_to_file(ws, wag, util.GERATIO_POINT_FMT)
    np.savez(ge_fn, KEY_POINT_AMBIENT=pts)
    util.ack(f'[detect_geratio_feature][{wag.puzzle_name}][{wag.geo_type}] saving {pts.shape} to {ge_fn}')

def _detect_notch_feature_worker(ws, wag):
    ws.current_trial = wag.current_trial
    pts = detect_notch_feature_worker(ws, wag)
    nt_fn = FMT_to_file(ws, wag, util.NOTCH_POINT_FMT)
    np.savez(nt_fn, NOTCH_POINT_AMBIENT=pts)
    util.ack(f'[detect_notch_feature][{wag.puzzle_name}][{wag.geo_type}] saving {pts.shape} to {nt_fn}')

def detect_geratio_feature(args, ws):
    _DETECTOR_TEMPLATE(args, ws,
                       pairing=False,
                       FMT=util.GERATIO_POINT_FMT,
                       worker_func=_detect_geratio_feature_worker)

def detect_notch_feature(args, ws):
    _DETECTOR_TEMPLATE(args, ws,
                       pairing=False,
                       FMT=util.NOTCH_POINT_FMT,
                       worker_func=_detect_notch_feature_worker)

def predict_geratio_key_conf(args, ws):
    _DETECTOR_TEMPLATE(args, ws,
                       pairing=True,
                       FMT=util.GERATIO_KEY_FMT,
                       worker_func=predict_geratio_key_worker)

def predict_notch_key_conf(args, ws):
    _DETECTOR_TEMPLATE(args, ws,
                       pairing=True,
                       FMT=util.NOTCH_KEY_FMT,
                       worker_func=predict_notch_key_worker)

function_dict = {
        'refine_mesh' : refine_mesh,
        'detect_geratio_feature': detect_geratio_feature,
        'detect_notch_feature': detect_notch_feature,
        'predict_geratio_key_conf': predict_geratio_key_conf,
        'predict_notch_key_conf': predict_notch_key_conf,
}

def setup_parser(subparsers):
    gk1_setup_parser(subparsers, module_name='geometrik2', function_dict=function_dict)

def run(args):
    if args.stage in function_dict:
        ws = util.create_workspace_from_args(args)
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
    if variant in [6,7]:
        ret = [
                ('refine_mesh', remote_refine_mesh),
                ('detect_geratio_feature', remote_detect_geratio_feature),
                ('detect_notch_feature', remote_detect_notch_feature),
                ('predict_geratio_key_conf', remote_predict_geratio_key_conf),
                ('predict_notch_key_conf', remote_predict_notch_key_conf),
              ]
    elif variant in [8]:
        ret = [
                ('detect_geratio_feature', remote_detect_geratio_feature),
                ('predict_geratio_key_conf', remote_predict_geratio_key_conf),
              ]
    elif variant in [9]:
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
