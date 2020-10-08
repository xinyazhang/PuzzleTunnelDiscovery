#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile, dirname
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
from . import partt
from . import touchq_util
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

def _predict_atlas2prim(tup):
    import pyosr
    import hashlib
    ws, puzzle_fn, puzzle_name = tup
    r = None
    puzzle, config = parse_ompl.parse_simple(puzzle_fn)
    for geo_type,flags,model_fn in zip(['rob', 'env'], [pyosr.Renderer.NO_SCENE_RENDERING, pyosr.Renderer.NO_ROBOT_RENDERING], [puzzle.rob_fn, puzzle.env_fn]):
        tgt_file = ws.local_ws(util.TESTING_DIR, puzzle_name, geo_type+'-a2p.npz')
        new_sha = None
        try:
            p = pathlib.Path(model_fn)
            new_sha = hashlib.blake2b(p.read_bytes()).digest()
            old_sha = bytes(matio.load(tgt_file)['MODEL_BLAKE2B'])
            if new_sha == old_sha:
                util.ack('[generate_atlas2prim] {} is updated (model: {})'.format(tgt_file, model_fn))
                continue
        except:
            pass
        if r is None:
            r = util.create_offscreen_renderer(puzzle_fn, ws.chart_resolution)
            r.avi = False
        r.render_mvrgbd(pyosr.Renderer.UV_MAPPINNG_RENDERING|flags)
        atlas2prim = np.copy(r.mvpid.reshape((r.pbufferWidth, r.pbufferHeight)))
        #imsave(geo_type+'-a2p-nt.png', atlas2prim) # This is for debugging
        atlas2prim = texture_format.framebuffer_to_file(atlas2prim)
        atlas2uv = np.copy(r.mvuv.reshape((r.pbufferWidth, r.pbufferHeight, 2)))
        atlas2uv = texture_format.framebuffer_to_file(atlas2uv)
        if new_sha is None:
            np.savez(tgt_file,
                     PRIM=atlas2prim,
                     UV=atlas2uv)
        else:
            np.savez(tgt_file,
                     PRIM=atlas2prim,
                     UV=atlas2uv,
                     MODEL_BLAKE2B=new_sha)
        imsave(ws.local_ws(util.TESTING_DIR, puzzle_name, geo_type+'-a2p.png'), atlas2prim) # This is for debugging

def generate_atlas2prim(args, ws):
    task_tup = []
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        task_tup.append((ws, puzzle_fn, puzzle_name))
    USE_MP = False
    if USE_MP:
        pgpu = multiprocessing.Pool(1)
        pgpu.map(_predict_atlas2prim, task_tup)
    else:
        for tup in task_tup:
            _predict_atlas2prim(tup)

def export_keyconf(ws, uw, puzzle_fn, puzzle_name, key_conf, FMT=util.UNSCREENED_KEY_PREDICTION_FMT):
    cfg, config = parse_ompl.parse_simple(puzzle_fn)
    iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
    gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
    util.log("[predict_keyconf(worker)] sampled {} keyconf from puzzle {}".format(len(key_conf), puzzle_name))
    qs_ompl = uw.translate_unit_to_ompl(key_conf)
    ompl_q = np.concatenate((iq, gq, qs_ompl), axis=0)
    key_fn = ws.keyconf_file_from_fmt(puzzle_name, FMT=FMT)
    util.log("[predict_keyconf(worker)] save to key file {}".format(key_fn))
    unit_q = uw.translate_ompl_to_unit(ompl_q)
    # np.savez(key_fn, KEYQ_OMPL=ompl_q, KEYQ_UNIT=unit_q)
    np.savez(key_fn, KEYQ_OMPL=ompl_q)
    matio.savetxt(key_fn + 'unit.txt', unit_q)
    return key_fn

def single_as_factory(ws, uw, puzzle_fn, puzzle_name, geo_type):
    GEO_TYPE_TO_ID = { 'rob' : uw.GEO_ROB, 'env': uw.GEO_ENV }
    return atlas.AtlasSampler(ws.local_ws(util.TESTING_DIR, puzzle_name, f'{geo_type}-a2p.npz'),
                              ws.atex_prediction_file(puzzle_fn, geo_type),
                              geo_type, GEO_TYPE_TO_ID[geo_type])


def _predict_worker(tup):
    DEBUG = True
    ws_dir, puzzle_fn, puzzle_name, trial, FMT, samples_per_puzzle, as_factory = tup
    ws = util.Workspace(ws_dir)
    ws.current_trial = trial
    uw = util.create_unit_world(puzzle_fn)
    rob_sampler = as_factory(ws, uw, puzzle_fn, puzzle_name, 'rob')
    env_sampler = as_factory(ws, uw, puzzle_fn, puzzle_name, 'env')

    if DEBUG:
        rob_sampler.enable_debugging()
        env_sampler.enable_debugging()
    key_conf = []
    nrot = ws.config.getint('Prediction', 'NumberOfRotations')
    margin = ws.config.getfloat('Prediction', 'Margin')
    #for i in progressbar(range(batch_size)):
    if True:
        with ProgressBar(max_value=samples_per_puzzle) as bar:
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
                    bar.update(min(samples_per_puzzle, len(key_conf)))
                if len(key_conf) > samples_per_puzzle:
                    break
    else:
        with ProgressBar(max_value=samples_per_puzzle) as bar:
            tups1 = rob_sampler.get_top_k_surface_tups(uw, 64)
            tups2 = env_sampler.get_top_k_surface_tups(uw, 64)
            for tup1 in tups1:
                for tup2 in tups2:
                    # print("tup1 {}".format(tup1))
                    # print("tup2 {}".format(tup2))
                    sys.stdout.flush()
                    qs_raw = uw.enum_free_configuration(tup1[0], tup1[1], tup2[0], tup2[1],
                                                        margin,
                                                        denominator=nrot,
                                                        only_median=True)
                    qs = [q for q in qs_raw if not uw.is_disentangled(q)] # Trim disentangled state
                    for q in qs:
                        key_conf.append(q)
                        bar.update(min(samples_per_puzzle, len(key_conf)))
                    if len(key_conf) > samples_per_puzzle:
                        break
                if len(key_conf) > samples_per_puzzle:
                    break
    key_fn = export_keyconf(ws, uw, puzzle_fn, puzzle_name, key_conf, FMT)
    if DEBUG:
        rob_sampler.dump_debugging(prefix=dirname(str(key_fn))+'/')
        env_sampler.dump_debugging(prefix=dirname(str(key_fn))+'/')

def predict_keyconf(args, ws, worker=_predict_worker):
    task_tup = []
    samples_per_puzzle = ws.config.getint('Prediction', 'SurfacePairsToSample')
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        task_tup.append((ws.dir, puzzle_fn, puzzle_name,
                         ws.current_trial,
                         util.UNSCREENED_KEY_PREDICTION_FMT,
                         samples_per_puzzle))
        util.log('[predict_keyconf] found puzzle {} at {}'.format(puzzle_name, puzzle_fn))
    if 'auto' == ws.config.get('Prediction', 'NumberOfPredictionProcesses'):
        ncpu = None
    else:
        ncpu = ws.config.getint('Prediction', 'NumberOfPredictionProcesses')
    # pcpu = multiprocessing.Pool(ncpu)
    # pcpu.map(_predict_worker, task_tup)
    for tup in task_tup:
        worker(tup)

def oversample_keyconf(args, ws, worker=_predict_worker):
    task_tup = []
    samples_per_puzzle = ws.config.getint('Prediction', 'SurfacePairsToSample') * ws.config.getint('Prediction', 'OversamplingRatio')
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        task_tup.append((ws.dir, puzzle_fn, puzzle_name,
                         ws.current_trial,
                         util.OVERSAMPLED_KEY_PREDICTION_FMT,
                         samples_per_puzzle,
                         single_as_factory))
        util.log('[predict_keyconf] found puzzle {} at {}'.format(puzzle_name, puzzle_fn))
    for tup in task_tup:
        worker(tup)


def multinet_oversample_keyconf(args, ws, worker=_predict_worker):
    task_tup = []
    samples_per_puzzle = ws.config.getint('Prediction', 'SurfacePairsToSample') * ws.config.getint('Prediction', 'OversamplingRatio')
    rews_dir = ws.config.get('Prediction', 'ReuseWorkspace', fallback='')
    assert rews_dir, 'Prediction.ReuseWorkspace is required for multinet_oversample_keyconf'

    rews_dir = join(ws.dir, rews_dir) # Relative path
    rews = util.Workspace(rews_dir)
    rews.nn_profile = ws.nn_profile

    def composite_as_factory(ws, uw, puzzle_fn, puzzle_name, geo_type):
        GEO_TYPE_TO_ID = { 'rob' : uw.GEO_ROB, 'env': uw.GEO_ENV }
        a2p_fn = ws.local_ws(util.TESTING_DIR, puzzle_name, f'{geo_type}-a2p.npz')
        pred_list = [ws.atex_prediction_file(puzzle_fn, geo_type, netid=netid) for netid in rews.training_groups]
        return atlas.CompositeAtlasSampler(a2p_fn, pred_list, geo_type, GEO_TYPE_TO_ID[geo_type])

    for puzzle_fn, puzzle_name in ws.test_puzzle_generator():
        task_tup.append((ws.dir, puzzle_fn, puzzle_name,
                         ws.current_trial,
                         util.OVERSAMPLED_KEY_PREDICTION_FMT,
                         samples_per_puzzle,
                         composite_as_factory))
        util.log('[predict_keyconf] found puzzle {} at {}'.format(puzzle_name, puzzle_fn))
    for tup in task_tup:
        worker(tup)

def _predict_2d_worker(tup):
    ws_dir, puzzle_fn, puzzle_name, trial, FMT, batch_size = tup
    ws = util.Workspace(ws_dir)
    ws.current_trial = trial
    uw = util.create_unit_world(puzzle_fn)
    rob_sampler = atlas.AtlasSampler(ws.local_ws(util.TESTING_DIR, puzzle_name, 'rob-a2p.npz'),
                                     ws.atex_prediction_file(puzzle_fn, 'rob'),
                                     'rob', uw.GEO_ROB)
    env_sampler = atlas.AtlasSampler(ws.local_ws(util.TESTING_DIR, puzzle_name, 'env-a2p.npz'),
                                     ws.atex_prediction_file(puzzle_fn, 'env'),
                                     'env', uw.GEO_ENV)
    rob_sampler.enable_debugging()
    env_sampler.enable_debugging()
    key_conf = []
    # nrot = ws.config.getint('Prediction', 'NumberOfRotations')
    nrot = 6
    margin = ws.config.getfloat('Prediction', 'Margin')
    batch_size = 128
    with ProgressBar(max_value=batch_size) as bar:
        while True:
            for i in range(12):
                tup1 = rob_sampler.sample(uw)
                tup2 = env_sampler.sample(uw)
                qs_raw = uw.enum_2drot_free_configuration(tup1[0], tup1[1], tup[3],
                                                          tup2[0], tup2[1], tup[3],
                                                          margin,
                                                          altitude_divider=nrot,
                                                          azimuth_divider=nrot,
                                                          return_all=True)
                # qs = [q for q in qs_raw if not uw.is_disentangled(q)] # Trim disentangled state
                qs = qs_raw
                for q in qs_raw:
                    key_conf.append(q)
                    bar.update(min(batch_size, len(key_conf)))
                #if len(key_conf) > batch_size:
            if len(key_conf) > 0:
                break
    cfg, config = parse_ompl.parse_simple(puzzle_fn)
    iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
    gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
    util.log("[predict_keyconf_2d (worker)] sampled {} keyconf from puzzle {}".format(len(key_conf), puzzle_name))
    qs_ompl = uw.translate_unit_to_ompl(key_conf)
    ompl_q = np.concatenate((iq, gq, qs_ompl), axis=0)
    key_fn = ws.keyconf_prediction_file(puzzle_name)
    util.log("[predict_keyconf_2d (worker)] save to key file {}".format(key_fn))
    unit_q = uw.translate_ompl_to_unit(ompl_q)
    # np.savez(key_fn, KEYQ_OMPL=ompl_q, KEYQ_UNIT=unit_q)
    # np.savez(key_fn, KEYQ_OMPL=ompl_q)
    matio.savetxt(key_fn + '.unit.txt', unit_q)
    rob_sampler.dump_debugging(prefix=dirname(str(key_fn))+'/')
    env_sampler.dump_debugging(prefix=dirname(str(key_fn))+'/')

def predict_keyconf_2d(args, ws):
    predict_keyconf(args, ws, worker=_predict_2d_worker)

def estimate_keyconf_clearance(args, ws):
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        if args.only_wait: # Better than indenting the whole loop
            break
        rel_scratch_dir = join(util.SOLVER_SCRATCH, puzzle_name, util.KEYCONF_CLEARANCE_DIR, str(ws.current_trial))
        oskey_fn = ws.oversampled_keyconf_prediction_file(puzzle_name)
        oskey = matio.load(oskey_fn)['KEYQ_OMPL']
        task_shape = (oskey.shape[0])
        total_chunks = partt.guess_chunk_number(task_shape,
                ws.config.getint('SYSTEM', 'CondorQuota') * 4,
                1)
        if args.task_id is None:
            condor_args = [ws.condor_local_exec('facade.py'),
                           'keyconf',
                           '--stage', 'estimate_keyconf_clearance',
                           '--current_trial', str(ws.current_trial),
                           '--puzzle_name', puzzle_name,
                           '--task_id', '$(Process)',
                           ws.local_ws()]
            condor.local_submit(ws,
                                util.PYTHON,
                                iodir_rel=rel_scratch_dir,
                                arguments=condor_args,
                                instances=total_chunks,
                                wait=False)
        else:
            import pyosr
            task_id = 0 if total_chunks == 1 else args.task_id
            tindices = partt.get_task_chunk(task_shape, total_chunks, task_id)
            print("[estimate_keyconf_clearance] tindices {}".format(tindices))
            uw = util.create_unit_world(puzzle_fn)
            nsample = ws.config.getint('Prediction', 'OversamplingClearanceSample')
            distances_batch = []
            for (key_index,) in progressbar(tindices):
                unit_q = uw.translate_ompl_to_unit(oskey[key_index,:])
                unit_q = unit_q.reshape((pyosr.STATE_DIMENSION, 1))
                tup = touchq_util.calc_touch(uw, unit_q, nsample, uw.recommended_cres)
                free_vertices, touch_vertices, to_inf, free_tau, touch_tau = tup
                distances = pyosr.multi_distance(unit_q, free_vertices)
                distances_batch.append(distances)
            ofn = ws.local_ws(rel_scratch_dir, 'clearance_batch-{}.npz'.format(task_id))
            np.savez(ofn, DISTANCE_BATCH=distances_batch)
    if args.task_id is not None:
        return
    if args.no_wait:
        return
    for puzzle_fn, puzzle_name in ws.test_puzzle_generator(args.puzzle_name):
        rel_scratch_dir = join(util.SOLVER_SCRATCH, puzzle_name, util.KEYCONF_CLEARANCE_DIR, str(ws.current_trial))
        condor.local_wait(ws.local_ws(rel_scratch_dir))

function_dict = {
        'generate_atlas2prim' : generate_atlas2prim,
        'predict_keyconf' : predict_keyconf,
        'predict_keyconf_2d' : predict_keyconf_2d,
        'oversample_keyconf' : oversample_keyconf,
        'multinet_oversample_keyconf' : multinet_oversample_keyconf,
        'estimate_keyconf_clearance' : estimate_keyconf_clearance,
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
    p.add_argument('--no_wait', action='store_true')
    p.add_argument('--puzzle_name', help='puzzle name for estimate_keyconf_clearance', default='')
    p.add_argument('--task_id', help='task id for estimate_keyconf_clearance worker process', type=int, default=None)
    util.set_common_arguments(p)


def run(args):
    if args.stage in function_dict:
        ws = util.create_workspace_from_args(args)
        function_dict[args.stage](args, ws)
    else:
        print("Unknown keyconf pipeline stage {}".format(args.stage))

def _remote_command(ws, cmd, auto_retry=True, alter_host='', extra_args=''):
    if not alter_host:
        alter_host = ws.condor_host
    ws.remote_command(alter_host,
                      ws.condor_exec(),
                      ws.condor_ws(),
                      'keyconf', cmd, auto_retry=auto_retry,
                      with_trial=True,
                      extra_args=extra_args)

def remote_estimate_keyconf_clearance(ws):
    _remote_command(ws, 'estimate_keyconf_clearance', extra_args='--no_wait')

#
# Automatic functions start here
#
def collect_stages(variant=0):
    if variant in [0]:
        ret = [
                ('generate_atlas2prim', lambda ws: generate_atlas2prim(None, ws)),
                ('predict_keyconf', lambda ws: predict_keyconf(None, ws)),
                ('upload_keyconf_to_condor', lambda ws: ws.deploy_to_condor(util.TESTING_DIR + '/'))
              ]
    elif variant in [4,6]:
        ret = [
                ('generate_atlas2prim', lambda ws: generate_atlas2prim(None, ws)),
                ('oversample_keyconf', lambda ws: oversample_keyconf(None, ws)),
                ('deploy_to_condor',
                  lambda ws: ws.deploy_to_condor(util.WORKSPACE_SIGNATURE_FILE,
                                                 util.WORKSPACE_CONFIG_FILE,
                                                 util.CONDOR_TEMPLATE,
                                                 util.TESTING_DIR+'/')
                ),
                ('estimate_keyconf_clearance', remote_estimate_keyconf_clearance)
              ]
    elif variant in [7, 10]:
        ret = [
                ('generate_atlas2prim', lambda ws: generate_atlas2prim(None, ws)),
                ('multinet_oversample_keyconf', lambda ws: multinet_oversample_keyconf(None, ws)),
                ('deploy_to_condor',
                  lambda ws: ws.deploy_to_condor(util.WORKSPACE_SIGNATURE_FILE,
                                                 util.WORKSPACE_CONFIG_FILE,
                                                 util.CONDOR_TEMPLATE,
                                                 util.TESTING_DIR+'/')
                ),
                ('estimate_keyconf_clearance', remote_estimate_keyconf_clearance)
              ]
    else:
        assert False, f'Keyconf Pipeline Variant {variant} has not been implemented'
    return ret

def autorun(args):
    ws = util.Workspace(args.dir)
    ws.verify_training_puzzle()
    ws.current_trial = args.current_trial
    pdesc = collect_stages()
    for _,func in pdesc:
        func(ws)
