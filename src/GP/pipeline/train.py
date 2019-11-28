#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join, isdir
import copy
import subprocess
import pathlib
import numpy as np
from imageio import imwrite as imsave
import shutil
import time
from multiprocessing import Process
import resource

from . import util
from . import choice_formatter
try:
    from . import hg_launcher
except ImportError as e:
    util.warn("[WARNING] CANNOT IMPORT hg_launcher. This node is incapable of training/prediction")
    # Note: do NOT raise exceptions in modules loaded by __init__.py

try:
    import fcntl
    def global_gpu_lock(ws):
        util.log('waiting for GPU resource')
        ws.timekeeper_start('wait for GPU resource')
        ws.gpu_lock_fd = open('/tmp/mkobs3d.gpulock', 'w')
        fcntl.lockf(ws.gpu_lock_fd, fcntl.LOCK_EX)
        pid = os.getpid()
        print(pid, file=ws.gpu_lock_fd)
        util.log('waited for GPU resource')
    def global_gpu_unlock(ws):
        ws.timekeeper_finish('wait for GPU resource')
        fcntl.lockf(ws.gpu_lock_fd, fcntl.LOCK_UN)
        ws.gpu_lock_fd.close()
        ws.gpu_lock_fd = None
except ImportError as e:
    util.warn("[WARNING] CANNOT IMPORT fcntl, global GPU lock is disabled")
    def global_gpu_lock(ws):
        pass
    def global_gpu_unlock(ws):
        pass

def _deploy(ws):
    ws.deploy_to_gpu(util.WORKSPACE_SIGNATURE_FILE,
                     util.WORKSPACE_CONFIG_FILE,
                     util.TRAINING_DIR+'/',
                     util.TESTING_DIR+'/')
    if isdir(ws.local_ws(util.EXTRA_TRAINING_DIR)):
        ws.deploy_to_gpu(util.EXTRA_TRAINING_DIR + '/')

def _fetch(ws):
    ws.fetch_gpu(util.TESTING_DIR+'/')

def write_pidfile(pidfile, pid):
    with open(pidfile, 'w') as f:
        print(pid, file=f)

def _train(args, ws, geo_type):
    if ws.nn_tags:
        params, ws.nn_profile = hg_launcher.create_config_from_tagstring(ws.nn_tags)
    elif ws.nn_profile:
        params = hg_launcher.create_config_from_profile(ws.nn_profile)
    else:
        params = hg_launcher.create_default_config()

    all_omplcfgs = []
    for puzzle_fn, puzzle_name in ws.training_puzzle_generator():
        all_omplcfgs.append(puzzle_fn)
    params['all_ompl_configs'] = all_omplcfgs

    params['what_to_render'] = geo_type
    params['checkpoint_dir'] = ws.checkpoint_dir(geo_type) + '/'
    params['suppress_hot'] = 0.0
    params['suppress_cold'] = 0.7
    params['dataset_name'] = f'{ws.dir}.{geo_type}'
    params['load_epoch'] = args.load_epoch
    global_gpu_lock(ws)
    ws.timekeeper_start('train_{}'.format(geo_type))

    os.makedirs(ws.local_ws(util.NEURAL_SCRATCH), exist_ok=True)
    pidfile = ws.local_ws(util.NEURAL_SCRATCH, geo_type + '.pid')
    write_pidfile(pidfile, os.getpid())
    hg_launcher.launch_with_params(params, do_training=True, load=args.load)
    write_pidfile(pidfile, -1)

    ws.timekeeper_finish('train_{}'.format(geo_type))
    global_gpu_unlock(ws)

def train_rob(args, ws):
    if args.only_wait:
        print("Note: --only_wait has no effect in train_rob")
    _train(args, ws, 'rob')

def train_env(args, ws):
    if args.only_wait:
        print("Note: --only_wait has no effect in train_env")
    _train(args, ws, 'env')

'''
Train a single network for both pieces of the geometry
'''
def train_both(args, ws):
    if args.only_wait:
        print("Note: --only_wait has no effect in train_both")
    _train(args, ws, 'both')

def wait_for_training(args, ws):
    for geo_type in ['rob', 'env']:
        pidfile = ws.local_ws(util.NEURAL_SCRATCH, geo_type + '.pid')
        util.log('[wait_for_training] wait for file {}'.format(pidfile))
        last_valid_pid = -1
        while True:
            pid = 0
            with open(pidfile, 'r') as f:
                for line in f:
                    for s in line.split(' '):
                        pid = int(s)
                        break
            if pid > 0:
                last_valid_pid = pid
            if pid < 0:
                break
        util.log("[wait_for_training] {} (pid: {}) waited".format(geo_type, pid))
        write_pidfile(pidfile, -1)

"""
Note: we separate geo_type and checkpoint_geo_type.
geo_type controls what to render as the testing image
checkpoint_geo_type controls where to load the trained weights.

The separation is due to the introduction of joint network.
"""
def _predict_surface(args, ws, geo_type, generator, checkpoint_geo_type=None):
    rews_dir = ws.config.get('Prediction', 'ReuseWorkspace', fallback='')
    if rews_dir:
        rews_dir = join(ws.dir, rews_dir) # Relative path
        rews = util.Workspace(rews_dir)
        rews.nn_profile = ws.nn_profile
    else:
        rews = ws
    if checkpoint_geo_type is None:
        checkpoint_geo_type = geo_type
    for puzzle_fn, puzzle_name in generator():
        if ws.nn_tags:
            params, rews.nn_profile = hg_launcher.create_config_from_tagstring(ws.nn_tags)
        elif ws.nn_profile:
            params = hg_launcher.create_config_from_profile(ws.nn_profile)
        else:
            params = hg_launcher.create_default_config()
        if args.puzzle_name is not None and args.puzzle_name != puzzle_name:
            continue
        global_gpu_lock(ws)
        ws.timekeeper_start('predict_{}'.format(geo_type), puzzle_name)
        params['all_ompl_configs'] = [puzzle_fn]
        params['what_to_render'] = geo_type
        params['checkpoint_dir'] = rews.checkpoint_dir(checkpoint_geo_type) + '/'
        if args.load_epoch is not None:
            params['epoch_to_load'] = args.load_epoch
            params['debug_predction'] = True
            params['prediction_epoch_size'] = 32
        if rews_dir:
            """
            In this case the checkpoint_dir may belong to another workspace.
            At the same time our current workspace may not have a checkpoint dir.
            """
            os.makedirs(ws.checkpoint_dir(checkpoint_geo_type), exist_ok=True)
            params['output_dir'] = ws.checkpoint_dir(checkpoint_geo_type) + '/'
            params['prediction_output'] = ws.atex_prediction_file(puzzle_fn, geo_type)
        params['dataset_name'] = puzzle_name # Enforce the generated filename
        util.log("[prediction] Predicting {}:{}".format(puzzle_fn, geo_type))
        # NEVER call launch_with_params in the same process for multiple times
        # TODO: add assertion to handle this problem
        proc = Process(target=hg_launcher.launch_with_params, args=(params, False))
        # hg_launcher.launch_with_params(params, do_training=False)
        proc.start()
        proc.join()
        """
        src = join(ws.checkpoint_dir(checkpoint_geo_type), '{}-atex.npz'.format(puzzle_name))
        dst = ws.atex_prediction_file(puzzle_fn, geo_type)
        util.log("[prediction] Copy surface prediction file {} => {}".format(src, dst))
        shutil.copy(src, dst)
        """
        ws.timekeeper_finish('predict_{}'.format(geo_type), puzzle_name)
        global_gpu_unlock(ws)

def predict_rob(args, ws):
    _predict_surface(args, ws, 'rob', ws.test_puzzle_generator)

def predict_env(args, ws):
    _predict_surface(args, ws, 'env', ws.test_puzzle_generator)

def predict_both(args, ws):
    _predict_surface(args, ws, 'rob', ws.test_puzzle_generator, checkpoint_geo_type='both')
    _predict_surface(args, ws, 'env', ws.test_puzzle_generator, checkpoint_geo_type='both')

def validate_rob(args, ws):
    _predict_surface(args, ws, 'rob', ws.training_puzzle_generator)

def validate_env(args, ws):
    _predict_surface(args, ws, 'env', ws.training_puzzle_generator)

function_dict = {
        'train_rob' : train_rob,
        'train_env' : train_env,
        'train_both' : train_both,
        'wait_for_training' : wait_for_training,
        'predict_rob' : predict_rob,
        'predict_env' : predict_env,
        'predict_both' : predict_both,
        'validate_rob' : validate_rob,
        'validate_env' : validate_env,
}

def setup_parser(subparsers):
    p = subparsers.add_parser('train',
                              help='Training/Prediction',
                              formatter_class=choice_formatter.Formatter)
    p.add_argument('--stage',
                   choices=list(function_dict.keys()),
                   help='R|Possible stages:\n'+'\n'.join(list(function_dict.keys())),
                   default='',
                   metavar='')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--nn_profile', help="NN Profile", default='256hg')
    p.add_argument('--load', help="Load existing checkpoints and continue", action='store_true')
    p.add_argument('--load_epoch', help="Load checkpoint at specific epoch either as the starting epoch to train or predict", default=None, type=int)
    p.add_argument('--puzzle_name', help="puzzle to predict", default=None)
    util.set_common_arguments(p)

# As always, run() serves as a separator between local function and remote proxy functions
def run(args):
    if args.stage in function_dict:
        # Unset resource limit
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        ws = util.create_workspace_from_args(args)
        if args.nn_profile.startswith('tag:'):
            ws.nn_tags = args.nn_profile[len('tag:'):]
            ws.nn_profile = ''
        else:
            ws.nn_profile = args.nn_profile
        function_dict[args.stage](args, ws)
    else:
        print("Unknown train pipeline stage {}".format(args.stage))

def _remote_command(ws, cmd, auto_retry, in_tmux):
    ws.remote_command(ws.gpu_host,
                      ws.gpu_exec(),
                      ws.gpu_ws(),
                      'train',
                      cmd,
                      auto_retry=auto_retry,
                      in_tmux=in_tmux,
                      with_trial=True,
                      use_nn_profile=True)

def remote_train_rob(ws):
    _remote_command(ws, 'train_rob', auto_retry=True, in_tmux=True)

def remote_train_env(ws):
    _remote_command(ws, 'train_env', auto_retry=True, in_tmux=True)

def remote_train_both(ws):
    _remote_command(ws, 'train_both', auto_retry=True, in_tmux=True)

def remote_wait_for_training(ws):
    _remote_command(ws, 'wait_for_training', auto_retry=True, in_tmux=False)

def remote_predict_rob(ws):
    _remote_command(ws, 'predict_rob', auto_retry=True, in_tmux=False)

def remote_predict_env(ws):
    _remote_command(ws, 'predict_env', auto_retry=True, in_tmux=False)

def remote_predict_both(ws):
    _remote_command(ws, 'predict_both', auto_retry=True, in_tmux=False)

def collect_stages(variant=0):
    if variant in [0]:
        return [ ('deploy_to_gpu', _deploy),
                 ('train_rob', remote_train_rob),
                 ('train_env', remote_train_env),
                 ('wait_for_training', remote_wait_for_training),
                 ('Break', lambda _: util.ack('[Break] Dummy stage between training phase and testing phase')),
                 ('predict_rob', remote_predict_rob),
                 ('predict_env', remote_predict_env),
                 ('fetch_from_gpu', _fetch),
               ]
    elif variant in [4,6]:
        return [
                 ('deploy_to_gpu', _deploy),
                 # ('predict_rob', remote_predict_rob),
                 # ('predict_env', remote_predict_env),
                 ('predict_both', remote_predict_both),
                 ('fetch_from_gpu', _fetch),
               ]
    assert False, f'Train Pipeline Variant {variant} has not been implemented'
