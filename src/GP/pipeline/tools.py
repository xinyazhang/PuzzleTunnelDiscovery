#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isdir, isfile
import copy
import pathlib
import numpy as np

from . import util
from . import matio
from . import condor
from . import parse_ompl

def read_roots(args):
    uw = util.create_unit_world(args.puzzle_fn)
    ompl_q = matio.load(args.roots, key=args.roots_key)
    print("OMPL_Q {}".format(ompl_q.shape))
    unit_q = uw.translate_ompl_to_unit(ompl_q)
    matio.savetxt(args.out, unit_q)

def _visgt(args):
    ws = util.Workspace(args.dir)
    cfg, config = parse_ompl.parse_simple(ws.training_puzzle)
    puzzle_fn = None
    puzzle_fn = cfg.env_fn if args.geo_type == 'env' else puzzle_fn
    puzzle_fn = cfg.rob_fn if args.geo_type == 'rob' else puzzle_fn
    p = pathlib.Path(puzzle_fn)
    util.shell(['./vistexture', puzzle_fn, str(p.with_suffix('.png')),])

def visenvgt(args):
    args.geo_type = 'env'
    _visgt(args)

def visrobgt(args):
    args.geo_type = 'rob'
    _visgt(args)

def vistraj(args):
    ws = util.Workspace(args.dir)
    ws.fetch_condor(util.TRAJECTORY_DIR + '/')
    trajfn = ws.local_ws(util.TRAJECTORY_DIR, 'traj_{}.npz'.format(args.traj_id))
    utrajfn = ws.local_ws(util.TRAJECTORY_DIR, 'traj_{}.unit.txt'.format(args.traj_id))
    uw = util.create_unit_world(ws.training_puzzle)
    ompl_q = matio.load(trajfn, key='OMPL_TRAJECTORY')
    unit_q = uw.translate_ompl_to_unit(ompl_q)
    matio.savetxt(utrajfn, unit_q)
    cfg, config = parse_ompl.parse_simple(ws.training_puzzle)
    util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, utrajfn, '0.5'])

def plotstat(args):
    import matplotlib
    import matplotlib.pyplot as plt
    '''
    Load keys
    '''
    ws = util.Workspace(args.dir)
    ws.fetch_condor(util.TRAINING_DIR, 'KeyCan.npz')
    keycan_ompl = matio.load(ws.local_ws(util.TRAINING_DIR, 'KeyCan.npz'), key='OMPL_CANDIDATES')
    uw = util.create_unit_world(ws.training_puzzle)
    keycan = uw.translate_ompl_to_unit(keycan_ompl)
    '''
    Load stats
    '''
    st = matio.load(ws.local_ws(util.KEY_FILE), key='_STAT')
    medians, maxs, mins, means, stddevs = st[:,:]
    indices=range(medians.shape[0])
    TOP_K= args.top_k if args.top_k is not None else ws.config.getint('TrainingKeyConf', 'KeyConf')
    top_k_medians = np.array(medians).argsort()[:TOP_K]
    top_k_means = np.array(means).argsort()[:TOP_K]
    matio.savetxt('keyq.unit.txt', keycan[top_k_means])
    '''
    Vis stats
    '''
    plt.plot(indices, means, 'r-')
    plt.show()

    cfg, config = parse_ompl.parse_simple(ws.training_puzzle)
    util.shell(['./vispath', cfg.env_fn, cfg.rob_fn, 'keyq.unit.txt', '0.5'])

function_dict = {
        'read_roots' : read_roots,
        'visenvgt' : visenvgt,
        'visrobgt' : visrobgt,
        'vistraj' : vistraj,
        'plotstat' : plotstat,
}

def setup_parser(subparsers):
    sp = subparsers.add_parser('tools', help='Various Tools.')
    toolp = sp.add_subparsers(dest='tool_name', help='Name of Tool')
    p = toolp.add_parser('read_roots', help='Dump roots of the forest to text file')
    p.add_argument('--roots_key', help='NPZ file of roots', default='KEYQ_OMPL')
    p.add_argument('puzzle_fn', help='OMPL config')
    p.add_argument('roots', help='NPZ file of roots')
    p.add_argument('out', help='output txt file')
    p = toolp.add_parser('visenvgt', help='Call vistexture to visualize the training data')
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('visrobgt', help='Call vistexture to visualize the training data')
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('vistraj', help='Call vistexture to visualize the training data')
    p.add_argument('--traj_id', help='Trajectory ID', default=0)
    p.add_argument('dir', help='Workspace directory')
    p = toolp.add_parser('plotstat', help='Call vistexture to visualize the training data')
    p.add_argument('--top_k', help='Top K', type=int, default=None)
    p.add_argument('dir', help='Workspace directory')

def run(args):
    function_dict[args.tool_name](args)
