#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from . import envconfig
from . import util
from . import add_puzzle

def setup_parser(subparsers):
    p = subparsers.add_parser('mimic', help='Initialize a workspace with a template workspace')
    p.add_argument('old', help='Template workspace directory')
    p.add_argument('dir', help='New workspace directory')
    p.add_argument('--training_puzzle', help='Puzzle as training data', default='')

def run(args):
    oldws = util.Workspace(args.old)
    ws = util.Workspace(args.dir, init=True)
    if ws.test_signature():
        util.fatal("{} is already initialized".format(ws.dir))
        exit()
    args.condor = oldws.condor_template
    os.makedirs(ws.dir, exist_ok=True)
    os.makedirs(ws.training_dir, exist_ok=True)
    ws.touch_signature()
    envconfig.init_config_file(args, ws, oldws=oldws)
    if args.training_puzzle:
        if not add_puzzle.copy_puzzle(ws.training_dir, args.training_puzzle, ws.chart_resolution):
            util.fatal('[init] Could not add puzzle {}'.format(args.training_puzzle))
            return
    else:
        util.warn('[init] NO TRAINING PUZZLE WAS SPECIFIED. This workspace is incapable of NN pipeline')
    util.ack("[init] workspace initalized: {}".format(ws.dir))
    util.ack("[init] The following puzzle has been added as training sample")
    util.ack("[init] \t{}".format(args.training_puzzle))
    util.ack("[init] NEXT STEP: run 'add_puzzle' to add more puzzles for testing, or use 'autorun' to start the whole pipeline")
