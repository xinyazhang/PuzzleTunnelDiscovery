#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from . import envconfig
from . import util

def setup_parser(subparsers):
    p = subparsers.add_parser('init', help='Initialize a directory as workspace')
    p.add_argument('dir', help='Workspace directory')
    p.add_argument('training_puzzle', help='Puzzle as training data')
    p.add_argument('--test_puzzle', help='Workspace directory', nargs='*', default=[])

def run(args):
    ws = util.Workspace(args.dir, init=True)
    os.makedirs(ws.dir, exist_ok=True)
    ws.touch_signature()
    envconfig.init_config_file(ws)
    add_puzzle.copy_puzzle(ws.training_dir(), args.training_puzzle, ws.chart_resolution)
    for p in args.test_puzzle:
        add_puzzle.add_testing_puzzle(ws, p)
    print("Current workspace {}".format(ws.dir))
    print("The following puzzle has been added as training sample")
    print("\t{}".format(args.train))
    print("The following puzzle(s) have been added as testing sample")
    for p in args.test_puzzle:
        print("\t{}".format(p))
    print("NEXT STEP: run 'add_puzzle' to add more puzzles for testing, or use 'autorun' to start the whole pipeline")
