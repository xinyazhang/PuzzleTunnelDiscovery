#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import util
import os
import envconfig

def setup_parser(subparsers):
    p = subparsers.add('init', help='Initialize a directory as workspace')
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
    edit_nn = util.ask_user(
    print("NEXT STEP: run 'add_puzzle' to add more puzzles for testing, or use 'autorun' to start the whole pipeline")
