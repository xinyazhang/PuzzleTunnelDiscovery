#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
# -*- coding: utf-8 -*-

import os
from . import envconfig
from . import util
from . import add_puzzle

def setup_parser(subparsers):
    p = subparsers.add_parser('init', help='Initialize a directory as workspace. Puzzles shall be added through add_puzzle command later.')
    p.add_argument('dir', help='Workspace directory')
    p.add_argument('--condor', help='Condor Template file', required=True)
    p.add_argument('--training_puzzle', help='Puzzle as training data', default='')
    p.add_argument('--quiet', help='Quietly create a workspace for single node environment. By default it creates a distrubted workspace and thus requires human inputs to complete.', action='store_true')
    p.add_argument('--trained_workspace', help='Reused trained networks from the given workspace. Note: this fills ReuseWorkspace automatically, NN pipeline would fail if ReuseWorkspace is not configured.', default='')
    # Testing puzzle shall be added later
    # p.add_argument('--test_puzzle', help='Workspace directory', nargs='*', default=[])

def run(args):
    ws = util.Workspace(args.dir, init=True)
    os.makedirs(ws.dir, exist_ok=True)
    os.makedirs(ws.training_dir, exist_ok=True)
    ws.touch_signature()
    envconfig.init_config_file(args, ws)
    if args.training_puzzle:
        if not add_puzzle.copy_puzzle(ws.training_dir, args.training_puzzle, ws.chart_resolution):
            util.fatal('[init] Could not add puzzle {}'.format(args.training_puzzle))
            return
    else:
        util.warn('[init] NO TRAINING PUZZLE WAS SPECIFIED. This workspace is incapable of preparing training data for NN')
    '''
    for p in args.test_puzzle:
        add_puzzle.add_testing_puzzle(args, ws, p)
    '''
    util.ack("[init] workspace initalized: {}".format(ws.dir))
    # util.ack("[init] The following puzzle has been added as training sample")
    # util.ack("[init] \t{}".format(args.training_puzzle))
    # util.ack("[init] The following puzzle(s) have been added as testing sample")
    # for p in args.test_puzzle:
    #     util.ack("[init] \t{}".format(p))
    util.ack("[init] NEXT STEP: run 'add_puzzle' to add testing puzzles, or use 'autorun' to start the whole pipeline")
