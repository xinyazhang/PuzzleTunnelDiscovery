#!/usr/bin/env python3

import shutil
import os
import pathlib
from os.path import join, basename, abspath
import configparser

from . import add_puzzle
from . import parse_ompl
from . import util

def setup_parser(subparsers):
    p = subparsers.add_parser('copy_training_data', help='Copy the training data from one workspace to another one')
    p.add_argument('old', help='Workspace with existing training data')
    p.add_argument('dir', help='Target workspace')
    p.add_argument('name', help='Name for this extra training data')

def run(args):
    oldws = util.Workspace(args.old)
    ws = util.Workspace(args.dir)
    tgt_puzzle_dir = ws.local_ws(util.EXTRA_TRAINING_DIR, args.name)
    add_puzzle.copy_puzzle(tgt_puzzle_dir, oldws.training_puzzle, oldws.chart_resolution,
                           vanilla=True)
    srcp = pathlib.Path(oldws.training_dir)
    tgtp = pathlib.Path(tgt_puzzle_dir)
    for geo_type in ['env', 'rob']:
        srctex = str(srcp.joinpath('{}_chart_screened_uniform.png'.format(geo_type)))
        tgttex = str(tgtp.joinpath('{}_chart_screened_uniform.png'.format(geo_type)))
        shutil.copy(srctex, tgttex)
        util.ack('[copy_training_data] {} -> {}'.format(srctex, tgttex))
