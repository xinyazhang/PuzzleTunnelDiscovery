#!/usr/bin/env python3

import shutil
import os
from pathlib import Path
from os.path import join, basename, abspath
import subprocess
import configparser
import multiprocessing

from . import parse_ompl
from . import util
from . import add_puzzle

def setup_parser(subparsers):
    p = subparsers.add_parser('add_extra', help='Add more ground truth to train')
    p.add_argument('dir', help='Target Workspace to add more ground truth')
    p.add_argument('--extras', help='Other Workspaces', nargs='+')

def _copy(src, tgt):
    util.log("[add_extra._copy] {} => {}".format(src, tgt))
    shutil.copy(src, tgt)

def add_extra_training_data(ws, ex):
    exws = util.Workspace(ex)
    puzzle_file = exws.training_puzzle
    exp = Path(ex)
    tgt_dir = ws.local_ws(util.EXTRA_TRAINING_DIR, exp.name)
    os.makedirs(tgt_dir)
    cfg, config = parse_ompl.parse_simple(puzzle_file)

    _copy(puzzle_file, tgt_dir)
    _copy(cfg.env_fn, tgt_dir)
    _copy(cfg.rob_fn, tgt_dir)
    _copy(Path(cfg.env_fn).with_suffix('.png'), tgt_dir)
    _copy(Path(cfg.rob_fn).with_suffix('.png'), tgt_dir)

def run(args):
    ws = util.Workspace(args.dir)
    for ex in args.extras:
        add_extra_training_data(ws, ex)
