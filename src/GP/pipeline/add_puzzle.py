#!/usr/bin/env python3

import shutil
import os
from os.path import join, basename, abspath
import subprocess

import parse_ompl
import util

def setup_parser(subparsers):
    p = subparsers.add('add_puzzle', help='Add a puzzle to solve, the puzzle will be named after its file name')
    p.add_argument('dir', help='Workspace directory')
    p.add_argument('puzzles', help='One or more OMPL .cfg file(s)', nargs='+')

def copy_puzzle_geometry(obj_from, target_file, chart_resolution):
    res = 64.0
    while subprocess.call(['./objautouv',
                           '-f',
                           '-r', str(chart_resolution),
                           '-w', str(res),
                           '-h', str(res),
                           '-m', str(util.PIXMARGIN),
                           obj_from, target_file]) != 0:
        res *= 2.0
    print('''[objautouv] {} => {}'''.format(obj_from, target_file))

'''
Copy the puzzle_file and its related geometry files to $workspace/$tgt_dir,
and substitute the problem.world and problem.robot with relative path.

We cannot simply copy the .cfg file from OMPL because
1. The geometry files were not there
2. OMPL .cfg contains absolute path, which is not portable.

Side effect:
    tgt_dir will be created if not exist
'''
def copy_puzzle(tgt_dir, puzzle_file, chart_resolution):
    puzzle, config = parse_ompl.parse_simple(puzzle_file)
    # Sanity check
    config.getfloat('problem', 'collision_resolution')
    os.makedirs(tgt_dir, exist_ok=True)
    copy_puzzle_geometry(cfg.env_fn, join(tgt_dir, cfg.env_fn_base), chart_resolution)
    if cfg.env_fn != cfg.rob_fn:
        assert cfg.env_fn_base != cfg.rob_fn_base
        copy_puzzle_geometry(cfg.rob_fn, join(tgt_dir, cfg.rob_fn_base))
    config.set("problem", "world", cfg.env_fn_base)
    config.set("problem", "robot", cfg.rob_fn_base)
    tgt_cfg = join(tgt_dir, util.PUZZLE_CFG_FILE)
    with open(tgt_cfg, 'w') as f:
        config.write(f)
        print('''[copy_puzzle] {} => {}'''.format(puzzle_file, abspath(tgt_cfg)))

def add_testing_puzzle(ws, fn):
    fn_base = os.path.splitext(basename(fn))[0]
    tgt_dir = join(ws.testing_dir, fn_base)
    if os.path.isdir(tgt_dir):
        do_override = util.ask_user("{} is already existing, override?".format(tgt_dir))
        if not do_override:
            return
    copy_puzzle(tgt_dir, fn, ws.chart_resolution)

def run(args):
    ws = util.Workspace(args.dir)
    for p in args.puzzles:
        add_testing_puzzle(ws, p)
