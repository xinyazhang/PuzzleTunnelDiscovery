#!/usr/bin/env python3

import shutil
import os
from os.path import join, basename, abspath
import subprocess
import configparser
import multiprocessing

from . import parse_ompl
from . import util

def setup_parser(subparsers):
    p = subparsers.add_parser('add_puzzle', help='Add a puzzle to solve, the puzzle will be named after its file name')
    p.add_argument('dir', help='Workspace directory')
    p.add_argument('puzzles', help='One or more OMPL .cfg file(s)', nargs='+')

def copy_puzzle_geometry(obj_from, target_file, chart_resolution):
    util.log('[copy_puzzle_geometry] {} -> {}'.format(obj_from, target_file))
    if os.path.isfile(target_file):
        if not util.ask_user("{} exists, overwriting?".format(target_file)):
            return
    res = 128.0
    while util.shell(['./objautouv',
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
    cfg, config = parse_ompl.parse_simple(puzzle_file)
    # Sanity check
    try:
        config.getfloat('problem', 'collision_resolution')
    except configparser.NoOptionError as e:
        util.warn('[copy_puzzle] missing collision_resolution in puzzle {}'.format(puzzle_file))
        util.warn('[copy_puzzle] This puzzle will not be added')
        return False
    # Copy geometry
    os.makedirs(tgt_dir, exist_ok=True)
    copy_puzzle_geometry(cfg.env_fn, join(tgt_dir, cfg.env_fn_base), chart_resolution)
    if cfg.env_fn != cfg.rob_fn:
        assert cfg.env_fn_base != cfg.rob_fn_base
        copy_puzzle_geometry(cfg.rob_fn, join(tgt_dir, cfg.rob_fn_base), chart_resolution)
    # Change geometry file name
    config.set("problem", "world", cfg.env_fn_base)
    config.set("problem", "robot", cfg.rob_fn_base)
    # Write to new cfg file
    tgt_cfg = join(tgt_dir, util.PUZZLE_CFG_FILE)
    with open(tgt_cfg, 'w') as f:
        config.write(f)
        util.ack('''[copy_puzzle] copy configuration file {} => {}'''.format(puzzle_file, abspath(tgt_cfg)))
    old_uw = util.create_unit_world(puzzle_file)
    new_uw = util.create_unit_world(tgt_cfg)
    old_iq = parse_ompl.tup_to_ompl(cfg.iq_tup)
    old_gq = parse_ompl.tup_to_ompl(cfg.gq_tup)
    van_iq = old_uw.translate_ompl_to_vanilla(old_iq)
    van_gq = old_uw.translate_ompl_to_vanilla(old_gq)
    new_iq = new_uw.translate_vanilla_to_ompl(van_iq)
    new_gq = new_uw.translate_vanilla_to_ompl(van_gq)
    if not new_uw.is_valid_state(new_uw.translate_ompl_to_unit(new_iq)[0]):
        util.fatal("[copy_puzzle] The update geometry renders its initial OMPL state invalid")
        return False
    if not new_uw.is_valid_state(new_uw.translate_ompl_to_unit(new_gq)[0]):
        util.fatal("[copy_puzzle] The update geometry renders its goal OMPL state invalid")
        return False
    # Sanity check
    san_iq = new_uw.translate_ompl_to_vanilla(new_iq)
    san_gq = new_uw.translate_ompl_to_vanilla(new_gq)
    parse_ompl.update_se3state(config, 'problem', 'start', new_iq)
    parse_ompl.update_se3state(config, 'problem', 'goal', new_gq)
    with open(tgt_cfg, 'w') as f:
        config.write(f)
        util.log('''[copy_puzzle] update config''')
        util.log('''[copy_puzzle] istate translation:''')
        util.log('''[copy_puzzle] \t old: {}'''.format(old_iq))
        util.log('''[copy_puzzle] \t van: {}'''.format(van_iq))
        util.log('''[copy_puzzle] \t new: {}'''.format(new_iq))
        util.log('''[copy_puzzle] \t san: {}'''.format(san_iq))
        util.log('''[copy_puzzle] gstate translation:''')
        util.log('''[copy_puzzle] \t old: {}'''.format(old_gq))
        util.log('''[copy_puzzle] \t van: {}'''.format(van_gq))
        util.log('''[copy_puzzle] \t new: {}'''.format(new_gq))
        util.log('''[copy_puzzle] \t san: {}'''.format(san_gq))
    return True

def add_testing_puzzle(ws, fn):
    fn_base = util.trim_suffix(basename(fn))
    tgt_dir = join(ws.testing_dir, fn_base)
    if os.path.isdir(tgt_dir):
        do_override = util.ask_user("{} is already existing, override?".format(tgt_dir))
        if not do_override:
            return
    copy_puzzle(tgt_dir, fn, ws.chart_resolution)

def _mt_add_testing_puzzle(arg_tup):
    args, puzzle = arg_tup
    ws = util.Workspace(args.dir)
    add_testing_puzzle(ws, puzzle)

def run(args):
    pcpu = multiprocessing.Pool()
    mpargs = [ (args, puzzle) for puzzle in args.puzzles]
    pcpu.map(_mt_add_testing_puzzle, mpargs)
