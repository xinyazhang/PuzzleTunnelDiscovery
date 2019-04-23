#!/usr/bin/env python3

from six.moves import configparser
import numpy as np
from os.path import abspath, dirname, join, isdir, basename

def read_xyz(config, section, prefix):
    ret = np.zeros(shape=(3), dtype=np.float64)
    for i,suffix in enumerate(['x','y','z']):
        ret[i] = config.getfloat(section, prefix + '.' + suffix)
    return ret

def read_se3state(config, section, prefix):
    tr = read_xyz(config, section, prefix)
    rot_axis = read_xyz(config, section, prefix + '.axis')
    rot_angle = config.getfloat(section, prefix + '.theta')
    return tr, rot_axis, rot_angle

def OmplCfg(object):
    pass

'''
parse_simple:
    Parse the OMPL's puzzle config file. Only the world and robot file names are returned.
'''
def parse_simple(fn):
    config = configparser.ConfigParser()
    config.read([fn])
    puzzle_dir = cfg.puzzle_dir = dirname(fn)
    cfg.env_fn = join(puzzle_dir, config.get("problem", "world"))
    cfg.rob_fn = join(puzzle_dir, config.get("problem", "robot"))
    cfg.env_fn_base = basename(cfg.env_fn)
    cfg.rob_fn_base = basename(cfg.rob_fn)
    cfg.iq_tup = read_se3state(config, 'problem', 'start')
    cfg.gq_tup = read_se3state(config, 'problem', 'goal')
    return cfg, config
