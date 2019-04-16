#!/usr/bin/env python3

from six.moves import configparser
import numpy as np
from os.path import abspath, dirname, join, isdir, basename

def read_xyz(config, section, prefix):
    ret = np.zeros(shape=(3), dtype=np.float64)
    for i,suffix in enumerate(['x','y','z']):
        ret[i] = config.getfloat(section, prefix + '.' + suffix)
    return ret

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
    return cfg, config
