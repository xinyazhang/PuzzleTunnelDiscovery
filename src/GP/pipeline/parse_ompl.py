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
    return tr, rot_angle, rot_axis

'''
Placeholder object like argparse.Namespace
'''
class OmplCfg(object):
    pass

'''
parse_simple:
    Parse the OMPL's puzzle config file. Only the world and robot file names are returned.

Convention:
    cfg is OmplCfg object
    config is ConfigParser object
'''
def parse_simple(fn):
    config = configparser.ConfigParser()
    config.read([fn])
    cfg = OmplCfg()
    puzzle_dir = cfg.puzzle_dir = dirname(fn)
    cfg.env_fn = join(puzzle_dir, config.get("problem", "world"))
    cfg.rob_fn = join(puzzle_dir, config.get("problem", "robot"))
    cfg.env_fn_base = basename(cfg.env_fn)
    cfg.rob_fn_base = basename(cfg.rob_fn)
    cfg.iq_tup = read_se3state(config, 'problem', 'start')
    cfg.gq_tup = read_se3state(config, 'problem', 'goal')
    return cfg, config

def tup_to_ompl(tup):
    import pyosr
    tr, rot_angle, rot_axis = tup
    q = pyosr.compose_from_angleaxis(tr, rot_angle, rot_axis)
    assert pyosr.STATE_DIMENSION == 7, "FIXME: More flexible w-first to w-last"
    q = q.reshape((1, pyosr.STATE_DIMENSION))
    q[:, [6,3,4,5]] = q[:, [3,4,5,6]] # W-first (pyOSR) to W-last (OMPL)
    return q

def ompl_to_tup(q):
    import pyosr
    q = np.copy(q) # Grab a copy instead of working in-place
    q[:, [3,4,5,6]] = q[:, [6,3,4,5]] # W-last to W-first so pyosr.decompose_3 works
    q = q.reshape((pyosr.STATE_DIMENSION, 1))
    tr, rot_angle, rot_axis = pyosr.decompose_3(q)
    return tr, rot_angle, rot_axis

def update_xyz(config, section, prefix, xyz):
    xyz = xyz.reshape((3))
    for i,suffix in enumerate(['x','y','z']):
        config.set(section, prefix + '.' + suffix, str(xyz[i]))

def update_se3state(config, section, prefix, q):
    tr, rot_angle, rot_axis = ompl_to_tup(q)
    update_xyz(config, section, prefix, tr)
    config.set(section, prefix + '.theta', str(rot_angle))
    update_xyz(config, section, prefix + '.axis', rot_axis)

