#!/usr/bin/env python3

import pyosr
from six.moves import configparser
from os.path import abspath, dirname, join, isdir

def read_xyz(config, section, prefix):
    ret = np.zeros(shape=(3), dtype=np.float64)
    for i,suffix in enumerate(['x','y','z']):
        ret[i] = config.getfloat(section, prefix + '.' + suffix)
    return ret

def read_state(config, section, prefix):
    tr = read_xyz(config, section, prefix)
    rot_axis = read_xyz(config, section, prefix + '.axis')
    rot_angle = config.getfloat(section, prefix + '.theta')
    q = pyosr.compose_from_angleaxis(tr, rot_angle, rot_axis)
    return q.reshape((1, pyosr.STATE_DIMENSION))

def parse(args, puzzle_fn):
    config = configparser.ConfigParser()
    config.read([puzzle_fn])
    iq = read_state(config, 'problem', 'start')
    gq = read_state(config, 'problem', 'goal')
    ompl_q = np.concatenate((iq, gq), axis=0)

    args._env_fn = join(puzzle_dir, config.get("problem", "world"))
    args._rob_fn = join(puzzle_dir, config.get("problem", "robot"))
    args._istate_ompl = iq
    args._gstate_ompl = gq
