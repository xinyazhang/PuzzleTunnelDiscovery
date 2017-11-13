import pyosr
import numpy as np
from math import sqrt,pi,sin,cos

def gen_init_state(uw):
    while True:
        tr = np.random.rand(3) * 2.0 - 1.0
        u1,u2,u3 = np.random.rand(3)
        quat = [sqrt(1-u1)*sin(2*pi*u2),
                sqrt(1-u1)*cos(2*pi*u2),
                sqrt(u1)*sin(2*pi*u3),
                sqrt(u1)*cos(2*pi*u3)]
        part1 = np.array(tr, dtype=np.float32)
        part2 = np.array(quat, dtype=np.float32)
        state = np.concatenate((part1, part2))
        if uw.is_disentangled(state):
            continue
        if uw.is_valid_state(state):
            break
    return uw.translate_from_unit_state(state)
