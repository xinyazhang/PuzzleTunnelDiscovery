# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import itertools
'''
uw: pyosr.UnitWorld object
q: Initial state (aka configuration) of the robot
'''
def collision_resolve(uw, q, iter_limit=None):
    STIFFNESS = 1e6
    D_TIME = 0.001
    for x in itertools.count():
        yield q
        if uw.is_valid_state(q):
            break
        if iter_limit is not None:
            if x >= iter_limit:
                break
        tup = uw.intersecting_segments(q)
        fmags = tup[2] * STIFFNESS
        fposs,fdirs = uw.force_direction_from_intersecting_segments(tup[0], tup[1], tup[3])
        reset_velocity = (x == 0)
        q = uw.push_robot(q, fposs, fdirs, fmags, 1.0, D_TIME, reset_velocity)
