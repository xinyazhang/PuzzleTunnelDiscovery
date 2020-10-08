#!/usr/bin/env python2

import os
import sys
sys.path.append(os.getcwd())

import pyosr
import aniconf12 as aniconf
import numpy as np
import phyutil

'''
solve: read a segment of configurations from .npz file, and write the solving
       trajectory to an output .npz file.

Input:
    fn_pred     : .npz file that stores predicted (by NN) Configurations (Qs) in 'TOCRQS'
    start       : start index in TOCRQS
    end         : end index in TOCRQS
    fn_resolved : output .npz file with elements:
                  ALL_RQS: a list of np.array of the solving trajectory
                           (including the input and the solved state)
                  ALL_IS_RESOLVED: a binary array indicating the final state is
                                   collision free or not.
                  RANGE: (start, end) tuple according to the input, clamped to
                         the size of TOCRQS.
'''
def solve(fn_pred, start, end, fn_resolved):
    r = pyosr.UnitWorld() # pyosr.Renderer is not avaliable in HTCondor
    r.loadModelFromFile(aniconf.env_wt_fn)
    r.loadRobotFromFile(aniconf.rob_wt_fn)
    r.enforceRobotCenter(aniconf.rob_ompl_center)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)

    dic = np.load(fn_pred)
    to_cr_qs = dic['TOCRQS']
    all_rqs = []
    all_is_solved = []
    end = min(len(to_cr_qs), end) # Bound to the size limit
    for q in to_cr_qs[start:end]:
        RQS = [] # Resolving Qs
        for rq in phyutil.collision_resolve(r, q, 1024):
            RQS.append(rq)
        all_rqs.append(RQS)
        all_is_solved.append(r.is_valid_state(RQS[-1]))
    np.savez(fn_resolved, ALL_RQS=all_rqs, ALL_IS_RESOLVED=all_is_solved, RANGE=[start, end])

def main():
    assert len(sys.argv) == 5
    solve(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])

if __name__ == '__main__':
    main()
