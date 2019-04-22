import numpy as np
import uw_random
import pyosr
import math

def sample_one_touch(uw, q0, stepping):
    tr = uw_random.random_on_sphere(1.0)
    aa = uw_random.random_within_sphere(2 * math.pi)
    to = pyosr.apply(q0, tr, aa)
    return uw.transit_state_to_with_contact(q0, to, stepping)

def calc_touch(uw, vertex, batch_size, stepping):
    q0 = uw.translate_to_unit_state(vertex)
    N_RET = 5
    ret_lists = [[] for i in range(N_RET)]
    for i in range(batch_size):
        tups = sample_one_touch(uw, q0, stepping)
        for i in range(N_RET):
            ret_lists[i].append(tups[i])
    rets = [np.array(ret_lists[i]) for i in range(N_RET)]
    #for i in range(N_RET):
        #print("{} shape {}".format(i, rets[i].shape))
    return rets
