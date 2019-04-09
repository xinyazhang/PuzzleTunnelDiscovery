#!/usr/bin/env python3

from base import *
import numpy as np
from task_partitioner import TaskPartitioner

def setup_parser(subparsers):
    sample_touch_parser = subparsers.add_parser("sample_touch",
            help='Sample #task_size touch configurations from Tunnel Vertex (#task_id mod (total number of tunnel vertices))')
    sample_touch_parser.add_argument('task_id',
            help='Index of the batch to process',
            type=int)
    sample_touch_parser.add_argument('nsample',
            help='Number of samples',
            type=int)
    sample_touch_parser.add_argument('out_dir',
            help='Size of the batch to process',
            type=str)
    show_parser = subparsers.add_parser("show", help='Show the number of tunnel vertices')


def calc_touch(uw, vertex, nsample):
    q0 = uw.translate_to_unit_state(vertex)
    N_RET = 5
    ret_lists = [[] for i in range(N_RET)]
    for i in range(nsample):
        tr = uw_random.random_on_sphere(1.0)
        aa = uw_random.random_within_sphere(2 * math.pi)
        to = pyosr.apply(q0, tr, aa)
        tups = uw.transit_state_to_with_contact(q0, to, STEPPING_FOR_TOUCH)
        for i in range(N_RET):
            ret_lists[i].append(tups[i])
    rets = [np.array(ret_lists[i]) for i in range(N_RET)]
    for i in range(N_RET):
        print("{} shape {}".format(i, rets[i].shape))
    return rets


def run(args):
    task_id = args.task_id
    nsample = args.nsample
    out_dir = args.out_dir
    task = ComputeTask(args)
    uw = task.get_uw()
    tp = TaskPartitioner(out_dir, None, nsample, tunnel_v = task._get_tunnel_v())

    vertex = tp.get_tunnel_vertex(task_id)
    out_fn = tp.get_tq_fn(task_id)

    free_vertices, touch_vertices, to_inf, free_tau, touch_tau = calc_touch(uw, vertex, nsample)
    np.savez(out_fn,
             FROM_V=np.repeat(np.array([vertex]), nsample, axis=0),
             FREE_V=free_vertices,
             TOUCH_V=touch_vertices,
             IS_INF=to_inf,
             FREE_TAU=free_tau,
             TOUCH_TAU=touch_tau)
