#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

from base import *
from task_partitioner import TaskPartitioner


def setup_parser(subparsers):
    uvproj_parser = subparsers.add_parser("uvproj",
            help='Project intersection results to rob/env surface as vertex tuples and barycentric coordinates')
    uvproj_parser.add_argument('geo_type', choices=['rob', 'env'])
    uvproj_parser.add_argument('task_id',
            help='Index of the batch to process', type=int)
    uvproj_parser.add_argument('geo_batch_size',
            help='Task granularity. Must divide tq_batch_size',
            type=int)
    uvproj_parser.add_argument('nsample',
            help='Number of samples in touch configuration files. Must be the same from \'run\' command',
            type=int)
    uvproj_parser.add_argument('io_dir',
            help='Directory of the input geometries and output projection data',
            type=str)


def run(args):
    task = ComputeTask(args)
    uw = task.get_uw()
    geo_type = args.geo_type
    task_id = args.task_id
    gp_batch = args.geo_batch_size
    ntq_sample = args.nsample
    io_dir = args.io_dir
    tp = TaskPartitioner(io_dir, gp_batch, ntq_sample, tunnel_v=self._get_tunnel_v())
    for tq, is_inf, vert_id, conf_id in tp.gen_touch_q(task_id):
        if is_inf:
            continue
        fn = tp.get_isect_fn(vert_id, conf_id)
        d = np.load(fn+'.npz')
        V = d['V']
        F = d['F']
        if geo_type == 'rob':
            IF, IBV = uw.intersecting_to_robot_surface(tq, True, V, F)
        elif geo_type == 'env':
            IF, IBV = uw.intersecting_to_model_surface(tq, True, V, F)
        else:
            assert False
        fn2 = tp.get_uv_fn(geo_type, vert_id, conf_id)
        np.savez_compressed(fn2, V=IBV, F=IF)
