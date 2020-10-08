#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

from base import *
import numpy as np
from task_partitioner import TaskPartitioner

def setup_parser(subparsers):
    isect_parser = subparsers.add_parser("isect",
            help='Calculate the intersecting geometry from touch configurations')
    isect_parser.add_argument('task_id',
            help='Index of the batch to process',
            type=int)
    isect_parser.add_argument('geo_batch_size',
            help='Task granularity. Must divide tq_batch_size',
            type=int)
    isect_parser.add_argument('nsample',
            help="Number of samples in touch configuration files. Must be the same from 'sample_touch' command",
            type=int)
    isect_parser.add_argument('io_dir',
            help='Directory of the input samples and output geometries',
            type=str)

def run(args):
    task = ComputeTask(args)
    uw = task.get_uw()
    tunnel_v = task.get_tunnel_v()

    task_id = args.task_id
    geo_batch_size = args.geo_batch_size
    tq_batch_size = args.nsample
    io_dir = args.io_dir

    tp = TaskPartitioner(io_dir, geo_batch_size, tq_batch_size, tunnel_v=tunnel_v)
    '''
    Task partition
    |------------------TQ Batch for Conf. Q--------------------|
    |--Geo Batch--||--Geo Batch--||--Geo Batch--||--Geo Batch--|
    Hence run's task id = isect's task id / (Touch Batch Size/Geo Batch Size)
    '''
    batch_per_tq = tq_batch_size // geo_batch_size
    run_task_id, geo_batch_id = divmod(task_id, batch_per_tq)
    tq_batch_id, vert_id = divmod(run_task_id, len(tunnel_v))

    for tq, is_inf, vert_id, conf_id in tp.gen_touch_q(task_id):
        if is_inf:
            continue
        V, F = uw.intersecting_geometry(tq, True)
        np.savez_compressed(tp.get_isect_fn(vert_id, conf_id), V=V, F=F)
