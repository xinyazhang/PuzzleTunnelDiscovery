#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

from base import *
from task_partitioner import TaskPartitioner, atlas_fn, atlastex_fn, TouchQGenerator, UVObjGenerator
import numpy as np
from scipy.misc import imsave

def setup_parser(subparsers):
    uvrender_parser = subparsers.add_parser("uvrender",
            help='Render and accumulate results from uvproj to numpy arrays and images.')
    uvrender_parser.add_argument('geo_type', choices=['rob', 'env'])
    uvrender_parser.add_argument('vert_id',
            help='Vertex ID with in the narrow tunnel vertices',
            type=int)
    uvrender_parser.add_argument('io_dir',
            help='Directory of the input projection data and rendered results',
            type=int)

def run(args):
    task = GpuTask(args)
    uw = task.get_uw()
    tunnel_v = task.get_tunnel_v()

    uw.avi = False
    geo_type = args.geo_type
    TYPE_TO_FLAG = {'rob' : uw.BARY_RENDERING_ROBOT,
                    'env' : uw.BARY_RENDERING_SCENE }
    geo_flag = TYPE_TO_FLAG[geo_type]

    vert_id = args.vert_id
    io_dir = args.io_dir
    iq = uw.translate_to_unit_state(tunnel_v[vert_id])
    afb = None
    afb_nw = None

    tq_gen = TouchQGenerator(in_dir=io_dir, vert_id=vert_id)
    obj_gen = UVObjGenerator(in_dir=io_dir, geo_type=geo_type, vert_id=vert_id)
    i = 0
    DEBUG_UVRENDER = False
    # TODO: Refactor this piece of xxxx
    for tq, is_inf in tq_gen:
        # print('tq {} is_inf {}'.format(tq, is_inf))
        IBV, IF = next(obj_gen) # Must, otherwise does not pair
        if is_inf:
            continue
        if IBV is None or IF is None:
            print('IBV {}'.format(None))
            continue
        print('{}: IBV {} IF {}'.format(i, IBV.shape, IF.shape))
        if DEBUG_UVRENDER:
            svg_fn = '{}.svg'.format(i)
            # Paint everything ...
            if i == 0 and geo_type == 'rob':
                V, F = uw.get_robot_geometry(tq, True)
                print("V {}\nF {}".format(V.shape, F.shape))
                IF, IBV = uw.intersecting_to_robot_surface(tq, True, V, F)
                print("IF {}\nIBV {}\n{}".format(IF.shape, IBV.shape, IBV[:5]))
                '''
                NPICK=3000
                IF = IF[:NPICK]
                IBV = IBV[:NPICK*3]
                '''
        else:
            svg_fn = ''
        uw.clear_barycentric(geo_flag)
        uw.add_barycentric(IF, IBV, geo_flag)
        if DEBUG_UVRENDER and i == 2:
            print("BaryF {}".format(IF))
            print("Bary {}".format(IBV))
        fb = uw.render_barycentric(geo_flag,
                                   np.array([ATLAS_RES, ATLAS_RES], dtype=np.int32),
                                   svg_fn=svg_fn)
        #np.clip(fb, 0, 1, out=fb) # Clip to binary
        nw = texture_format.texture_to_file(fb.astype(np.float32))
        distance = np.clip(pyosr.distance(tq, iq), 1e-4, None)
        w = nw * (1.0 / distance)
        if afb is None:
            afb = w
            afb_nw = nw
        else:
            afb += w
            afb_nw += nw
            np.clip(afb_nw, 0, 1.0, out=afb_nw) # afb_nw is supposed to be binary
        # Debugging code
        if DEBUG_UVRENDER:
            print('afb shape {}'.format(afb.shape))
            print('distance {}'.format(distance))
            rgb = np.zeros(list(afb.shape) + [3])
            rgb[...,1] = w
            imsave(atlastex_fn(io_dir, geo_type, vert_id, i), rgb)
            np.savez(atlas_fn(io_dir, geo_type, vert_id, i), w)
            print('NW NZ {}'.format(nw[np.nonzero(nw)]))
            V1, F1 = uw.get_robot_geometry(tq, True)
            pyosr.save_obj_1(V1, F1, '{}.obj'.format(i))
            V2, F2 = uw.get_scene_geometry(tq, True)
            pyosr.save_obj_1(V2, F2, '{}e.obj'.format(i))
            if i >= 16:
               return
        i+=1
    rgb = np.zeros(list(afb.shape) + [3])
    rgb[...,1] = afb
    # FIXME: Give savez an explicity array name
    imsave(atlastex_fn(io_dir, geo_type, vert_id, None), rgb)
    np.savez(atlas_fn(io_dir, geo_type, vert_id, None), afb)
    rgb[...,1] = afb_nw
    imsave(atlastex_fn(io_dir, geo_type, vert_id, None, nw=True), rgb)
    np.savez(atlas_fn(io_dir, geo_type, vert_id, None, nw=True), afb)
