#!/usr/bin/env python2
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())

import pyosr
import numpy as np
import math

import multiprocessing
from dualdata.template import *
import dualdata.full
import dualdata.tiny
import dualdata.two
import dualdata.four
import argparse

def create_cube(X, Y, Z):
    V = np.copy(UC_V)
    V[:,0] *= X
    V[:,1] *= Y
    V[:,2] *= Z
    return V, UC_F

def pos_to_tangent_offset(pos):
    d,m = divmod(pos, 2)
    tangent_offset = d * (HOLLOW_SQUARE_SIZE + STICK_WIDTH)
    if m == 0:
        tangent_offset += STICK_WIDTH
    else:
        tangent_offset += HOLLOW_SQUARE_SIZE + STICK_WIDTH - FILLISTER_LENGTH
    return tangent_offset

def pos_to_carve_origin(pos, tangent, normal, up, is_top=True):
    tangent_offset = pos_to_tangent_offset(pos)
    carve_origin = float(tangent_offset) * tangent
    carve_origin -= FILLISTER_MARGIN * normal
    if is_top:
        carve_origin += (STICK_HEIGHT - FILLISTER_DEPTH) * up # Carve the top
    else:
        carve_origin += -FILLISTER_MARGIN * up # Remove the margin
    return carve_origin

def build_stick_x(desc):
    is_debug = desc['debug']
    V, F = create_cube(desc['len'], STICK_WIDTH, STICK_HEIGHT)
    origin = np.array(list(desc['origin'])+[0.0], dtype=np.float64)
    # print(desc)
    # print(origin)
    # Template at origin
    VT, FT = create_cube(FILLISTER_LENGTH,
                         STICK_WIDTH+2*FILLISTER_MARGIN, # Both sides
                         FILLISTER_DEPTH+FILLISTER_MARGIN) # One side
    tangent = np.array([1,0,0], dtype=np.float64)
    normal = np.array([0,1,0], dtype=np.float64)
    up = np.array([0,0,1], dtype=np.float64)
    Mpos = [(V+origin, F)]
    # print('Mpos V {}'.format(V+origin))
    Mneg = []
    for pos in desc['up']:
        carve_origin = pos_to_carve_origin(pos, tangent, normal, up, is_top=True)
        VC = VT + carve_origin
        if is_debug:
            # print('Mneg V {}'.format(VC+origin))
            Mneg.append((VC+origin, FT))
        else:
            V, F = pyosr.mesh_bool(V, F, VC, FT, pyosr.MESH_BOOL_MINUS)
    # FIXME: Dedup the code
    for pos in desc['down']:
        carve_origin = pos_to_carve_origin(pos, tangent, normal, up, is_top=False)
        #print("carve_origin {}".format(carve_origin))
        VC = VT + carve_origin
        if is_debug:
            # print('V {}'.format(V+origin))
            Mneg.append((VC+origin, FT))
        else:
            V, F = pyosr.mesh_bool(V, F, VC, FT, pyosr.MESH_BOOL_MINUS)
    V += origin
    if is_debug:
        return Mpos, Mneg
    else:
        return V, F

def translate_from_x_to_y(V_x, F_x):
    V = np.copy(V_x)
    F = np.copy(F_x)
    # Swap X and Y
    V[:, 0] = V_x[:,1]
    V[:, 1] = V_x[:,0]
    # Flip the face orientation
    F[:, 0] = F_x[:,1]
    F[:, 1] = F_x[:,0]
    return V, F

def build_stick_y(desc):
    is_debug = desc['debug']
    desc_x = dict(desc)
    desc_x['origin'] = (desc['origin'][1], desc['origin'][0]) # Pretend it's arrange in X direction
    V_x, F_x = build_stick_x(desc_x)
    if is_debug:
        Mpos_x, Mneg_x = V_x, F_x
        Mpos = []
        Mneg = []
        for V,F in Mpos_x:
            Mpos.append(translate_from_x_to_y(V, F))
            print("Flip pos {}".format(Mpos[-1][1]))
        for V,F in Mneg_x:
            Mneg.append(translate_from_x_to_y(V, F))
            print("Flip neg {}".format(Mneg[-1][1]))
        return Mpos,Mneg
    return translate_from_x_to_y(V_x, F_x)

def merge_mesh(mesh_pair):
    mesh_0, mesh_1 = mesh_pair
    V0, F0 = mesh_0
    V1, F1 = mesh_1
    V,F = pyosr.mesh_bool(V0, F0, V1, F1, pyosr.MESH_BOOL_UNION)
    '''
    print('{} {} + {} {} = {} {}'.format(V0.shape, F0.shape,
                                         V1.shape, F1.shape,
                                         V.shape, F.shape))
    '''
    return V,F

def multi_save(meshes, fn, color):
    Vall = None
    Fall = None
    for V,F in meshes:
        Vall = V if Vall is None else np.concatenate((Vall, V), axis=0)
        # print("FT {}".format(F))
        F = np.copy(F) + Vall.shape[0] - V.shape[0]
        # print("F delta {}".format(Vall.shape[0] - V.shape[0]))
        Fall = F if Fall is None else np.concatenate((Fall, F), axis=0)
    empty = np.empty(shape=(0, 0))
    color = color.reshape((1,3))
    Vc = np.repeat(color, Vall.shape[0], axis=0)
    print(Vall.shape)
    print(Vc.shape)
    V = np.concatenate((Vall, Vc), axis=1)
    F = Fall
    pyosr.save_obj_1(V, F, fn)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pkg', help='Package name', nargs=None, type=str)
    parser.add_argument('--debug', help='Enable debugging output', action='store_true')
    args = parser.parse_args()
    pkg = getattr(dualdata, args.pkg)

    # p = multiprocessing.Pool(8)
    p = multiprocessing.Pool(1)
    for i,e in enumerate(pkg.STICKS_X_DESC):
        e['debug'] = args.debug
        e['bar_id'] = i
    for i,e in enumerate(pkg.STICKS_Y_DESC):
        e['debug'] = args.debug
        e['bar_id'] = i
    print(pkg.STICKS_X_DESC)
    print(pkg.STICKS_Y_DESC)
    '''
    meshes_x = []
    for desc in pkg.STICKS_X_DESC:
        meshes_x.append(build_stick_x(desc))
    '''
    meshes_x = p.map(build_stick_x, pkg.STICKS_X_DESC)
    # print('Mpos {}'.format(meshes_x[0][0]))
    # print('Mneg {}'.format(meshes_x[0][1]))
    # meshes_x = []
    '''
    meshes_y = []
    for desc in pkg.STICKS_Y_DESC:
        meshes_y.append(build_stick_y(desc))
    '''
    meshes_y = p.map(build_stick_y, pkg.STICKS_Y_DESC)
    # meshes_y = []
    meshes = meshes_x + meshes_y
    if args.debug:
        Mpos = []
        Mneg = []
        for i,e in enumerate(meshes):
            #print("{}: {}".format(i, e))
            pmesh, nmesh = e
            Mpos += pmesh
            Mneg += nmesh
        # print(Mneg)
        multi_save(Mpos, 'dual-pos.obj', color=np.array([0.0, 0.0, 0.75]))
        multi_save(Mneg, 'dual-neg.obj', color=np.array([0.75, 0.0, 0.0]))
        return
    # meshes = meshes[0:1] # Debug
    print(len(meshes))
    while len(meshes) > 1:
        meshes_next = p.map(merge_mesh, zip(meshes[0::2], meshes[1::2]))
        if len(meshes) % 2 == 1:
            meshes_next.append(meshes[-1])
        print(len(meshes_next))
        meshes = meshes_next
    V, F = meshes[0]
    print("Final V F {} {}".format(V.shape, F.shape))
    pyosr.save_obj_1(V,F,'dual.obj')

if __name__ == '__main__':
    main()
