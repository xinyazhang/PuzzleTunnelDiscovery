#!/usr/bin/env python2

from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())

import pyosr
import numpy as np
import math
import multiprocessing

import dual

from dual import FILLISTER_DEPTH, FILLISTER_LENGTH, FILLISTER_MARGIN, create_cube, pos_to_carve_origin, merge_mesh

STICK_WIDTH = dual.STICK_WIDTH
STICK_HEIGHT = dual.STICK_HEIGHT
HOLLOW_SQUARE_SIZE = dual.HOLLOW_SQUARE_SIZE
STICK_LENGTH = HOLLOW_SQUARE_SIZE + 2 * STICK_WIDTH

STICKS_X_DESC = [
        {
            'origin': (0,0),
            'up': [0],
            'down': []
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0),
            'up': [],
            'down': []
        },
]

STICKS_Y_DESC = [
        {
            'origin': (0,0),
            'up': [],
            'down': [],
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0, 0.0),
            'up': [],
            'down': []
        },
]

'''
FIXME: this is coped from dual.py
'''
def build_stick_x(desc):
    V, F = create_cube(STICK_LENGTH, STICK_WIDTH, STICK_HEIGHT)
    # Template at origin
    VT, FT = create_cube(FILLISTER_LENGTH,
                         STICK_WIDTH+2*FILLISTER_MARGIN, # Both sides
                         FILLISTER_DEPTH+FILLISTER_MARGIN) # One side
    tangent = np.array([1,0,0], dtype=np.float64)
    normal = np.array([0,1,0], dtype=np.float64)
    up = np.array([0,0,1], dtype=np.float64)
    for pos in desc['up']:
        carve_origin = pos_to_carve_origin(pos, tangent, normal, up, is_top=True)
        VC = VT + carve_origin
        V, F = pyosr.mesh_bool(V, F, VC, FT, pyosr.MESH_BOOL_MINUS)
    # FIXME: Dedup the code
    for pos in desc['down']:
        carve_origin = pos_to_carve_origin(pos, tangent, normal, up, is_top=False)
        #print("carve_origin {}".format(carve_origin))
        VC = VT + carve_origin
        V, F = pyosr.mesh_bool(V, F, VC, FT, pyosr.MESH_BOOL_MINUS)
    origin = np.array(list(desc['origin'])+[0.0], dtype=np.float64)
    V += origin
    return V, F

def build_stick_y(desc):
    desc_x = dict(desc)
    desc_x['origin'] = (desc['origin'][1], desc['origin'][0]) # Pretend it's arrange in X direction
    V_x, F_x = build_stick_x(desc_x)
    V = np.copy(V_x)
    F = np.copy(F_x)
    # Swap X and Y
    V[:, 0] = V_x[:,1]
    V[:, 1] = V_x[:,0]
    # Flip the face orientation
    F[:, 0] = F_x[:,1]
    F[:, 1] = F_x[:,0]
    return V, F

def main():
    p = multiprocessing.Pool(8)
    meshes_x = p.map(build_stick_x, STICKS_X_DESC)
    # meshes_x = []
    meshes_y = p.map(build_stick_y, STICKS_Y_DESC)
    # meshes_y = []
    meshes = meshes_x + meshes_y
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
    pyosr.save_obj_1(V,F,'dual_tiny.obj')

if __name__ == '__main__':
    main()
