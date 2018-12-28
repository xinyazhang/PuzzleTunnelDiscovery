#!/usr/bin/env python2

from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())

import pyosr
import numpy as np
import math

import multiprocessing

THICKNESS = 4.57
FILLISTER_DEPTH = THICKNESS - 3.76
FILLISTER_LENGTH = 3.58
FILLISTER_MARGIN = 0.02

UC_V = np.array(
[[1,0,0],
[1,1,1],
[0,1,1],
[1,0,1],
[1,1,0],
[0,0,1],
[0,0,0],
[0,1,0]], dtype=np.float64)

UC_F = np.array(
[
[4,0,6],
[4,6,7],
[1,2,5],
[1,5,3],
[4,1,3],
[4,3,0],
[0,3,5],
[0,5,6],
[6,5,2],
[6,2,7],
[1,4,7],
[1,7,2]], dtype=np.int32)

STICK_LENGTH = 61.60
HOLLOW_SQUARE_SIZE = 14.00
# Caliper gives different numbers for different bars.
# Hence we decided to assume the width is uniform
STICK_WIDTH = (STICK_LENGTH - 3 * HOLLOW_SQUARE_SIZE) / 4.0
STICK_HEIGHT = THICKNESS

# Describe the sticks
#   origin: the bottom left cornor
#   up: fillister locations on upside surface (0: beginning at the first hollow, 1: ending at the first hollow, ...)
#   down: fillister locations on downside surface
STICKS_X_DESC = [
        {
            'origin': (0,0),
            'up': [0,3],
            'down': [0]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0),
            'up': [2,3,4],
            'down': [1,2,3]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 2.0),
            'up': [0,2,3],
            'down': [2,4]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 3.0),
            'up': [],
            'down': [0,2,3,5]
        }
]

STICKS_Y_DESC = [
        {
            'origin': (0,0),
            'up': [],
            'down': [0,2,3],
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0, 0.0),
            'up': [0,2,3],
            'down': [2,5]
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 2.0, 0.0),
            'up': [2,3,5],
            'down': [3]
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 3.0, 0.0),
            'up': [],
            'down': [2,3,5]
        },
]

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
    pyosr.save_obj_1(V,F,'dual.obj')

if __name__ == '__main__':
    main()
