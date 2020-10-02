# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import numpy as np

'''
The copied framebuffer has different order from the texture file used by blender or OpenGL.
'''

'''
This is for uvrender
'''
def texture_to_file(arr):
    #return arr
    #return np.transpose(arr)
    '''
    np.flipud is verified by intersecting with self, and modifiy TriTriCopIsect to return identity
    '''
    return np.flipud(arr)
    #return np.flipud(np.transpose(arr))
    #return np.fliplr(arr)
    #return np.fliplr(np.transpose(arr))
    #return np.fliplr(np.flipud(arr))

'''
This is for atlas to prim(itive)
'''
def framebuffer_to_file(arr):
    #return np.transpose(np.fliplr(arr))
    #return np.transpose(np.flipud(arr))
    return np.flipud(arr)

'''
We are accessing the atlas ARR translated by framebuffer_to_file with indices
UV * ARR.shape, where uv is in [0,1]^2.
We can get the corresponding surface UV coordinates by calling uv_numpy_to_surface(uv)
'''
def uv_surface_to_numpy(uv):
    return np.array([1.0 - uv[1], uv[0]])

def uv_numpy_to_surface(uv):
    return np.array([uv[1], 1.0 - uv[0]])
