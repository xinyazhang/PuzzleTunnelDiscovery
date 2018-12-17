import numpy as np

'''
The copied framebuffer has different order from the texture file used by blender or OpenGL.
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
