#!/usr/bin/env -S blender  --
# Note: the shebang line above requires coreutils 8.30 and above
# It is recommended to invoke this script with facade.py for portability

import sys
import bpy
import argparse
import math
import os
from math import pi as PI
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.linalg import expm
from scipy.linalg import norm as scipy_norm

def M(axis, theta):
    return expm(np.cross(np.eye(3), axis/scipy_norm(axis)*theta))

"""
Tired of handling the import. Can't python be more informal when dealing with import?
Just copy the code from SolVis.py
"""
def norm(vec):
    return np.linalg.norm(vec)

def normalized(vec):
    a = np.array(vec)
    return a / norm(vec)

def enable_cuda():
    """
    Enable CUDA
    """
    P=bpy.context.preferences
    prefs=P.addons['cycles'].preferences
    prefs.compute_device_type='CUDA'
    print(prefs.compute_device_type)
    print(prefs.get_devices())
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

def set_matrix_world(name, origin, lookat, up):
    origin = np.array(origin)
    lookat = np.array(lookat)
    up = np.array(up)
    lookdir = normalized(lookat - origin)
    up -= np.dot(up, lookdir) * lookdir
    mat = np.eye(4)
    mat[:3, 3] = origin
    mat[:3, 2] = -lookdir
    mat[:3, 1] = normalized(up)
    mat[:3, 0] = normalized(np.cross(lookdir, up))
    obj = bpy.data.objects[name]
    mw = obj.matrix_world
    for i in range(4):
        for j in range(4):
            mw[i][j] = mat[i, j]
    return origin, lookat, up, lookdir

def _add_key(rob, t, quat, frame):
    rob.rotation_mode = 'QUATERNION'
    rob.location.x = t[0]
    rob.location.y = t[1]
    rob.location.z = t[2]
    rob.rotation_quaternion.x = quat[0]
    rob.rotation_quaternion.y = quat[1]
    rob.rotation_quaternion.z = quat[2]
    rob.rotation_quaternion.w = quat[3]
    # rob.select = True
    rob.select_set(True)
    rob.keyframe_insert(data_path="location", frame=frame)
    rob.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    # rob.select = False
    rob.select_set(False)

def make_mat_emissive_texture(mat, texImage, energy=600):
    mat.use_nodes = True # First, otherwise node_tree won't be avaliable
    nodes = mat.node_tree.nodes
    glossy = nodes.new('ShaderNodeEmission')
    glossy.inputs[0].default_value = (0.0, 0.0, 0.0, 0.0)
    glossy.inputs[1].default_value = energy
    texNode = nodes.new('ShaderNodeTexImage')
    texNode.image = texImage
    colorRamp = nodes.new('ShaderNodeValToRGB')
    colorRamp.color_ramp.color_mode = 'HSL'
    colorRamp.color_ramp.hue_interpolation = 'FAR'
    colorRamp.color_ramp.elements[0].color = (0.0, 0.0, 1.0, 1.0)
    colorRamp.color_ramp.elements[0].position = 0.0
    colorRamp.color_ramp.elements[1].color = (1.0, 0.0, 0.0, 1.0)
    colorRamp.color_ramp.elements[1].position = 0.25
    # glossy.inputs[1].default_value = 90.0
    links = mat.node_tree.links
    out = nodes.get('Material Output')
    links.new(glossy.outputs[0], out.inputs[0])
    links.new(texNode.outputs[0], colorRamp.inputs[0])
    links.new(colorRamp.outputs[0], glossy.inputs[0])

def add_mat(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('geo', help='Geometry')
    p.add_argument('tex', help='Texture File (in NPZ)')
    p.add_argument('--saveas', help='Save the Blender file as', default='')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--flat', help='Flat shading', action='store_true')
    p.add_argument('--quit', help='Quit without running blender', action='store_true')
    p.add_argument('--camera_origin',
                   help='Origin of camera from SolVis, only used to calculate the camera distance',
                   type=float, nargs=3, default=None)
    p.add_argument('--camera_up', help='Up direction of the camera in the world, i.e. human that holds the camera.', type=float, nargs=3, default=None)
    p.add_argument('--camera_lookat', help='Point to Look At of camera', type=float, nargs=3, default=None)
    p.add_argument('--camera_rotation_axis', type=float, nargs=3, default=None)
    p.add_argument('--total_frames', help='Total number of frames to render', default=180)
    p.add_argument('--animation_single_frame', help='Render single frame of animation. Use in conjuction with --save_animation_dir. Override animation_end.', type=int, default=None)
    p.add_argument('--save_animation_dir', help='Save the Rendered animation sequence image to', default='')

    argv = sys.argv
    return p.parse_args(argv[argv.index("--") + 1:])

def main():
    args = parse_args()

    bpy.context.scene.render.engine = 'CYCLES'
    emission_mat = bpy.data.materials.new(name='Emission HSV')

    bpy.ops.import_scene.obj(filepath=args.geo, axis_forward='Y', axis_up='Z')
    geo = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    geo.name = 'Geo'
    if not args.flat:
        bpy.ops.object.shade_smooth()
    add_mat(geo, emission_mat)

    tex = np.load(args.tex)['ATEX']
    tex = np.flipud(tex)
    ma = np.max(tex)
    mi = np.min(tex)
    ntex = np.array(tex - mi, dtype=np.float32)
    if ma - mi > 0:
        print(f"ntex {ntex.shape} {ntex.dtype} ma {ma} mi {mi}")
        ntex /= ma - mi
    """
    # Binary texture for debugging
    ntex = np.copy(tex)
    ntex[np.nonzero(tex)] = 1.0
    """
    rgbtex = np.zeros(shape=list(tex.shape) + [4], dtype=np.float32)
    rgbtex[:,:,0] = ntex
    bpy.ops.image.new(name='InMemoryTexture',
                      width=tex.shape[0], height=tex.shape[1],
                      alpha=False,
                      float=True)
    texImage = outputImg = bpy.data.images["InMemoryTexture"]
    texImage.pixels = rgbtex.ravel()

    make_mat_emissive_texture(emission_mat, texImage, energy=1.0)

    # Clean up
    bpy.data.meshes.remove(bpy.data.meshes['Cube'])
    bpy.data.objects.remove(bpy.data.objects['Light'])

    cam_obj = bpy.data.objects['Camera']
    cam_obj.data.clip_end = 1000.0
    """
    Auto center
    """
    args.camera_lookat = np.mean(np.array(geo.bound_box), axis=0)

    camera_lookat = np.array(args.camera_lookat)
    # camera_rotation_axis = np.array(args.camera_rotation_axis)
    """
    Auto axis
    """
    min_bb = np.min(np.array(geo.bound_box), axis=0)
    max_bb = np.max(np.array(geo.bound_box), axis=0)
    A,B,C = diff_bb = max_bb - min_bb
    if B > A > C or C > A > B:
        camera_rotation_axis = np.array([1,0,0], dtype=np.float)
        if A > C:
            args.camera_up = np.array([0,0,1], dtype=np.float)
        else:
            args.camera_up = np.array([0,1,0], dtype=np.float)
    elif A > B > C or C > B > A:
        camera_rotation_axis = np.array([0,1,0], dtype=np.float)
        if B > C:
            args.camera_up = np.array([0,0,1], dtype=np.float)
        else:
            args.camera_up = np.array([1,0,0], dtype=np.float)
    else:
        assert A > C > B or B > C > A
        if C > B:
            args.camera_up = np.array([0,1,0], dtype=np.float)
        else:
            args.camera_up = np.array([1,0,0], dtype=np.float)
        camera_rotation_axis = np.array([0,0,1], dtype=np.float)
    init_lookvec = norm(np.array(args.camera_lookat) - np.array(args.camera_origin)) * np.array(args.camera_up)
    print(f"camera_rotation_axis {camera_rotation_axis}")
    print(f"camera_up {args.camera_up}")
    print(f"camera_lookat {args.camera_lookat}")
    print(f"init_lookvec {init_lookvec}")

    # if args.camera_lookat is not None and args.camera_origin and args.camera_rotation_axis:
    desiredFrames = args.total_frames
    bpy.context.scene.frame_end = desiredFrames - 1
    """
    cam_obj.rotation_mode = 'QUATERNION'
    for frame in range(desiredFrames):
        theta = frame * (PI * 2.0 / desiredFrames)
        lookvec = np.dot(M(camera_rotation_axis, theta), init_lookvec)
        t = lookvec + camera_lookat
        quat = Rotation.from_rotvec(camera_rotation_axis * theta).as_quat()
        _add_key(cam_obj, t, quat, frame)
    """
    set_matrix_world('Camera',
                     np.array(args.camera_lookat) + init_lookvec,
                     args.camera_lookat,
                     camera_rotation_axis)
    print(cam_obj.matrix_world)
    geo.rotation_mode = 'QUATERNION'
    for frame in range(desiredFrames):
        theta = frame * (PI * 2.0 / desiredFrames)
        lookvec = np.dot(M(camera_rotation_axis, theta), init_lookvec)
        t = -args.camera_lookat
        quat = Rotation.from_rotvec(camera_rotation_axis * theta).as_quat()
        _add_key(geo, t, quat, frame)

    bpy.context.scene.cycles.samples = 4
    bpy.context.scene.render.film_transparent = True
    if args.saveas and args.animation_single_frame is None:
        bpy.ops.wm.save_as_mainfile(filepath=args.saveas, check_existing=False)
    if args.save_animation_dir:
        os.makedirs(args.save_animation_dir, exist_ok=True)
        if args.cuda:
            enable_cuda()
        bpy.context.scene.render.filepath = args.save_animation_dir + '/'
        if args.animation_single_frame is not None:
            bpy.context.scene.frame_start = args.animation_single_frame
            bpy.context.scene.frame_end = args.animation_single_frame
        bpy.context.scene.render.use_overwrite = False
        bpy.ops.render.render(animation=True)
    if args.quit:
        bpy.ops.wm.quit_blender()

if __name__ == "__main__":
    main()
