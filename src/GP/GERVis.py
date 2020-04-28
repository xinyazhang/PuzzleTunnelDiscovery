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

def unit_orth(v):
    prod = np.array([0,0,0])
    i = np.argmin(np.abs(v1))
    prod[i] = 1
    return v.cross(prod)

def rot_from_two_vecs(a, b): # Ported from Eigen::Quaternion<Scalar>::setFromTwoVectors
    v0 = normalized(a);
    v1 = normalized(b);
    c = v0.dot(v1);

    if np.allclose(c, 1):
        vec = [0,0,0]
        w = 1.0
    elif np.allclose(c, -1):
        vec = unit_orth(v0)
        w = 0;
    else:
        axis = np.cross(v0, v1)
        s = math.sqrt((1.0+c)*2.0)
        invs = 1.0/s;
        vec = axis * invs;
        w = s * 0.5;
    return Rotation.from_quat(list(vec) + [w])

def add_square(name, origin, euler_in_deg, size):
    euler = [e / 180.0 * PI for e in euler_in_deg]
    bpy.ops.mesh.primitive_plane_add(size=2.0 * size,
                                     align='WORLD',
                                     location=origin,
                                     rotation=euler)
    p = bpy.context.selected_objects[0]
    p.name = name
    return p

def make_mat_emissive(mat, val, energy=600.0):
    mat.use_nodes = True # First, otherwise node_tree won't be avaliable
    nodes = mat.node_tree.nodes
    glossy = nodes.new('ShaderNodeEmission')
    glossy.inputs[0].default_value = val
    glossy.inputs[1].default_value = energy
    # glossy.inputs[1].default_value = 90.0
    links = mat.node_tree.links
    out = nodes.get('Material Output')
    links.new(glossy.outputs[0], out.inputs[0])


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
    p.add_argument('--env', help='Env Geometry', required=True)
    p.add_argument('--rob', help='Rob Geometry', default='')
    p.add_argument('--opt_data', help='GER optimization log (in NPZ)', default='')
    p.add_argument('--key_data', help='GER Key Configurations (in NPZ)', default='')

    p.add_argument('--O', help='OMPL center of gemoetry', type=float, nargs=3)

    p.add_argument('--saveas', help='Save the Blender file as', default='')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--flat', help='Flat shading', action='store_true')
    p.add_argument('--quit', help='Quit without running blender', action='store_true')
    p.add_argument('--camera_origin',
                   help='Origin of camera from SolVis, only used to calculate the camera distance',
                   type=float, nargs=3, default=None)
    p.add_argument('--camera_up', help='Up direction of the camera in the world, i.e. human that holds the camera.', type=float, nargs=3, default=None)
    p.add_argument('--camera_lookat', help='Point to Look At of camera', type=float, nargs=3, default=None)
    p.add_argument('--camera_from_bottom', help='flip_camera w.r.t. the lookat and up direction. This also make a transparent floor', action='store_true')
    p.add_argument('--light_auto', help='Set the light configuration automatically', action='store_true')
    p.add_argument('--floor_origin', help='Center of the floor',
                                     type=float, nargs=3, default=[0, 0, -40])
    p.add_argument('--floor_euler', help='Rotate the floor with euler angle',
                                    type=float, nargs=3,
                                    default=[0,0,0])
    p.add_argument('--floor_size', help='Size of the floor, from the center to the edge',
                                   type=float, default=2500)
    p.add_argument('--total_frames', help='Total number of frames to render', default=180)
    p.add_argument('--remove_vn', help='Remove VN from the mesh', choices=['env', 'rob'], nargs='*', default=[])
    p.add_argument('--enable_autosmooth', help='Enable autosmooth (if not enabled by default)', choices=['env', 'rob'], nargs='*', default=[])
    p.add_argument('--resolution_x', type=int, default=1920)
    p.add_argument('--resolution_y', type=int, default=1080)
    p.add_argument('--animation_single_frame', help='Render single frame of animation. Use in conjuction with --save_animation_dir. Override animation_end.', type=int, default=None)
    p.add_argument('--save_animation_dir', help='Save the Rendered animation sequence image to', default='')
    p.add_argument('--marker_scale', help='Size of the marker', type=float, default=1.0)

    argv = sys.argv
    return p.parse_args(argv[argv.index("--") + 1:])

class GeoWithFeat(object):
    def __init__(self, name, fn, mat, cone_mat, sp_mat, args):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_scene.obj(filepath=fn, axis_forward='Y', axis_up='Z')
        geo = bpy.context.selected_objects[0]
        # bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
        geo.name = name
        add_mat(geo, mat)
        if not args.flat:
            bpy.ops.object.shade_smooth()
        if name in args.remove_vn:
            print("REMOVING VN")
            bpy.context.view_layer.objects.active = geo
            print(bpy.ops.mesh.customdata_custom_splitnormals_clear())
        if name in args.enable_autosmooth:
            geo.data.use_auto_smooth = True

        self._geo = geo
        self._mat = mat
        self._name = name
        self._cone1 = self._create_cone(f'{name} Point V', cone_mat, args.marker_scale)
        self._cone2 = self._create_cone(f'{name} Point W', cone_mat, args.marker_scale)
        if sp_mat is None:
            self._sp = None
        else:
            self._sp = self._create_sphere(f'{name} Center', sp_mat, args.marker_scale)

    def _create_cone(self, name, mat, scale):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.mesh.primitive_cone_add()
        cone = bpy.context.selected_objects[0]
        cone.name = name
        cone.scale = [scale * 1.0] * 3
        add_mat(cone, mat)
        return cone

    def _create_sphere(self, name, mat, scale):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.mesh.primitive_uv_sphere_add()
        sp = bpy.context.selected_objects[0]
        sp.name = name
        sp.scale = [scale * 0.5] * 3
        add_mat(sp, mat)
        return sp

    def add_animation(self, V, W):
        Z = np.array([0,0,1])
        bpy.ops.object.select_all(action='DESELECT')
        for i in range(V.shape[0]):
            frame = i+1
            di = normalized(V[i] - W[i])
            rot1 = rot_from_two_vecs(Z, -di)
            rot2 = rot_from_two_vecs(Z,  di)
            # print(f'Exp {di} Get {rot2.apply(Z)}')
            _add_key(self._cone1, V[i], rot1.as_quat(), frame)
            _add_key(self._cone2, W[i], rot2.as_quat(), frame)
            if self._sp:
                _add_key(self._sp, 0.5 * (V[i]+W[i]), [0,0,0,1], frame)

    def add_keys(self, qs, pts):
        V, W = pts[:,:3], pts[:,3:6]
        if qs is None:
            self.add_animation(V, W)
            return
        Z = np.array([0,0,1])
        for i, q in enumerate(qs):
            frame = i + 1
            t = q[:3]
            quat = q[3:7]
            R = Rotation.from_quat(quat)
            v = R.apply(V[i]) + t
            w = R.apply(W[i]) + t
            di = normalized(v - w)
            rot1 = rot_from_two_vecs(Z, -di)
            rot2 = rot_from_two_vecs(Z,  di)
            _add_key(self._cone1, v, rot1.as_quat(), frame)
            _add_key(self._cone2, w, rot2.as_quat(), frame)
            if self._sp:
                _add_key(self._sp, 0.5 * (v+w), [0,0,0,1], frame)
            _add_key(self._geo, t, quat, frame)


def main():
    args = parse_args()
    assert args.opt_data or args.key_data, 'Either --opt_data or --key_data shall present'

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.meshes.remove(bpy.data.meshes['Cube'])
    bpy.data.objects.remove(bpy.data.objects['Light'])

    def make_mat_glossy(mat, val, transparent=None):
        mat.use_nodes = True # First, otherwise node_tree won't be avaliable
        nodes = mat.node_tree.nodes
        glossy = nodes.new('ShaderNodeBsdfGlossy')
        glossy.inputs[0].default_value = val
        # glossy.inputs[1].default_value = 0.618
        glossy.inputs[1].default_value = 0.316
        links = mat.node_tree.links
        out = nodes.get('Material Output')
        if transparent:
            trans = nodes.new('ShaderNodeBsdfTransparent')
            # trans.inputs[0].default_value = [1.0, 0.0, 0.0, 0.3]
            trans.inputs[0].default_value = [1.0, 1.0, 1.0, 0.10]
            # trans.inputs[0].default_value[3] = 0.25

            mix = nodes.new('ShaderNodeMixShader')
            links.new(trans.outputs[0], mix.inputs[1])
            links.new(glossy.outputs[0], mix.inputs[2])
            links.new(mix.outputs[0], out.inputs[0])
        else:
            links.new(glossy.outputs[0], out.inputs[0])
        if args.flat:
            env = nodes.new('ShaderNodeNewGeometry')
            links.new(env.outputs[3], glossy.inputs[2])
            print("Applying flat shading")

    red_mat = bpy.data.materials.new(name='Material Red')
    make_mat_glossy(red_mat, [1.0, 0.0, 0.0, 1.0])
    cyan_mat = bpy.data.materials.new(name='Material Cyan')
    make_mat_glossy(cyan_mat, [0.0, 0.4, 0.4, 1.0])
    green_mat = bpy.data.materials.new(name='Material Green')
    make_mat_glossy(green_mat, [0.0, 1.0, 0.0, 1.0])
    gold_mat = bpy.data.materials.new(name='Material Gold')
    make_mat_glossy(gold_mat, [0.777, 0.8, 0.0, 1.0])

    emission_mat = bpy.data.materials.new(name='Emission White')
    make_mat_emissive(emission_mat, [1.0, 1.0, 1.0, 1.0], energy=600)

    if args.opt_data:
        geo = GeoWithFeat('Geo', args.env,
                           mat=red_mat, cone_mat=green_mat, sp_mat=gold_mat,
                           args=args)
        d = np.load(args.opt_data)
        V = d['V']
        W = d['W']
        print(f'Load V {V.shape} W {W.shape}')
        geo.add_animation(V=V, W=W)
        bpy.context.scene.frame_end = V.shape[0]
    else:
        env = GeoWithFeat('env', args.env,
                           mat=cyan_mat, cone_mat=green_mat, sp_mat=None,
                           args=args)
        rob = GeoWithFeat('rob', args.rob,
                           mat=red_mat, cone_mat=green_mat, sp_mat=gold_mat,
                           args=args)
        d = np.load(args.key_data)
        rob.add_keys(qs=d['KEYQ_AMBIENT'], pts=d['ROB_KEYS'])
        env.add_keys(qs=None, pts=d['ENV_KEYS'])
        bpy.context.scene.frame_end = d['KEYQ_AMBIENT'].shape[0]

    # floor = add_square('Floor', args.floor_origin, args.floor_euler, args.floor_size)

    if args.camera_origin is not None and args.camera_lookat is not None and args.camera_up is not None:
        camera_origin, camera_lookat, camera_up, camera_lookdir = set_matrix_world('Camera', args.camera_origin, args.camera_lookat, args.camera_up)
        cam_obj = bpy.data.objects['Camera']
        # cam_obj.data.clip_end = 2.0 * norm(camera_origin - camera_lookat)
        cam_obj.data.clip_end = 1000.0
        camera_dist = norm(camera_lookat - camera_origin)
        alight = add_square('Area light', [0,0,0],
                            [45, 0, 0], 0.2 * camera_dist)
        add_mat(alight, emission_mat)
        if args.light_auto:
            camera_right = normalized(np.cross(camera_lookdir, camera_up))
            world_up = normalized(np.array(args.camera_up))
            set_matrix_world('Area light',
                             # camera_lookat + camera_dist * camera_up - camera_dist * camera_right,
                             camera_origin - camera_dist * camera_right,
                             camera_lookat,
                             args.camera_up)

            sun_light = bpy.ops.object.light_add(type='SUN')
            set_matrix_world('Sun',
                             # camera_lookat + camera_dist * camera_up - camera_dist * camera_right,
                             camera_lookat + 1.0 * camera_dist * camera_up,
                             camera_lookat,
                             args.camera_up)
            bpy.data.lights["Sun"].energy = camera_dist * 0.05

        elif args.light_panel_origin is not None and args.light_panel_lookat is not None and args.light_panel_up is not None:
            set_matrix_world('Area light',
                             args.light_panel_origin,
                             args.light_panel_lookat,
                             args.light_panel_up)
        if args.camera_from_bottom:
            world_up = normalized(np.array(args.camera_up))
            height = np.dot(world_up, np.array(args.camera_origin) - np.array(args.camera_lookat))
            print(f'height {height}')
            rev_origin = np.array(args.camera_origin) - 2 * height * world_up
            print(f'old origin {args.camera_origin} new origin {rev_origin}')
            set_matrix_world('Camera', rev_origin, args.camera_lookat, args.camera_up)
            # floor.cycles_visibility.camera = False
            bpy.data.objects.remove(bpy.data.objects['Floor'])

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1,1,1,0)

    scene = bpy.data.scenes['Scene']
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    bpy.context.scene.cycles.samples = 128
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
