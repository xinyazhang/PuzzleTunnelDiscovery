#!/usr/bin/env -S blender -P
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

def norm(vec):
    return np.linalg.norm(vec)

def normalized(vec):
    a = np.array(vec)
    return a / norm(vec)

#ui stuff... can ignore this

class SimplePanel(bpy.types.Panel):
    bl_id_name = "object.simple_operator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    # bl_category = "Motion planning"
    bl_label = "Simple Panel"

    def draw(self, context):
        layout = self.layout
        col = layout.row(align=True)

        col.label(text="Index Selection")

        col2 = layout.row(align=True)
        col2.operator("w.prev")
        col2.operator("w.increase")

        col3 = layout.row(align=True)
        col3.prop(context.scene, "i", text = "index", slider = True)

class Next(bpy.types.Operator):
    bl_idname = "w.increase"
    bl_label = "next"

    def execute(self, context):
        context.scene.i = increase(context.scene.i, listSize)
        select(frameSamples, context.scene.i)
        print(context.scene.i)
        return {'FINISHED'}

class Prev(bpy.types.Operator):
    bl_idname = "w.prev"
    bl_label = "previous"

    def execute(self, context):
        context.scene.i = decrease(context.scene.i, listSize)
        print(context.scene.i)
        select(frameSamples, context.scene.i)
        return {'FINISHED'}

#some quaternion operations... probably not needed
def eulerangle(q):
    t = q.components
    a1 = math.atan2(t[0]*t[1] + t[2]*t[3], 1-2*(t[1]**2+t[2]**2))
    a2 = math.asin(2*(t[0]*t[2]-t[3]*t[1]))
    a3 = math.atan2(2*(t[0]*t[3]+t[1]*t[2]), 1-2*(t[2]**2+t[3]**2))
    return [a1,a2,a3]

def distance(q1, q2, w1, w2):
    dq = abs(np.dot(q1[3:7], q2[3:7]))
    #print(f'dq: {dq}')
    dq = np.arccos(np.clip(dq, a_min=-1.0, a_max=1.0))
    return w1 * ((q1[0] - q2[0])**2 + (q1[1] - q2[1])**2 + (q1[2] - q2[2])**2 )**(1/2) + w2 * dq

#needed for blender's interpolation
def anim_distance(q1,q2):
    return (q1[0]-q2[0])**2 + (q1[1] - q2[1])**2 + (q1[2] - q2[2])**2 + (q1[3] - q2[3])**2

def select(li, i, objkey = "Rob"):
    '''sets coordinates of obj to list[i] '''
    obj = bpy.data.objects[objkey]
    obj.rotation_mode = 'QUATERNION'

    obj.location.x = li[i][0]
    obj.location.y = li[i][1]
    obj.location.z = li[i][2]
    obj.rotation_quaternion.w = li[i][3]
    obj.rotation_quaternion.x = li[i][4]
    obj.rotation_quaternion.y = li[i][5]
    obj.rotation_quaternion.z = li[i][6]
    print(f'{objkey} quat {obj.rotation_quaternion}')

def _mix(tau, key_0, key_1):
    t = (1.0 - tau) * key_0[0] + tau * key_1[0]
    rots = Rotation.from_quat([key_0[1].as_quat(), key_1[1].as_quat()])
    slerp = Slerp([0.0, 1.0], rots)
    r, = slerp([tau])
    return (t, r)

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

def displace(initCoords, transCoords, obj):
    """applies rigid body transform to obj, with initial position (x,y,z) initCoords"""
    r = Rotation.from_quat(transCoords[3:7])
    t1 = transCoords[0:3]
    t0 = initCoords[0:3]
    res = t1 + r.apply(t0)
    bpy.data.objects[obj].location.x = res[0]
    bpy.data.objects[obj].location.y = res[1]
    bpy.data.objects[obj].location.z = res[2]

def decrease (i, n):
    if (i>0): return i-1
    return n-1

def increase (i,m):
    if (i==m-1): return 0
    return i+1

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

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('env', help='ENV geometry')
    p.add_argument('rob', help='ROB geometry')
    p.add_argument('qpath', help='Vanilla configuration path')
    p.add_argument('--rendering_styles', help="Rendering styles", choices=['Physical', 'ShadowOnWhite'], default='ShadowOnWhite')
    p.add_argument('--total_frames', help='Total number of frames to render', default=1440)
    p.add_argument('--O', help='OMPL center of gemoetry', type=float, nargs=3, required=True)
    p.add_argument('--camera_origin', help='Origin of camera', type=float, nargs=3, default=None)
    p.add_argument('--camera_lookat', help='Point to Look At of camera', type=float, nargs=3, default=None)
    p.add_argument('--camera_up', help='Up direction of the camera in the world, i.e. human that holds the camera.', type=float, nargs=3, default=None)
    p.add_argument('--light_auto', help='Set the light configuration automatically', action='store_true')
    p.add_argument('--light_panel_origin', help='Origin of light_panel', type=float, nargs=3, default=None)
    p.add_argument('--light_panel_lookat', help='Point to Look At of light_panel', type=float, nargs=3, default=None)
    p.add_argument('--light_panel_up', help='Up direction of light_panel', type=float, nargs=3, default=None)
    p.add_argument('--floor_origin', help='Center of the floor',
                                     type=float, nargs=3, default=[0, 0, -40])
    p.add_argument('--floor_euler', help='Rotate the floor with euler angle',
                                    type=float, nargs=3,
                                    default=[0,0,0])
    p.add_argument('--floor_size', help='Size of the floor, from the center to the edge',
                                   type=float, default=2500)
    p.add_argument('--flat_env', help='Flat shading', action='store_true')
    p.add_argument('--image_frame', help='Image Frame', type=int, default=0)
    p.add_argument('--animation_end', help='End frame of animation', type=int, default=-1)
    p.add_argument('--animation_floor_origin', help='Center of the floor when rendering the animation',
                                     type=float, nargs=3, default=None)
    p.add_argument('--saveas', help='Save the Blender file as', default='')
    p.add_argument('--save_image', help='Save the Rendered image as', default='')
    p.add_argument('--save_animation_dir', help='Save the Rendered animation sequence image to', default='')
    p.add_argument('--enable_animation_preview', action='store_true')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--preview', action='store_true')
    p.add_argument('--quit', help='Quit without running blender', action='store_true')
    argv = sys.argv
    return p.parse_args(argv[argv.index("--") + 1:])

def main():
    args = parse_args()

    bpy.utils.register_class(SimplePanel)
    bpy.utils.register_class(Prev)
    bpy.utils.register_class(Next)
    bpy.context.scene.render.engine = 'CYCLES'

    def make_mat_glossy(mat, val):
        mat.use_nodes = True # First, otherwise node_tree won't be avaliable
        nodes = mat.node_tree.nodes
        glossy = nodes.new('ShaderNodeBsdfGlossy')
        glossy.inputs[0].default_value = val
        # glossy.inputs[1].default_value = 0.618
        glossy.inputs[1].default_value = 0.316
        links = mat.node_tree.links
        out = nodes.get('Material Output')
        links.new(glossy.outputs[0], out.inputs[0])
        if args.flat_env:
            geo = nodes.new('ShaderNodeNewGeometry')
            links.new(geo.outputs[3], glossy.inputs[2])
            print("Applying flat shading")
    cyan_mat = bpy.data.materials.new(name='Material Cyan')
    make_mat_glossy(cyan_mat, [0.0, 0.748, 0.8, 1.0])

    green_mat = bpy.data.materials.new(name='Material Pure Green')
    make_mat_glossy(green_mat, [0.0, 1.0, 0.0, 1.0])

    gold_mat = bpy.data.materials.new(name='Material Gold')
    make_mat_glossy(gold_mat, [0.777, 0.8, 0.0, 1.0])

    red_mat = bpy.data.materials.new(name='Material Red')
    make_mat_glossy(red_mat, [1.0, 0.0, 0.0, 1.0])

    def make_mat_diffuse(mat, val):
        mat.use_nodes = True # First, otherwise node_tree won't be avaliable
        nodes = mat.node_tree.nodes
        glossy = nodes.new('ShaderNodeBsdfDiffuse')
        glossy.inputs[0].default_value = val
        glossy.inputs[1].default_value = 1.0
        links = mat.node_tree.links
        out = nodes.get('Material Output')
        links.new(glossy.outputs[0], out.inputs[0])
        if args.flat_env:
            geo = nodes.new('ShaderNodeNewGeometry')
            links.new(geo.outputs[3], glossy.inputs[2])
            print("Applying flat shading")
    diffuse_mat = bpy.data.materials.new(name='Diffuse White')
    # make_mat_diffuse(diffuse_mat, [1.0, 1.0, 1.0, 1.0])
    make_mat_diffuse(diffuse_mat, [0.01, 0.01, 0.01, 1.0])

    def make_mat_ao(mat, val):
        mat.use_nodes = True # First, otherwise node_tree won't be avaliable
        nodes = mat.node_tree.nodes
        # diffuse = nodes.new('ShaderNodeBsdfDiffuse')
        diffuse = nodes.new('ShaderNodeBsdfGlossy')
        diffuse.inputs[0].default_value = val
        # diffuse.inputs[1].default_value = 1.0
        diffuse.inputs[1].default_value = 0.316
        ao = nodes.new('ShaderNodeAmbientOcclusion')
        ao.inputs[0].default_value = val

        links = mat.node_tree.links
        out = nodes.get('Material Output')
        links.new(diffuse.outputs[0], out.inputs[0])
        links.new(ao.outputs[0], diffuse.inputs[0])
        geo = nodes.new('ShaderNodeNewGeometry')
        if args.flat_env:
            links.new(geo.outputs[3], diffuse.inputs[2])
            links.new(geo.outputs[3], ao.inputs[2])
            print("Applying flat shading")
        else:
            links.new(geo.outputs[1], ao.inputs[2])
            links.new(geo.outputs[1], diffuse.inputs[2])
    ao_green_mat = bpy.data.materials.new(name='Material Green with AO')
    make_mat_ao(ao_green_mat, [0.0, 0.4, 0.4, 1.0])

    def make_mat_emissive(mat, val):
        mat.use_nodes = True # First, otherwise node_tree won't be avaliable
        nodes = mat.node_tree.nodes
        glossy = nodes.new('ShaderNodeEmission')
        glossy.inputs[0].default_value = val
        glossy.inputs[1].default_value = 600.0
        # glossy.inputs[1].default_value = 90.0
        links = mat.node_tree.links
        out = nodes.get('Material Output')
        links.new(glossy.outputs[0], out.inputs[0])
    emission_mat = bpy.data.materials.new(name='Emission White')
    make_mat_emissive(emission_mat, [1.0, 1.0, 1.0, 1.0])


    def add_mat(obj, mat):
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    def add_square(name, origin, euler_in_deg, size):
        euler = [e / 180.0 * PI for e in euler_in_deg]
        bpy.ops.mesh.primitive_plane_add(size=2.0 * size,
                                         align='WORLD',
                                         location=origin,
                                         rotation=euler)
        p = bpy.context.selected_objects[0]
        p.name = name
        return p
    floor_origin = args.floor_origin
    if args.save_animation_dir and args.animation_floor_origin is not None:
        floor_origin = args.animation_floor_origin
    if args.enable_animation_preview and args.animation_floor_origin is not None:
        floor_origin = args.animation_floor_origin
    floor = add_square('Floor', floor_origin, args.floor_euler, args.floor_size)

    floor.cycles_visibility.glossy = False
    if args.rendering_styles == 'ShadowOnWhite':
        floor.cycles.is_shadow_catcher = True
    else:
        add_mat(floor, diffuse_mat)

    # Load meshes
    bpy.ops.import_scene.obj(filepath=args.env, axis_forward='Y', axis_up='Z')
    env = bpy.context.selected_objects[0]
    env.name = 'Env'
    # add_mat(env, green_mat)
    add_mat(env, ao_green_mat)
    # IMPORTANT: For some reason we don't know, Mobius needs this explicitly.
    if not args.flat_env:
        bpy.ops.object.shade_smooth()
    """
    if args.flat_env:
        bpy.ops.object.shade_flat()
    """
    bpy.ops.import_scene.obj(filepath=args.rob, axis_forward='Y', axis_up='Z')
    rob = bpy.context.selected_objects[0]
    rob.name = 'Rob'
    if not args.flat_env:
        bpy.ops.object.shade_smooth()
    """
    if args.flat_env:
        bpy.ops.object.shade_flat()
    """
    add_mat(rob, red_mat)
    '''
    bpy.ops.mesh.primitive_uv_sphere_add()
    bpy.context.selected_objects[0].name = 'Witness'
    '''
    bpy.data.meshes.remove(bpy.data.meshes['Cube'])
    bpy.data.objects.remove(bpy.data.objects['Light'])
    # print(bpy.data.objects['Camera'].matrix_world)

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
            '''
            set_matrix_world('Area light',
                             camera_origin + camera_right * camera_dist + 1.0 * camera_dist * normalized(np.array(args.camera_up)),
                             camera_lookat,
                             args.camera_up)
            '''
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

            '''
            bpy.ops.object.light_add(type='POINT',
                                     location = camera_origin + 2.0 * (camera_origin - camera_lookat)
            )
            point = bpy.data.lights["Point"]
            point.energy = camera_dist * 10.0
            '''
        elif args.light_panel_origin is not None and args.light_panel_lookat is not None and args.light_panel_up is not None:
            set_matrix_world('Area light',
                             args.light_panel_origin,
                             args.light_panel_lookat,
                             args.light_panel_up)
    print(bpy.data.objects)
    print(bpy.data.objects['Camera'].matrix_world)


    # matrix detailing path of the object. framesamples[i] is coordinates of the robot at step i
    frameSamples = []
    key_confs = []
    # distance travelled per step, using weighted distance. Determines keyrame placement
    distances = [0]
    rotationWeight = 50     #weight to rotational component of distance
    totalDist = 0           #total distance of path.
    currSelection = 0

    # input file location, change as required
    vanilla_path = np.loadtxt(args.qpath)

    prevQ = []
    O = np.array(args.O)
    for el in vanilla_path:
            el = [float(el) for el in el]
            T = el[:3]
            if not prevQ:
                prevQ = [el[6], el[3], el[4], el[5]]
            #blender has weird interpolation with respect to quaternions. This fixesit
            d1 = anim_distance(prevQ, [el[6], el[3], el[4], el[5]])
            d2 = anim_distance(prevQ, [-el[6], -el[3], -el[4], -el[5]])
            if (d1 <= d2):
                Q = [el[6], el[3], el[4], el[5]]
            else:
                Q = [-el[6], -el[3], -el[4], -el[5]]
            res = T
            res.extend(Q)
            prevQ = list(Q)
            if frameSamples:
                d = distance(frameSamples[-1], res, 1, rotationWeight)
                distances.append(distances[-1] + d)
                totalDist += d
            frameSamples.append(res) # w-first
            R = Rotation.from_quat(el[3:7]) # w-last
            # Translate to OMPL configuration
            key_confs.append((np.array(el[:3]) + R.apply(O), R))
    listSize = len(frameSamples)

    def i_changed(self, context):
        index = context.scene.i
        select(frameSamples, index)

    """
    # This copies the robot and puts in various positions. Make sure robot is transparent enough.
    # Note: should be done before inserting key frames.
    scn = bpy.context.scene
    src_obj = bpy.data.objects["Rob"]
    for i in range(1, len(frameSamples), 4):
        new_obj = src_obj.copy()
        new_obj.data = src_obj.data.copy()
        new_obj.name = f"Rob_{i}"
        scn.objects.link(new_obj)
        select(frameSamples, i, objkey=new_obj.name)
    """


    bpy.types.Scene.i = bpy.props.IntProperty(
                                        default=3,
                                        min=0,
                                        max=listSize-1,
                                        update = i_changed
                                        )

    desiredFrames = args.total_frames

    # bpy.context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'
    bpy.types.PreferencesEdit.keyframe_new_interpolation_type = 'LINEAR'

#initial coordinates of point on robot when robot's position is identity
    witnessCoords = np.array([40,41,-2])

    '''
    # keyframe insertion
    # Note this is buggy due to C-space translation
    for ind in range(listSize):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.scene.i = ind
        select(frameSamples, ind)    #move robot
        bpy.data.objects["Rob"].select = True
        bpy.data.objects["Rob"].keyframe_insert(data_path="location",frame = desiredFrames/totalDist*distances[ind]+1)
        bpy.data.objects["Rob"].keyframe_insert(data_path="rotation_quaternion",
                                                frame=desiredFrames/totalDist*distances[ind]+1)
        displace(witnessCoords, frameSamples[ind], "Witness")   #move witness
        bpy.data.objects["Rob"].select = False
        bpy.data.objects["Witness"].select = True
        bpy.data.objects["Witness"].keyframe_insert(data_path="location",frame = desiredFrames/totalDist*distances[ind]+1)
        bpy.data.objects["Witness"].keyframe_insert(data_path="rotation_quaternion",
                                                    frame=desiredFrames/totalDist*distances[ind]+1)
    '''
    # All-frame insertion
    distance_per_frame = 1.0 / float(desiredFrames) * totalDist
    print(args.O)
    DEBUG = False
    rob = bpy.data.objects["Rob"]
    if DEBUG:
        key_0 = key_confs[44]
        key_1 = key_confs[45]
        t,r = _mix(0.0, key_0, key_1)
        t = t - r.apply(O) # Translate back to vanilla
        quat = r.as_quat() # Note scipy.spatial.transform.Rotation uses w-last
        _add_key(rob, t, quat, 1)
        t,r = _mix(0.7507349475305162, key_0, key_1)
        t = t - r.apply(O) # Translate back to vanilla
        quat = r.as_quat()
        print(f"intermediate key {t} {quat}")
        _add_key(rob, t, quat, 2)
        t,r = _mix(1.0, key_0, key_1)
        t = t - r.apply(O) # Translate back to vanilla
        quat = r.as_quat()
        _add_key(rob, t, quat, 3)
    else:
        for frame in range(desiredFrames):
            d = frame * distance_per_frame
            index_1 = np.argmax(distances > d)
            # print(f'index_1 {index_1}')
            index_0 = index_1 - 1
            d_0 = distances[index_0]
            d_1 = distances[index_1]
            key_0 = key_confs[index_0]
            key_1 = key_confs[index_1]
            tau = (d - d_0) / (d_1 - d_0)
            # if frame == 107:
            #    print(f"Frame 107 is interpolated from {index_0} and {index_1}, with tau {tau}")
            t,r = _mix(tau, key_0, key_1)
            t = t - r.apply(O) # Translate back to vanilla
            quat = r.as_quat()
            _add_key(rob, t, quat, frame+1)

    if args.animation_end >= 0:
        bpy.context.scene.frame_end = min(args.animation_end, desiredFrames - 1)
    else:
        bpy.context.scene.frame_end = desiredFrames - 1
    if args.image_frame >= 0:
        bpy.context.scene.frame_set(args.image_frame)

    if args.rendering_styles == 'ShadowOnWhite':
        bpy.context.scene.render.film_transparent = True
        # switch on nodes and get reference
        scene = bpy.data.scenes['Scene']
        scene.use_nodes = True
        tree = bpy.context.scene.node_tree

        # create input image node
        r_layer = tree.nodes.get('Render Layers')
        alpha_node = tree.nodes.new('CompositorNodeAlphaOver')
        out_node = tree.nodes.get('Composite')

        # link nodes
        links = tree.links
        link1 = links.new(r_layer.outputs[0], alpha_node.inputs[2])
        link2 = links.new(alpha_node.outputs[0], out_node.inputs[0])

    """
    ob = bpy.data.objects["Witness"]
    mp = ob.motion_path

    '''I copied this code from the web... it appears to sometimes bug out.
    This generates a curve along the path of the witness point '''
    # FIXME: it certainly would be buggy, we are expecting a piecewise linear curve,
    #        rather than a Bezier curve.
    if mp and "Curve" not in bpy.data.objects:
        path = bpy.data.curves.new('path','CURVE')
        curve = bpy.data.objects.new('Curve',path)
        bpy.context.scene.objects.link(curve)
        path.dimensions = '3D'
        spline = path.splines.new('BEZIER')
        spline.bezier_points.add(len(mp.points)-1)

        for i,o in enumerate(spline.bezier_points):
            o.co = mp.points[i].co
            o.handle_right_type = 'AUTO'
            o.handle_left_type = 'AUTO'
    """
    if args.saveas:
        bpy.ops.wm.save_as_mainfile(filepath=args.saveas, check_existing=False)
    if args.save_image:
        bpy.context.scene.cycles.samples = 512
        if args.cuda:
            enable_cuda()
        bpy.context.scene.render.filepath = args.save_image
        bpy.ops.render.render(write_still=True)
    if args.save_animation_dir:
        os.makedirs(args.save_animation_dir, exist_ok=True)
        bpy.context.scene.cycles.samples = 512
        if args.cuda:
            enable_cuda()
        bpy.context.scene.render.filepath = args.save_animation_dir
        if args.preview:
            bpy.ops.render.opengl(animation=True, view_context=False)
        else:
            bpy.ops.render.render(animation=True)
    if args.quit:
        bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
