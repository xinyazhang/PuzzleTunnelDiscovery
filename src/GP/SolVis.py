#!/usr/bin/env -S blender -P
# Note: the shebang line above requires coreutils 8.30 and above
# It is recommended to invoke this script with facade.py for portability

import sys
import bpy
import argparse
import math
import numpy as np
from scipy.spatial.transform import Rotation


#ui stuff... can ignore this

class SimplePanel(bpy.types.Panel):
    bl_id_name = "object.simple_operator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_category = "Motion planning"
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

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('env', help='ENV geometry')
    p.add_argument('rob', help='ROB geometry')
    p.add_argument('qpath', help='Vanilla configuration path')
    p.add_argument('out', help='Output dir')
    p.add_argument('--total_frames', help='Total number of frames to render', default=1440)
    argv = sys.argv
    return p.parse_args(argv[argv.index("--") + 1:])

def main():
    args = parse_args()

    bpy.utils.register_class(SimplePanel)
    bpy.utils.register_class(Prev)
    bpy.utils.register_class(Next)

    # Load meshes
    bpy.ops.import_scene.obj(filepath=args.env, axis_forward='Y', axis_up='Z')
    bpy.context.selected_objects[0].name = 'Env'
    bpy.ops.import_scene.obj(filepath=args.rob, axis_forward='Y', axis_up='Z')
    bpy.context.selected_objects[0].name = 'Rob'
    bpy.ops.mesh.primitive_uv_sphere_add()
    bpy.context.selected_objects[0].name = 'Witness'
    print(bpy.data.objects)

    # matrix detailing path of the object. framesamples[i] is coordinates of the robot at step i
    frameSamples = []
    # distance travelled per step, using weighted distance. Determines keyrame placement
    distances = [0]
    rotationWeight = 50     #weight to rotational component of distance
    totalDist = 0           #total distance of path.
    currSelection = 0

    # input file location, change as required
    vanilla_path = np.loadtxt(args.qpath)

    prevQ = []
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

    bpy.context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'

#initial coordinates of point on robot when robot's position is identity
    witnessCoords = np.array([40,41,-2])

#keyframe insertion
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

    ob = bpy.data.objects["Witness"]
    mp = ob.motion_path

    """
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

if __name__ == "__main__":
    main()
