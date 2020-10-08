import vision
import config
import numpy as np
import pyosr
import uw_random
import copy

def get_view_cfg(args):
    VIEW_CFG = config.VIEW_CFG
    if args.viewset == 'cube':
        VIEW_CFG = [(0, 4), (90, 1), (-90, 1)]
    elif args.viewset == '14' or (not args.viewset and args.ferev >= 4):
        VIEW_CFG = config.VIEW_CFG_REV4
    elif args.viewset == '22' or (not args.viewset and args.ferev != 1):
        VIEW_CFG = config.VIEW_CFG_REV2
    view_array = vision.create_view_array_from_config(VIEW_CFG)
    if args.view >= 0:
        view_num = 1
    else:
        view_num = len(view_array)
    return view_num, view_array

def create_renderer(args, creating_ctx=True):
    view_num, view_array = get_view_cfg(args)
    w = h = args.res

    if creating_ctx:
        dpy = pyosr.create_display()
        glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    if args.avi:
        r.avi = True
    r.pbufferWidth = w
    r.pbufferHeight = h
    r.setup()
    r.loadModelFromFile(args.envgeo)
    r.loadRobotFromFile(args.robgeo)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    if args.robcenter is not None:
        r.enforceRobotCenter(args.robcenter)

    if args.view >= 0:
        if args.obview >= 0:
            va = [view_array[args.obview]]
        else:
            va = [view_array[args.view]]
    else:
        va = view_array
    r.views = np.array(va, dtype=np.float32)
    return r

def actions_to_adist_array(actions, dim=uw_random.DISCRETE_ACTION_NUMBER, hotvector=None):
    n = len(actions)
    adists = np.zeros(
            shape=(n, 1, dim),
            dtype=np.float32)
    for i in range(n):
        adists[i, 0, actions[i]] = 1.0 if hotvector is None else hotvector[i]
    return adists

'''
Return a list of args for multiprocessing
The following fields are added to args according to --localcluster_nsampler and
--localcluster_portbase
  * ps_hosts : to locate parameter server (shared)
  * worker_hosts : to locate other workers (shared)
  * job_name : either 'ps' or 'worker' (non-shared)
  * task_index : for job_name == 'worker", indicate the index of task (non-shared)
'''
def assemble_distributed_arguments(args):
    args.ps_hosts += ['localhost:{}'.format(args.localcluster_portbase)]
    # Worker 0 only calculates gradients, others only sample
    nworker = args.localcluster_nsampler + 1
    for i in range(nworker):
        args.worker_hosts.append('localhost:{}'.format(args.localcluster_portbase+i+1))
    ps_args = copy.deepcopy(args)
    ps_args.job_name = 'ps'
    ps_args.task_index = 0 # task_index is also mandatory for ps
    ret = [ps_args]
    for i in range(nworker):
        w_args = copy.deepcopy(args)
        w_args.job_name = 'worker'
        w_args.task_index = i
        ret.append(w_args)
    return ret

def create_cluster_dic(args):
    if not args.ps_hosts or not args.worker_hosts:
        return None
    return {"ps": args.ps_hosts, "worker": args.worker_hosts}

SC_PRED_PERMUTATION = 1
SC_ACTION_PERMUTATION = 2
