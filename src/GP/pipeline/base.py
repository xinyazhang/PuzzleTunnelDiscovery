#!/usr/bin/env python3

import pyosr
import puzzle_parser

'''
Base classes of various tasks

To create subcommands:
    1. Create a new python module under pipeline/
    2. Add the name of module to pipe_driver.py
       Note: we cannot inspect modules automatically because we want to maintain the order of subcommands

Subcommands can setup the following members in args for different behaviours
* _tunnel_v_fn: for tunnel vertex file
* _env_fn: Environment Geometry [MUST SET]
* _rob_fn: Robot Geometry [MUST SET]
* _fb_resolution: frame buffer resolution for rendering
'''

class TaskInterface(object):
    def __init__(self, args):
        puzzle_parser.parse(args, args.puzzle)
        self._args = args
        self._tv_cache = np.load(args._tunnel_v_fn)['TUNNEL_V'] if args.tunnel_v_fn is not None else None
        self._uw = None

    def get_tunnel_v(self):
        return self._tv_cache

    def get_uw(self):
        return self._uw

def _setup_unitworld(uw, args):
    uw.loadModelFromFile(args._env_fn)
    uw.loadRobotFromFile(args._rob_fn)
    if hasattr(args, '_rob_ompl_center'):
        uw.enforceRobotCenter(args._rob_ompl_center)
    uw.scaleToUnit()
    uw.angleModel(0.0, 0.0)

def _create_renderer(args):
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    if hasattr(args, '_fb_resolution')
        r.pbufferWidth = args._fb_resolution
        r.pbufferHeight = args._fb_resolution
    r.setup()
    r.views = np.array([[0.0,0.0]], dtype=np.float32)
    return r

class ComputeTask(TaskBase):
    def __init__(self, args):
        super().__init__(args)
        self._uw = pyosr.UnitWorld()
        _setup_unitworld(self._uw, args)

class GpuTask(TaskBase):
    def __init__(self, args):
        super().__init__(args)
        self._uw = _create_renderer(args)
        _setup_unitworld(self._uw, args)
