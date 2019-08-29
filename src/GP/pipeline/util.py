#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from six.moves import configparser
import os
import time
import sys
import subprocess
import colorama
import itertools
import numpy as np
from datetime import datetime;

from . import parse_ompl

# CAVEAT: sys.executable is empty string on Condor Worker node.
PYTHON = sys.executable
#assert PYTHON is not None and PYTHON != '', 'Cannot find python through sys.executable, which is {}'.format(PYTHON)

WORKSPACE_SIGNATURE_FILE = '.puzzle_workspace'
# Core files
WORKSPACE_CONFIG_FILE = 'config'
CONDOR_TEMPLATE = 'template.condor'
PUZZLE_CFG_FILE = 'puzzle.cfg' # In every puzzle directory
# Top level Directories
TRAINING_DIR = 'train'
EXTRA_TRAINING_DIR = 'extrain'
TESTING_DIR = 'test'
CONDOR_SCRATCH = 'condor_scratch'
NEURAL_SCRATCH = 'nn_scratch'
SOLVER_SCRATCH = 'solver_scratch'
BASELINE_SCRATCH = 'baseline_scratch'
PERFORMANCE_LOG_DIR = 'performance_log'
# Protocol files/directories
# Used by multiple pipeline parts
#
# Note: we use UPPERCASE KEY to indicate this is training data/ground truth.
PREP_KEY_CAN_SCRATCH = os.path.join(CONDOR_SCRATCH, 'training_key_can')
KEY_CANDIDATE_FILE = os.path.join(TRAINING_DIR, 'KeyCan.npz')
# Multi trajectory key candidate file
MT_KEY_CANDIDATE_FILE = os.path.join(TRAINING_DIR, 'KeyCan.hdf5')
KEY_FILE = os.path.join(TRAINING_DIR, 'KEY.npz')
TRAJECTORY_DIR = os.path.join(CONDOR_SCRATCH, 'training_trajectory')
UV_DIR = os.path.join(CONDOR_SCRATCH, 'training_key_uvproj')

PIXMARGIN = 2

KEYCONF_CLEARANCE_DIR = 'keyconf_clearance'
KEY_POINT_FMT = 'geometrik_key_point_of_{}-{}.npz'
GEOMETRIK_KEY_PREDICTION_FMT = 'geometrik_forest_roots-{}.npz'
OVERSAMPLED_KEY_PREDICTION_FMT = 'oversampled_forest_roots-{}.npz'
UNSCREENED_KEY_PREDICTION_FMT = 'unscreened_forest_roots-{}.npz'
SCREENED_KEY_PREDICTION_FMT = 'forest_roots-{}.npz'
SOLUTION_FMT = 'path-{trial}.{type_name}.txt'
PDS_SUBDIR = 'pds'

RDT_FOREST_ALGORITHM_ID = 15
RDT_FOREST_INIT_AND_GOAL_RESERVATIONS = 2
RDT_CONNECT_ALGORITHM_ID = 17


'''
WORKSPACE HIERARCHY

workspace/
+-- .puzzle_workspace   # Signature
+-- config              # Runtime configuration
+-- template.condor     # Template HTCondor submission file
+-- train/              # training data
|   +-- puzzle.cfg      # OMPL cfg
|   +-- <Env>.obj       # .OBJ file for environment geometry, name may vary
|   +-- <Rob>.obj       # .OBJ file for robot geometry, name may vary
|   +-- KEY.npz         # Detetcted key configurations from the training puzzle
|   +-- env_chart.npz   # Weight Chart for environment geometry
|   +-- env_chart.png   #  ... in PNG format
|   +-- <Env>.png       #  Screened weight in PNG format ready for pyosr to load
|   +-- rob_chart.npz   # Weight Chart for robot geometry
|   +-- rob_chart.png   #  ... in PNG format
|   +-- <Rob>.png       #  Screened weight in PNG format ready for pyosr to load
+-- test/               # testing data
|   +-- <Puzzle 1>/     # Each puzzle has its own directory
|   |   +-- puzzle.cfg  # OMPL cfg
|   |   +-- <Env>.obj   # .OBJ file for environment geometry, name may vary
|   |   +-- <Rob>.obj   # .OBJ file for robot geometry, name may vary
|   |   +-- env-atex.npz# Environment surface distribution
|   |   +-- rob-atex.npz# Robot surface distribution
|   |   +-- key.npz     # sampled key configurations
|   +-- <Puzzle 2>/     # Each puzzle has its own directory
|    ...
+-- condor_scratch/     # Scratch directory that store stdin stdout stderr log generated by HTCondor
|   +-- training_trajectory     # search for the solution trajectory (a.k.a. solution path)
|   +-- training_key_can        # search for the key configuration by estimating the clearance
|   +-- training_key_touch      #
|   +-- training_key_isect      #
|   +-- training_key_uvproj     #
+-- nn_scratch/         # Scratch directory for NN checkpoints/logs
|   +-- env.pid         # PID file of the training process for env geometry
|   +-- rob.pid         # PID file of the training process for rob geometry
|   +-- rob/            # checkpoints for rob
|   +-- env/            # checkpoints for env
+-- solver_scratch/     # Scratch directory for OMPL solvers
|   +-- <Puzzle 1>/     # Each puzzle has its own directory
|   |   +-- keyconf_clearance   #
|   |   +-- pds/        # Predefined sample set
'''

def _load_unit_world(uw, puzzle_file):
    puzzle, config = parse_ompl.parse_simple(puzzle_file)
    uw.loadModelFromFile(puzzle.env_fn)
    uw.loadRobotFromFile(puzzle.rob_fn)
    uw.scaleToUnit()
    uw.angleModel(0.0, 0.0)
    uw.recommended_cres = uw.scene_scale * config.getfloat('problem', 'collision_resolution', fallback=0.001)

def create_unit_world(puzzle_file):
    # Well this is against PEP 08 but we do not always need pyosr
    # (esp in later pipeline stages)
    # Note pyosr is a heavy-weight module with
    #   + 55 dependencies on Fedora 29 (control node)
    #   + 43 dependencies on Ubuntu 18.04 (GPU node)
    #   + 26 dependencies on Ubuntu 16.04 (HTCondor node)
    import pyosr
    uw = pyosr.UnitWorld()
    _load_unit_world(uw, puzzle_file)
    return uw

def shell(args):
    log('Running {}'.format(args))
    return subprocess.call(args)

_egl_dpy = None

def create_offscreen_renderer(puzzle_file, resolution=256):
    global _egl_dpy
    import pyosr
    if _egl_dpy is None:
        pyosr.init()
        _egl_dpy  = pyosr.create_display()
    glctx = pyosr.create_gl_context(_egl_dpy)
    r = pyosr.Renderer()
    r.pbufferWidth = resolution
    r.pbufferHeight = resolution
    r.setup()
    r.views = np.array([[0.0,0.0]], dtype=np.float32)
    _load_unit_world(r, puzzle_file)
    return r

def _rsync(from_host, from_pather, to_host, to_pather, *paths):
    # Note: do NOT use single target multiple source syntax
    #       the target varies among source paths.
    from_prefix = '' if from_host is None else from_host+':'
    to_prefix = '' if to_host is None else to_host+':'
    for rel_path in paths:
        ret = shell(['rsync', '-aR',
                     '{}{}/./{}'.format(from_prefix, from_pather(), rel_path),
                     '{}{}/'.format(to_prefix, to_pather())])

class Workspace(object):
    _egl_dpy = None

    def __init__(self, workspace_dir, init=False):
        self.workspace_dir = os.path.abspath(workspace_dir)
        log("[Workspace] created as {}".format(self.workspace_dir))
        self._config = None
        # We may cache multiple UnitWorld objects with this directory
        self._uw_dic = {}
        if not init:
            self.verify_signature()
        self._current_trial = 0
        self.nn_profile = ''
        self._timekeeper = {}
        self._override_condor_host = None
        self._extra_condor_hosts = None

    def get_path(self, optname):
        return self.config.get('DEFAULT', optname)

    @property
    def dir(self):
        return self.workspace_dir

    @property
    def config(self):
        if self._config is None:
            self._config = configparser.ConfigParser()
            self._config.read(self.configuration_file)
        return self._config

    @property
    def chart_resolution(self):
        return self.config.getint('DEFAULT', 'ChartReslution')

    # This function is designed to be called on non-condor hosts
    def condor_exec(self, xfile=''):
        return os.path.join(self.get_path('CondorExecPath'), xfile)

    # This function is designed to be called on condor hosts locally
    # The home directory is only known on local system
    # and hence expanduser can return correct path.
    def condor_local_exec(self, *paths):
        return os.path.abspath(os.path.expanduser(self.condor_exec(*paths)))

    def gpu_exec(self, xfile=''):
        return os.path.join(self.get_path('GPUExecPath'), xfile)

    def gpu_local_exec(self, xfile=''):
        return os.path.abspath(os.path.expanduser(self.gpu_exec(*paths)))

    def local_ws(self, *paths):
        return os.path.abspath(os.path.expanduser(os.path.join(self.workspace_dir, *paths)))

    # Get the path inside the condor workspace
    # Most code is supposed to run locally. Only use it on remote calling code!
    def condor_ws(self, *paths):
        return os.path.join(self.get_path('CondorWorkspacePath'), *paths)

    def gpu_ws(self, *paths):
        return os.path.join(self.get_path('GPUWorkspacePath'), *paths)

    @property
    def signature_file(self):
        return self.local_ws(WORKSPACE_SIGNATURE_FILE)

    def touch_signature(self):
        pathlib.Path(self.signature_file).touch()

    def test_signature(self):
        return os.path.isfile(self.signature_file)

    def verify_signature(self):
        if not self.test_signature():
            fatal("{} is not initialized as a puzzle workspace. Exiting".format(self.workspace_dir))
            exit()

    def verify_training_puzzle(self):
        if not os.path.isfile(self.training_puzzle):
            fatal("{} is not initialized with a training puzzle. Exiting".format(self.dir))
            exit()

    @property
    def training_dir(self):
        return self.local_ws(TRAINING_DIR)

    @property
    def training_puzzle(self):
        return self.local_ws(TRAINING_DIR, PUZZLE_CFG_FILE)

    @property
    def testing_dir(self):
        return self.local_ws(TESTING_DIR)

    @property
    def configuration_file(self):
        return self.local_ws(WORKSPACE_CONFIG_FILE)

    @property
    def condor_template(self):
        return self.local_ws(CONDOR_TEMPLATE)

    def override_condor_host(self, new_host):
        self._override_condor_host = str(new_host)

    @property
    def condor_host(self):
        if self._override_condor_host:
            return self._override_condor_host
        return self.config.get('DEFAULT', 'CondorHost')

    @property
    def condor_extra_hosts(self):
        if self._extra_condor_hosts is None:
            hostlist = self.config.get('DEFAULT', 'ExtraCondorHosts', fallback='')
            self._extra_condor_hosts = hostlist.split(',')
        return self._extra_condor_hosts

    @property
    def condor_all_hosts(self):
        return [self.condor_host] + self.condor_extra_hosts

    @property
    def gpu_host(self):
        return self.config.get('DEFAULT', 'GPUHost')

    def condor_unit_world(self, puzzle_dir):
        if puzzle_dir not in self._uw_dic:
            self._uw_dic[puzzle_dir] = create_unit_world(self.condor_ws(puzzle_dir, PUZZLE_CFG_FILE))
        return self._uw_dic[puzzle_dir]

    def remote_command(self, host, exec_path, ws_path,
                       pipeline_part, cmd,
                       auto_retry=True,
                       in_tmux=False,
                       with_trial=False,
                       extra_args='',
                       use_nn_profile=False):
        script = ''
        script += 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi; '
        script += 'cd {}\n'.format(exec_path)
        if in_tmux:
            script += 'tmux new-session -A -s puzzle_workspace '
        script += './facade.py {ppl} --stage {cmd} '.format(ppl=pipeline_part, cmd=cmd)
        if with_trial:
            script += ' --current_trial {} '.format(self.current_trial)
        if use_nn_profile and self.nn_profile:
            script += ' --nn_profile {} '.format(self.nn_profile)
        if extra_args:
            script += ' {} '.format(extra_args)
        script += ' {ws}'.format(ws=ws_path)
        if in_tmux:
            # tmux needs a terminal
            remoter = ['ssh', '-t', host]
        else:
            remoter = ['ssh', host]
        ret = shell(remoter + [script])
        while ret == 255:
            if not auto_retry:
                return ret
            print("SSH Connection to {} is probably broken, retry after 5 secs".format(host))
            time.sleep(5)
            ret = shell(['ssh', host, script + ' --only_wait'])
        if ret != 0:
            print("Remote error, exiting")
            exit()
        return ret

    '''
    Note: directory must end with /
    '''
    def deploy_to_condor(self, *paths):
        shell(['ssh', self.condor_host, 'mkdir', '-p', self.condor_ws()])
        _rsync(None, self.local_ws, self.condor_host, self.condor_ws, *paths)

    def fetch_condor(self, *paths):
        _rsync(self.condor_host, self.condor_ws, None, self.local_ws, *paths)

    def deploy_to_gpu(self, *paths):
        shell(['ssh', self.gpu_host, 'mkdir', '-p', self.gpu_ws()])
        _rsync(None, self.local_ws, self.gpu_host, self.gpu_ws, *paths)

    def fetch_gpu(self, *paths):
        _rsync(self.gpu_host, self.gpu_ws, None, self.local_ws, *paths)

    def checkpoint_dir(self, geo_type):
        if self.nn_profile:
            name = '{}.{}'.format(geo_type, self.nn_profile)
        else:
            name = geo_type
        return self.local_ws(NEURAL_SCRATCH, name)

    def training_puzzle_generator(self):
        # More flexible, allows ws with only 'extrain'
        if os.path.isfile(self.training_puzzle):
            yield self.training_puzzle, 'train'
        exdir = self.local_ws(EXTRA_TRAINING_DIR)
        if not os.path.isdir(exdir):
            return
        for ent in os.listdir(exdir):
            puzzle_fn = self.local_ws(EXTRA_TRAINING_DIR, ent, 'puzzle.cfg')
            if not os.path.isfile(puzzle_fn):
                log("Cannot find puzzle file {}. continue to next dir".format(puzzle_fn))
                continue
            yield puzzle_fn, ent

    def test_puzzle_generator(self, target_puzzle_name=''):
        for ent in os.listdir(self.local_ws(TESTING_DIR)):
            if target_puzzle_name and target_puzzle_name != ent:
                continue
            puzzle_fn = self.local_ws(TESTING_DIR, ent, 'puzzle.cfg')
            if not os.path.isfile(puzzle_fn):
                log("Cannot find puzzle file {}. continue to next dir".format(puzzle_fn))
                continue
            yield puzzle_fn, ent

    def condor_host_vs_test_puzzle_generator(self):
        hosts = self.condor_all_hosts
        for i,(puzzle_fn,puzzle_name) in enumerate(self.test_puzzle_generator()):
            yield hosts[i % len(hosts)], puzzle_fn, puzzle_name

    def atex_prediction_file(self, puzzle_fn, geo_type, trial_override=None):
        trial = self.current_trial if trial_override is None else trial_override
        return os.path.join(pathlib.Path(puzzle_fn).parent, '{}-atex_{}.npz'.format(geo_type, trial))

    def keypoint_prediction_file(self, puzzle_name, geo_type, trial_override=None):
        trial = self.current_trial if trial_override is None else trial_override
        return self.local_ws(TESTING_DIR, puzzle_name,
                             KEY_POINT_FMT.format(geo_type, trial))

    def screened_keyconf_prediction_file(self, puzzle_name, trial_override=None):
        return self.keyconf_file_from_fmt(puzzle_name, SCREENED_KEY_PREDICTION_FMT, trial_override)
        # trial = self.current_trial if trial_override is None else trial_override
        # ret = self.local_ws(TESTING_DIR, puzzle_name,
        #                     SCREENED_KEY_PREDICTION_FMT.format(trial))
        '''
        if for_read and not os.path.exists(ret):
            warn("[sample_pds] forest root file {} does not exist")
            probe_trial = self.current_trial - 1
            while probe_trial >= 0:
                key_fn_0 = self.local_ws(TESTING_DIR, puzzle_name,
                                         SCREENED_KEY_PREDICTION_FMT.format(probe_trial))
                if os.path.exists(key_fn_0):
                    break
                probe_trial -= 1
            if probe_trial < 0:
                fatal("[util.keyconf_prediction_file] Cannot {}, nor any old old prediction files".format(ret))
            link_target = SCREENED_KEY_PREDICTION_FMT.format(probe_trial)
            os.symlink(link_target, ret)
            warn("[util.keyconf_prediction_file] sylink {} to {} as forest root file".format(link_target, ret))
        '''
        return ret

    def keyconf_prediction_file(self, puzzle_name, trial_override=None):
        return self.keyconf_file_from_fmt(puzzle_name, UNSCREENED_KEY_PREDICTION_FMT, trial_override)

    def oversampled_keyconf_prediction_file(self, puzzle_name, trial_override=None):
        return self.keyconf_file_from_fmt(puzzle_name, OVERSAMPLED_KEY_PREDICTION_FMT, trial_override)

    def keyconf_file_from_fmt(self, puzzle_name, FMT, trial_override=None):
        trial = self.current_trial if trial_override is None else trial_override
        ret = self.local_ws(TESTING_DIR, puzzle_name,
                            FMT.format(trial))
        return ret

    def solution_file(self, puzzle_name, type_name, trial_override=None):
        trial = self.current_trial if trial_override is None else trial_override
        return self.local_ws(TESTING_DIR, puzzle_name,
                             SOLUTION_FMT.format(trial=trial, type_name=type_name))

    def set_current_trial(self, trial):
        if trial is not None:
            self._current_trial = trial

    def get_current_trial(self):
        return self._current_trial

    current_trial = property(get_current_trial, set_current_trial)

    def open_performance_log(self):
        os.makedirs(self.local_ws(PERFORMANCE_LOG_DIR), exist_ok=True)
        return open(self.local_ws(PERFORMANCE_LOG_DIR, 'log.{}'.format(self.current_trial)), 'a')

    def timekeeper_start(self, stage_name, puzzle_name='*'):
        with self.open_performance_log() as f:
            t = datetime.utcnow()
            print('[{}][{}] starting at {}'.format(stage_name, puzzle_name, t), file=f)
        self._timekeeper[stage_name] = t

    def timekeeper_finish(self, stage_name, puzzle_name='*'):
        t = datetime.utcnow()
        if stage_name in self._timekeeper:
            delta = t - self._timekeeper[stage_name]
        else:
            delta = None
        with self.open_performance_log() as f:
            print('[{}][{}] finished at {}'.format(stage_name, puzzle_name, t), file=f)
            print('[{}][{}] cost {}'.format(stage_name, puzzle_name, delta), file=f)

def trim_suffix(fn):
    return os.path.splitext(fn)[0]

def padded(current:int, possible_max:int):
    return str(current).zfill(len(str(possible_max)))

def ask_user(question):
    check = str(input(question + " (Y/N): ")).lower().strip()
    try:
        if check[0] == 'y':
            return True
        elif check[0] == 'n':
            return False
        else:
            print('Invalid Input')
            return ask_user(question)
    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return ask_user(question)

def _colorp(color, s):
    print(color + s + colorama.Style.RESET_ALL)

def log(s):
    _colorp(colorama.Style.DIM, s)

def warn(s):
    #_colorp(colorama.Style.BRIGHT + colorama.Fore.RED, s)
    _colorp(colorama.Style.BRIGHT + colorama.Fore.YELLOW, s)

def fatal(s):
    _colorp(colorama.Style.BRIGHT + colorama.Fore.RED, s)

def ack(s):
    ts = '[{}]'.format(datetime.utcnow())
    _colorp(colorama.Style.BRIGHT + colorama.Fore.GREEN, ts + s)

def pwait(pid):
    if pid < 0:
        return 0
    return subprocess.run(['tail', '--pid={}'.format(pid), '-f', '/dev/null'])

def rangestring_to_list(x):
    result = []
    for part in x.split(','):
        if '-' in part:
            a, b = part.split('-')
            a, b = int(a), int(b)
            result.extend(range(a, b + 1))
        else:
            a = int(part)
            result.append(a)
    return result

def xz(fn):
    shell(['xz', '-f', fn])

def safe_concatente(nparray_list, axis=0):
    true_list = []
    for arr in nparray_list:
        if arr.shape[axis] != 0:
            true_list.append(arr)
    return np.concatenate(true_list, axis=axis)

def access_keys(d, keys):
    ret = []
    for k in keys:
        if k in d:
            ret.append(d[k])
        else:
            ret.append(None)
    return ret

def access_keypoints(d, geo_type):
    return safe_concatente(access_keys(d, ['KEY_POINT_AMBIENT', 'NOTCH_POINT_AMBIENT']), axis=0)
    # debug code
    if geo_type == 'rob':
        return safe_concatente(access_keys(d, ['KEY_POINT_AMBIENT']), axis=0)
    else:
        return safe_concatente(access_keys(d, ['NOTCH_POINT_AMBIENT']), axis=0)
    # Old implementation
    kps = d['KEY_POINT_AMBIENT']
    if 'NOTCH_POINT_AMBIENT' in d:
        nps = d['NOTCH_POINT_AMBIENT']
        if nps.shape[0] != 0:
            if kps.shape[0] != 0:
                kps = np.concatenate((kps, nps), axis=0)
            else:
                kps = nps
    return kps

def lsv(indir, prefix, suffix):
    ret = []
    for i in itertools.count(0):
        fn = "{}/{}{}{}".format(indir, prefix, i, suffix)
        if not os.path.exists(fn):
            if not ret:
                raise FileNotFoundError("Cannot even locate the a single file under {}. Complete path: {}".format(indir, fn))
            return ret
        ret.append(fn)
