#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join, normpath
from six.moves import configparser
import subprocess

from . import util
from . import condor

def setup_parser(subparsers):
    p = subparsers.add('config', help='Initialize a directory as workspace')
    p.add_argument('dir', help='Workspace directory')
    # p.add_argument('cfg', help='Config file')
    p.add_argument('condor', help='HTCondor submission file template')

_CONFIG_TEMPLATE = \
'''
# THE RUNNING ENVIRONMENT CONFIGURATION FILE OF PUZZLE SOLVER
#
# Design:
# The whole pipeline is partitoned to run on three types of nodes:
# 1. The local node, in which:
#    a) user should have root access so a rich set of packages can be installed
#    b) the processor should have relatively high frequency for single threaded tasks
#    c) A GPU with OpenGL and EGL support is mandatory
#    d) Ideally this node should have largest amount of memory installed
# 2. A GPU node, where the TensorFlow based training/testing is done
#    a) Must support OpenGL and EGL as well.
#       + Note: if docker is used, the nvidia/cudagl image should be used instead of nvidia/cuda.
#               The latter one does not support OpenGL.
# 3. The HTCondor submission node, where the massive parallel executions are offloaded
#    This node should also be capable of (or allowed for) running moderate workloads.
#    GPU is NOT required on this node.
#
# libosr.so and pyosr.so must be compiled on all three nodes. GPU support can
# (and should) be disabled on HTCondor node.
#
# Local node should be able to ssh into GPU node and HTCondor submission node.
# Other ssh access is not necessary.
#
# On all three types of nodes, the ExecPath is directory that stores facade.py,
# and the WorkspacePath is the directory that store the workspace information.
#
# You only need to create workspace in the local node.
# The workspaces on the other two nodes will be deployed automatically with autorun()
#
# Note: each type of node only stores necessary files for its compute task,
# in order to save harddrive space.
[DEFAULT]
# Host name of GPU node, SSH host alias can be used
GPUHost = {GPUHost}
# facade.py path on GPU node
GPUExecPath = {GPUExecPath}
# Workspace path on GPU node
GPUWorkspacePath = {GPUWorkspacePath}

# Host name of HTCondor submission node, SSH host alias can be used
CondorHost = {CondorHost}
# facade.py path on HTCondor node
CondorExecPath = {CondorExecPath}
# Workspace path on NTCondor node
CondorWorkspacePath = {CondorWorkspacePath}

# How many jobs are you authroized to run in parallel on HTCondor
# This is a hint for tasks partitioning
CondorQuota = 150

ChartReslution = 2048

[TrainingTrajectory]
# RDT algorithm. This is usually the best choice among classical algorithms
PlannerAlgorithmID = 15
# Time limit of each instance, unit: day(s)
CondorTimeThreshold = 0.05
# Number of instances to run on HTCondor in order to find the solution path
CondorInstances = 100

# In this section, we use numerical method to approximate the minimal clearance
[TrainingKeyConf]
# Limit the number of trajectories
TrajectoryLimit = -1
# Number of points on each solution trajectory as candidate key configurations
CandidateNumber = 1024
# How many samples do we create to estimate the clearance volume in C-space
ClearanceSample = 4096
# HTCondor task granularity, tradeoff between minimizing overhead and
# maximizing the parallelism
# Default as 4 to prefer parallelism
ClearanceTaskGranularity = 4
# How many configurations we pick from candidates as key configurations.
# This varies among the training models
KeyConf = 1
# How many samples do we create to mark up the key surface
CollisionSample = 4096

[TrainingWeightChart]
# How many touch configuration we shall generate for each key configuration
TouchSample = 32768
# Hint about the task partition
# i.e. How many samples shall we generate in each worker process
# Note: each worker process produces its own output file
TouchSampleGranularity = 32768
# Minimal task size hint: mesh boolean
MeshBoolGranularity = 1024
# Minimal task size hint: mesh boolean
# UVProjectGranularity = 1024

[Prediction]
# Set the number of processes that predict the key configuration from
# the surface distribution, auto means number of (logic) processors
NumberOfPredictionProcesses = auto
NumberOfRotations           = 64
SurfacePairsToSample        = 1024
Margin                      = 1e-6

[GeometriK]

KeyPointAttempts  = 12
KeyConfigRotations = 64

[Solver]
# Number of samples in the PreDefined Sample set
PDSSize = 4194304
PDSBloom = 3072
# Maximum trials before cancel
Trials = 1


# In day(s), 0.01 ~= 14 minutes, 0.02 ~= 0.5 hour
TimeThreshold = 0.02
'''

def init_config_file(args, ws, oldws=None):
    try:
        condor.extract_template(open(args.condor, 'r'), open(ws.condor_template, 'w'))
        cfg = ws.configuration_file
        if not os.path.isfile(cfg):
            dic = {}
            if oldws is not None:
                rel_old_to_new = os.path.relpath(ws.dir, start=oldws.dir)
                dic = {
                        'GPUHost': oldws.config.get('DEFAULT', 'GPUHost'),
                        'GPUExecPath': oldws.config.get('DEFAULT', 'GPUExecPath'),
                        'GPUWorkspacePath': normpath(join(oldws.config.get('DEFAULT', 'GPUWorkspacePath'), rel_old_to_new)),
                        'CondorHost': oldws.config.get('DEFAULT', 'CondorHost'),
                        'CondorExecPath': oldws.config.get('DEFAULT', 'CondorExecPath'),
                        'CondorWorkspacePath': normpath(join(oldws.config.get('DEFAULT', 'CondorWorkspacePath'), rel_old_to_new)),
                      }
            print(_CONFIG_TEMPLATE.format(**dic), file=open(cfg, 'w'))
        EDITOR = os.environ.get('EDITOR', 'vim')
        subprocess.run([EDITOR, cfg])
    except FileNotFoundError as e:
        print(e)
        return
    '''
    config = configparser.ConfigParser()
    config.read(cfg)
    util.deploy_workspace(args.dir, cfg.get('DEFAULT', 'GPUHost'), cfg.get('DEFAULT', 'GPUWorkspacePath'))
    util.deploy_workspace(args.dir, cfg.get('DEFAULT', 'CondorHost'), cfg.get('DEFAULT', 'CondorWorkspacePath'))
    '''
    # print('''The Puzzle Workspace is Ready! Use 'runall' to run the pipeline automatically.''')
    # print('''Use -h to list commands to run each pipeline stage independently.''')
