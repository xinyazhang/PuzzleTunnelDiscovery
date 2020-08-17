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

[SYSTEM]
# Host name of GPU node, SSH host alias can be used
GPUHost = {GPUHost}
# facade.py path on GPU node
GPUExecPath = {GPUExecPath}
# Workspace path on GPU node
GPUWorkspacePath = {GPUWorkspacePath}

# Host name of HTCondor submission node, SSH host alias can be used
CondorHost = {CondorHost}
# Extra HTCondor submission nodes, comma separated
ExtraCondorHosts =
# facade.py path on HTCondor node
CondorExecPath = {CondorExecPath}
# Workspace path on NTCondor node
CondorWorkspacePath = {CondorWorkspacePath}

# How many jobs are you authroized to run in parallel on HTCondor
# This is a hint for tasks partitioning
CondorQuota = 150

ChartReslution = 2048

# the email address to send the notifications
# Note only situations that require user interactions will be notified, e.g.:
#  A job is on hold on HTCondor.
mailto = SHOULD_NOT_BE_HERE_AND_KEEP_IT_PRIVATE
# Sometimes we do not have access to mail locally
mailfrom_host = SHOULD_NOT_BE_HERE_AND_KEEP_IT_PRIVATE

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

[TrainingCluster]
# Format Group# = <puzzle name>.piece1,<puzzle name>.piece2
# Example
# Group0 = alpha.piece1,alpha.piece2
# Group1 = duet.piece1
# Group2 = duet.piece2

[Prediction]
Enable = yes
# Set the number of processes that predict the key configuration from
# the surface distribution, auto means number of (logic) processors
NumberOfPredictionProcesses = auto
NumberOfRotations           = 256
SurfacePairsToSample        = 1024
Margin                      = 1e-6
# Reuse trained workspace so we can separate the training workspace from testing workspace
# May use relative path
ReuseWorkspace              = {ReuseWorkspace}
OversamplingRatio           = 10
OversamplingClearanceSample = 128


[GeometriK]

FineMeshV = 500000
KeyPointAttempts  = 32
KeyConfigRotations = 512

EnableNotchDetection = yes

[RoboGeoK]
KeyPointAttempts = 32
EnvKeyPoints = 1024
KeyConfigRotations = 64

[Solver]
EnableKeyConfScreening = yes
# Number of samples in the PreDefined Sample set
# Not used
# PDSSize = 4194304
PDSBloom = 3072
# Maximum trials before cancel
# Not used
# Trials = 1


# In day(s), 0.01 ~= 14 minutes, 0.02 ~= 0.5 hour
TimeThreshold = 0.02

'''

def init_config_file(args, ws, oldws=None):
    print(f'calling init_config_file')
    interactive = (not hasattr(args, 'quiet') or not args.quiet)
    try:
        condor.extract_template(open(args.condor, 'r'), open(ws.condor_template, 'w'))
        cfg = ws.configuration_file
        print(f'config file {cfg}')
        if not os.path.isfile(cfg):
            if oldws is not None:
                old_config = configparser.ConfigParser()
                old_config.read_string(_CONFIG_TEMPLATE)
                old_dic = oldws.config_as_dict
                if 'SYSTEM' not in old_dic:
                    """
                    Copy DEFAULT to SYSTEM
                    This handles DEFAULT -> SYSTEM section renaming
                    """
                    old_dic['SYSTEM'] = { k:v for k,v in oldws.config.items("DEFAULT")}
                util.update_config_with_dict(old_config, old_dic)
                rel_old_to_new = os.path.relpath(ws.dir, start=oldws.dir)
                old_reuse = old_config.get('Prediction', 'ReuseWorkspace')
                gpu_ws = normpath(join(old_config.get('SYSTEM', 'GPUWorkspacePath'), rel_old_to_new))
                if old_reuse:
                    new_reuse = os.path.relpath(join(oldws.dir, old_reuse), start=gpu_ws)
                else:
                    new_reuse = ''
                dic = {
                        'GPUHost': old_config.get('SYSTEM', 'GPUHost'),
                        'GPUExecPath': old_config.get('SYSTEM', 'GPUExecPath'),
                        'GPUWorkspacePath': gpu_ws,
                        'CondorHost': old_config.get('SYSTEM', 'CondorHost'),
                        'CondorExecPath': old_config.get('SYSTEM', 'CondorExecPath'),
                        'CondorWorkspacePath': normpath(join(old_config.get('SYSTEM', 'CondorWorkspacePath'), rel_old_to_new)),
                        'ReuseWorkspace': new_reuse
                      }
                if hasattr(args, 'override') and args.override is not None:
                    patch = dict(item.split("=") for item in args.override.split(","))
                    dic.update(patch)
            elif not interactive:
                pwd = str(os.getcwd())
                wspath = os.path.join(pwd, ws.dir)
                dic = {
                        'GPUHost': 'localhost',
                        'GPUExecPath': pwd,
                        'GPUWorkspacePath': wspath,
                        'CondorHost': 'localhost',
                        'CondorExecPath': pwd,
                        'CondorWorkspacePath': wspath,
                        'ReuseWorkspace': args.trained_workspace,
                      }
            else:
                dic = {
                        'GPUHost': '',
                        'GPUExecPath': '',
                        'GPUWorkspacePath': '',
                        'CondorHost': '',
                        'CondorExecPath': '',
                        'CondorWorkspacePath': '',
                        'ReuseWorkspace': args.trained_workspace,
                      }
            print(f'Creating config file at {cfg}')
            print(_CONFIG_TEMPLATE.format(**dic), file=open(cfg, 'w'))
        if interactive:
            EDITOR = os.environ.get('EDITOR', 'vim')
            subprocess.run([EDITOR, cfg])
    except FileNotFoundError as e:
        print(e)
        return
    # print('''The Puzzle Workspace is Ready! Use 'runall' to run the pipeline automatically.''')
    # print('''Use -h to list commands to run each pipeline stage independently.''')
