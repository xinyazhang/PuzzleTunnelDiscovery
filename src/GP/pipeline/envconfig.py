#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import util
import os
from os.path import join
from six.moves import configparser
import subprocess
import condor

def setup_parser(subparsers):
    p = subparsers.add('config', help='Initialize a directory as workspace')
    p.add_argument('dir', help='Workspace directory')
    # p.add_argument('cfg', help='Config file')
    p.add_argument('condor', help='HTCondor submission file template')

_CONFIG_TEMPLATE =
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
LocalExecPath = {localpath} # The facade.py path on local node

GPUHost = TODO # Host name of GPU node, SSH host alias can be used
GPUExecPath = TODO # facade.py path on GPU node
GPUWorkspacePath = TODO # Workspace path on GPU node

CondorHost = TODO # Host name of HTCondor submission node, SSH host alias can be used
CondorExecPath = TODO # facade.py path on HTCondor node
CondorWorkspacePath = TODO # Workspace path on NTCondor node

# How many jobs are you authroized to run in parallel on HTCondor
# This is a hint for tasks partitioning
CondorQuota = 150

ChartReslution = 2048

[TrainingTrajectory]
PlannerAlgorithmID = 15 # RDT algorithm. This is usually the best choice among classical algorithms
CondorTimeThreshold = 0.05 # Time limit of each instance, unit: day(s)
CondorInstances = 100 # Number of instances to run on HTCondor in order to find the solution path

# In this section, we use numerical method to approximate the minimal clearance
[TrainingKeyConf]
CandidateNumber = 1024 # Number of points on each solution trajectory as candidate key configurations
ClearanceSample = 4096 # How many samples do we create to estimate the clearance volume in C-space
ClearanceTaskGranularity = 32768 # HTCondor task granularity, to minimize overhead
KeyConf = 1            # How many configurations we pick from candidates as key configurations.
                       # This varies among the training models
CollisionSample = 4096 # How many samples do we create to mark up the key surface

[TrainingWeightChart]
TouchSample = 32786    # How many touch configuration we shall generate for each key configuration
TouchSampleGranularity = 32768 # Hint about the task partition
MeshBoolGranularity = 1024 # Minimal task size hint: mesh boolean
UVProjectGranularity = 1024 # Minimal task size hint: mesh boolean

'''

def init_config_file(ws):
    try:
        condor.extract_template(open(args.condor, 'r'), open(ws.condor_template, 'w'))
        cfg = ws.configuration_file
        if not os.path.isfile(cfg):
            print(_CONFIG_TEMPLATE.format(localpath=os.getcwd()), file=open(cfg, 'w'))
        EDITOR = os.environ.get('EDITOR', 'vim')
        subprocess.call([EDITOR, cfg])
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
