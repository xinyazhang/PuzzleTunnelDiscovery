#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import subprocess
import pathlib
import numpy as np
from scipy.misc import imsave

def autorun(args):
    ws = util.Workspace(args.dir)
    ws.deploy_to_gpu(util.WORKSPACE_SIGNATURE_FILE,
                     util.WORKSPACE_CONFIG_FILE,
                     util.TRAINING_DIR+'/')
