# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
import pyosr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave
import vision
import tensorflow as tf
import rldriver
import cat
import config

def train_classification():
    pyosr.init()
    pyosr.create_gl_context(pyosr.create_display()) # FIXME: Each thread has one ctx
    init_state = np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        cater = cat.TrainCat(global_step)
        cater.run()

if __name__ == '__main__':
    train_classification()
