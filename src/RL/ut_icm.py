# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave
import vision
import tensorflow as tf
import icm
import config

def dump_model(model):
    [params, out] = model
    for layer in params:
        [w,b] = layer
        print(w.name)
        print(b.name)

if __name__ == '__main__':
    action = tf.placeholder(tf.float32, shape=[None, 6])
    rgb_1 = tf.placeholder(tf.float32, shape=[None, 12, 112, 112, 3])
    rgb_2 = tf.placeholder(tf.float32, shape=[None, 12, 112, 112, 3])
    dep_1 = tf.placeholder(tf.float32, shape=[None, 12, 112, 112, 1])
    dep_2 = tf.placeholder(tf.float32, shape=[None, 12, 112, 112, 1])
    model = icm.IntrinsicCuriosityModule(action,
            rgb_1, dep_1,
            rgb_2, dep_2,
            config.SV_VISCFG,
            config.MV_VISCFG,
            256)
    print('## INVERSE MODEL')
    dump_model(model.get_inverse_model())
    print('## FORWARD MODEL')
    dump_model(model.get_forward_model())
