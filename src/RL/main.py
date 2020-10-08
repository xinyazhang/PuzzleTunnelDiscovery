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
import config

def test():
    pyosr.init()
    pyosr.create_gl_context(pyosr.create_display())

    r=pyosr.Renderer()
    r.setup()
    r.loadModelFromFile('../res/alpha/env-1.2.obj')
    r.loadRobotFromFile('../res/alpha/robot.obj')
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.views = np.array([[30.0, float(i)] for i in range(0, 360, 30)], dtype=np.float32)
    print(r.views)
    print(r.views.shape)
    w = r.pbufferWidth
    h = r.pbufferHeight
    r.state = np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    r.render_mvrgbd()
    img = r.mvrgb.reshape(w * r.views.shape[0], h, 3)
    dep = r.mvdepth.reshape(w * r.views.shape[0], h)
    depimg = Image.fromarray(dep)
    imsave('mvrgb.png', img)
    depimg.save('mvdepth.tiff')

    # Vision NN

    vis0 = vision.VisionLayerConfig(16)
    vis1 = vision.VisionLayerConfig(16)
    vis1.strides = [1, 3, 3, 1]
    vis1.kernel_size = [5, 5]
    vis2 = vision.VisionLayerConfig(32)
    vis3 = vision.VisionLayerConfig(64)

    imgin = img.reshape(r.views.shape[0], w, h, 3)
    depin = r.mvdepth.reshape(r.views.shape[0], w , h, 1)
    print('imgin shape: {}'.format(imgin.shape))
    mv_color = vision.VisionNetwork(imgin.shape,
            [vis0, vis1, vis2, vis3], 0, 256)
    featvec = mv_color.features
    print('mv_color.featvec.shape = {}'.format(featvec.shape))

    sq_featvec = tf.reshape(featvec, [-1, 16, 16, 1]) # Squared feature vector
    chmv_featvec = tf.transpose(sq_featvec, [3, 1, 2, 0])
    print('chmv_featvec {}'.format(chmv_featvec.shape))

    pvis0 = vision.VisionLayerConfig(64)
    color = vision.VisionNetwork(None, [pvis0], 0, 256, chmv_featvec)
    featvec = color.features
    print('featvec.shape = {}'.format(featvec.shape))

    exit()

    mvpix = r.render_mvdepth_to_buffer()
    img = mvpix.reshape(w * r.views.shape[0], h)
    print(img.shape)
    plt.pcolor(img)
    plt.show()
    r.teardown()
    pyosr.shutdown()

def train_puzzle():
    pyosr.init()
    pyosr.create_gl_context(pyosr.create_display()) # FIXME: Each thread has one ctx
    init_state = np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    with tf.Graph().as_default():
        driver = rldriver.RLDriver(['../res/alpha/env-1.2.obj', '../res/alpha/robot.obj'],
                np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)],
                config.SV_VISCFG,
                config.MV_VISCFG,
                use_rgb=True)

if __name__ == '__main__':
    train_puzzle()
