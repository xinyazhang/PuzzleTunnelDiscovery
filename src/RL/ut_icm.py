import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave
import vision
import tensorflow as tf
import icm
import config

if __name__ == '__main__':
    action = tf.placeholder(tf.float32, shape=[None, 6])
    rgb_1 = tf.placeholder(tf.float32, shape=[None, 12, 112, 112, 3])
    rgb_2 = tf.placeholder(tf.float32, shape=[None, 12, 112, 112, 3])
    dep_1 = tf.placeholder(tf.float32, shape=[None, 12, 112, 112, 1])
    dep_2 = tf.placeholder(tf.float32, shape=[None, 12, 112, 112, 1])
    model = icm.IntrinsicCuriosityModule(action,
            rgb_1, dep_1,
            rgb_2, dep_2,
            [(45, 6), (-45, 6)],
            config.SV_VISCFG,
            config.MV_VISCFG,
            256)
    print(model.get_inverse_model())
    print(model.get_forward_model())
