
import numpy as np
import tensorflow as tf
import tfutil
from math import sin,cos,tan,pi

RANK = 2
SHAPE = [3, 2]

AA_SHAPE = SHAPE + [3]
QUAT_SHAPE = SHAPE + [4]

assert len(SHAPE) == RANK

aa_tensor = tf.placeholder(tf.float32, shape=AA_SHAPE, name='aa_input')
quat_tensor = tfutil.aa_to_w_first_quaternion(aa_tensor)

assert quat_tensor.shape.as_list() == QUAT_SHAPE, '{} != {}'.format(quat_tensor.shape.as_list(), QUAT_SHAPE)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    aa = np.random.rand(*AA_SHAPE) * 2 * pi
    print(sess.run(quat_tensor, feed_dict={aa_tensor: aa}))

aa1_tensor = tf.placeholder(tf.float32, shape=AA_SHAPE, name='aa1_input')
aa2_tensor = tf.placeholder(tf.float32, shape=AA_SHAPE, name='aa2_input')
geodist_tensor = tfutil.axis_angle_geodesic_distance(aa1_tensor, aa2_tensor)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    aa1 = np.random.rand(*AA_SHAPE) * 2 * pi
    aa2 = np.random.rand(*AA_SHAPE) * 2 * pi
    print(sess.run(geodist_tensor, feed_dict={aa1_tensor: aa1, aa2_tensor: aa2}))
