# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
import tensorflow as tf
import tfutil
from math import sin,cos,tan,pi

DTYPE=np.float64

RANK = 1
SHAPE = [1]

AA_SHAPE = SHAPE + [3]
QUAT_SHAPE = SHAPE + [4]

assert len(SHAPE) == RANK

aa_tensor = tf.placeholder(DTYPE, shape=AA_SHAPE, name='aa_input')
quat_tensor = tfutil.aa_to_w_first_quaternion(aa_tensor)

assert quat_tensor.shape.as_list() == QUAT_SHAPE, '{} != {}'.format(quat_tensor.shape.as_list(), QUAT_SHAPE)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    aa = np.random.rand(*AA_SHAPE) * 2 * pi
    print(sess.run(quat_tensor, feed_dict={aa_tensor: aa}))

aa1_tensor = tf.placeholder(DTYPE, shape=AA_SHAPE, name='aa1_input')
aa2_tensor = tf.placeholder(DTYPE, shape=AA_SHAPE, name='aa2_input')
q1_tensor = tfutil.aa_to_w_first_quaternion(aa1_tensor)
q2_tensor = tfutil.aa_to_w_first_quaternion(aa2_tensor)
geodist_tensor = tfutil.axis_angle_geodesic_distance(aa1_tensor, aa2_tensor)
dot_tensor = tf.reduce_sum(tf.multiply(q1_tensor, q2_tensor), axis=-1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    aa1 = np.random.rand(*AA_SHAPE) * 2 * pi
    aa2 = np.random.rand(*AA_SHAPE) * 2 * pi
    print(sess.run(geodist_tensor, feed_dict={aa1_tensor: aa1, aa2_tensor: aa2}))
    print(sess.run(geodist_tensor, feed_dict={aa1_tensor: aa1, aa2_tensor: aa1}))
    print(sess.run(quat_tensor, feed_dict={aa_tensor: aa1}))
    print(sess.run(dot_tensor, feed_dict={aa1_tensor: aa1, aa2_tensor: aa1}))

aa_var = tf.get_variable('aa_value', shape=AA_SHAPE, dtype=DTYPE,
                         initializer=tf.ones_initializer())
loss = tf.reduce_sum(tf.abs(tfutil.axis_angle_geodesic_distance(aa_tensor, aa_var)))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-6)
opt = optimizer.minimize(loss)
grads = optimizer.compute_gradients(loss)
debug_tup = tfutil.axis_angle_geodesic_distance_debug(aa_tensor, aa_var)

# aa1 = np.array([[1, 1, 1]], dtype=DTYPE)
aa1 = np.array([[1.0000951, 0.9992925, 1.0000951]], dtype=DTYPE)

print('-----------------')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(loss, feed_dict={aa_tensor:aa1}))
    print(sess.run(aa_var))
    #exit()
    #for i in range(50000):
    for i in range(2):
        val_tup = sess.run(debug_tup, feed_dict={aa_tensor:aa1})
        print('gradients {}'.format(sess.run(grads, feed_dict={aa_tensor:aa1})))
        print(val_tup)
        print(np.multiply(val_tup[1], val_tup[2]))
        print(np.sum(np.multiply(val_tup[1], val_tup[2])))
        print(val_tup[-1][0] > 1.0)
        print(sess.run(tf.acos(val_tup[-1][0])))
        lv,_ = sess.run([loss,opt], feed_dict={aa_tensor:aa1})
        print(sess.run(aa_var))
        print('loss {}'.format(lv))
    print(aa1)
    print(sess.run(aa_var))
