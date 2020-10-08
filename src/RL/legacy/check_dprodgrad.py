# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import tensorflow as tf
import numpy as np

K = 4
A = 2
N = 1024
GT = np.array([float(i+1) for i in range(K)])
print(GT)
# exit()
V = tf.Variable(np.zeros(shape=(K,A)))
p_a = tf.placeholder(dtype=tf.float64, shape=(K,A))
adist = np.zeros(shape=(K,A))
for i in range(K):
    adist[i, i % A] = 1.0
print('adist {}'.format(adist))

loss = tf.nn.l2_loss(tf.reduce_sum(tf.multiply(V,p_a), axis=[1])-GT)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('p_a {}'.format(sess.run(p_a, feed_dict={p_a : adist})))
    print('V * p_a {}'.format(sess.run(tf.multiply(V,p_a), feed_dict={p_a : adist})))
    for i in range(N):
        print(sess.run(V))
        sess.run(train_op, feed_dict={p_a : adist})
    print(sess.run(V))
