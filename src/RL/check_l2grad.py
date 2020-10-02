# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

import tensorflow as tf
import numpy as np

K = 4
N = 512
GT = np.array([[float(i) for i in range(K)]])
V = tf.Variable(np.zeros(shape=(K)))

loss = tf.nn.l2_loss(V-GT)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(N):
        print(sess.run(V))
        sess.run(train_op)
    print(sess.run(V))
