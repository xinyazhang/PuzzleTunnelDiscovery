
import tensorflow as tf
import numpy as np

action_space_dimension = 12
actionset = [0,1,2,3]
asize = len(actionset)

def actions_to_adist_array(actions, dim):
    n = len(actions)
    adists = np.zeros(
            shape=(n, 1, dim),
            dtype=np.float32)
    for i in range(n):
        adists[i, 0, actions[i]] = 1.0
    return adists

# GT Input
action_tensor = tf.placeholder(tf.float32, shape=[1, 1, action_space_dimension], name='ActionPh')
V_tensor = tf.placeholder(tf.float32, shape=[1], name='VPh')

# Intermediate output to check their partial grads 
raw_pi = tf.get_variable("tablepi", shape=(1,1,12), dtype=tf.float32,
                         initializer=tf.zeros_initializer())
raw_value = tf.get_variable("tablevalue", shape=(1,1,1), dtype=tf.float32,
                            initializer=tf.zeros_initializer())

# Selective softmax
extractor = np.zeros((action_space_dimension, asize), dtype=np.float32)
for i, a in enumerate(actionset):
    extractor[a, i] = 1.0
adaptor = np.transpose(extractor)
compact = tf.tensordot(raw_pi, extractor, [[2], [0]])
compact_softmax = tf.nn.softmax(compact)
softmax_policy = tf.tensordot(compact_softmax, adaptor, [[2],[0]])

# build loss
flattened_value = tf.reshape(raw_value, [-1])
policy = tf.multiply(softmax_policy, action_tensor)
policy = tf.reduce_sum(policy, axis=[1,2])
log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
criticism = V_tensor - flattened_value
policy_per_sample = log_policy * tf.stop_gradient(criticism)
policy_loss = tf.reduce_sum(-policy_per_sample)
value_loss = tf.nn.l2_loss(criticism)

loss = policy_loss + value_loss
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
grad_op = optimizer.compute_gradients(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dic = {
            action_tensor : actions_to_adist_array([3], action_space_dimension),
            V_tensor: [10.0],
          }
    grad = sess.run(grad_op, feed_dict=dic)
    print("grad 1\n{}".format(grad))
    sess.run(optimizer.apply_gradients(sess.run(grad_op, feed_dict=dic)))
    dic = {
            action_tensor : actions_to_adist_array([0], action_space_dimension),
            V_tensor: [1.0],
          }
    grad = sess.run(grad_op, feed_dict=dic)
    print("grad 2\n{}".format(grad))
