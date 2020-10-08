
import tensorflow as tf
import numpy as np
import vision

#action_space_dimension = 12
action_space_dimension = 4
actionset = [0,1,2,3]
asize = len(actionset)
FEAT=256
HIDDEN=1024
# BSize=None
BSize=54

def actions_to_adist_array(actions, dim):
    n = len(actions)
    adists = np.zeros(
            shape=(n, 1, dim),
            dtype=np.float32)
    for i in range(n):
        adists[i, 0, actions[i]] = 1.0
    return adists

# GT Input
action_tensor = tf.placeholder(tf.float32, shape=[BSize, 1, action_space_dimension], name='ActionPh')
V_tensor = tf.placeholder(tf.float32, shape=[BSize], name='VPh')
# Feature Vector
featvec = tf.placeholder(tf.float32, shape=[BSize, 1, FEAT], name='ActionPh')

# pi and value net
pi_net = vision.ConvApplier(None, [HIDDEN, HIDDEN, action_space_dimension], 'PiNet', elu=True,
                initialized_as_zero=False,
                nolu_at_final=True,
                batch_normalization=None)
_, raw_pi = pi_net.infer(featvec)
print("raw_pi {}".format(raw_pi.shape))
value_net = vision.ConvApplier(None, [HIDDEN, HIDDEN, 1], 'ValueNet', elu=True,
                initialized_as_zero=False,
                nolu_at_final=True,
                batch_normalization=None)
_, raw_value = value_net.infer(featvec)
print("raw_value {}".format(raw_value.shape))

if len(actionset) != action_space_dimension:
    # Selective softmax
    extractor = np.zeros((action_space_dimension, asize), dtype=np.float32)
    for i, a in enumerate(actionset):
        extractor[a, i] = 1.0
    adaptor = np.transpose(extractor)
    compact = tf.tensordot(raw_pi, extractor, [[2], [0]])
    compact_softmax = tf.nn.softmax(compact)
    softmax_policy = tf.tensordot(compact_softmax, adaptor, [[2],[0]])
else:
    compact = raw_pi
    softmax_policy = tf.nn.softmax(raw_pi)

# build loss
flattened_value = tf.reshape(raw_value, [-1])
print("raw_value {}".format(raw_value.shape))
print("flattened_value {}".format(flattened_value.shape))
policy = tf.multiply(softmax_policy, action_tensor)
policy = tf.reduce_sum(policy, axis=[1,2])
log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
criticism = V_tensor - flattened_value
policy_per_sample = log_policy * tf.stop_gradient(criticism)
policy_loss = tf.reduce_sum(-policy_per_sample)
value_loss = tf.nn.l2_loss(criticism)

policy_loss *= 1e-3

loss = policy_loss + value_loss
#loss = value_loss
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
grad_op = optimizer.compute_gradients(loss)
train_op = optimizer.minimize(loss)

# Load training data
TFILE='fake3d-6-msa-fv.npz'
d = np.load(TFILE)
FV_IN = d['TRAJ_FV'].reshape(55,1,256)
A_IN = d['TRAJ_A']
V = 10.0
V_IN = [V]
for i in range(len(A_IN)):
    # V *= 0.9
    V -= 0.2
    V_IN.append(V)

V_IN.reverse()

A_DIST = actions_to_adist_array(A_IN, action_space_dimension)

dic = {
        V_tensor : V_IN[:-1],
        featvec : FV_IN[:-1],
        action_tensor : A_DIST
      }

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    TTL = 1024*1024
    for i in range(TTL):
        _,pl,vl = sess.run([train_op, policy_loss, value_loss], feed_dict=dic)
        print("iter {} policy_loss {} value_loss {}".format(i, pl, vl))
        if (i+1) % 100 == 0 or i+1 == TTL:
            v,c,p = sess.run([flattened_value,criticism,compact], feed_dict=dic)
            print("\tval {}".format(v))
            print("\tcri {}".format(c))
            print("\tcpi {}".format(p))

