import tensorflow as tf

def aa_to_w_first_quaternion(aa):
    unit_axis = tf.nn.l2_normalize(aa, axis=-1)
    half_theta = 0.5 * tf.norm(aa, axis=-1, keepdims=True)
    return tf.concat([tf.cos(half_theta), tf.sin(half_theta) * unit_axis], axis=-1)

'''
Geodesic distance between two axis angle tensors.
Input:
    aa0: a tensor to represent axis angle in last dimension
    aa1: the same shape as aa0
'''
def axis_angle_geodesic_distance(aa0, aa1, keepdims=False):
    q0 = aa_to_w_first_quaternion(aa0)
    q1 = aa_to_w_first_quaternion(aa1)
    # Notation from: https://en.wikipedia.org/wiki/Hamilton_product
    # a1,b1,c1,d1 = tf.split(q0, 1, axis=-1)
    # a2,b2,c2,d2 = tf.split(q1, 1, axis=-1)
    cos_half_theta = tf.reduce_sum(tf.multiply(q0, q1), axis=-1, keepdims=keepdims)
    return 2.0 * tf.acos(tf.clip_by_value(cos_half_theta, -1.0, 1.0))

def axis_angle_geodesic_distance_debug(aa0, aa1):
    q0 = aa_to_w_first_quaternion(aa0)
    q1 = aa_to_w_first_quaternion(aa1)
    cos_half_theta = tf.reduce_sum(tf.multiply(q0, q1), axis=-1, keepdims=False)
    distance = 2.0 * tf.acos(tf.clip_by_value(cos_half_theta, -1.0, 1.0))
    # distance = 2.0 * tf.acos(cos_half_theta)
    return distance, q0, q1, cos_half_theta
