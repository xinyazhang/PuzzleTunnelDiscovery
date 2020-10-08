import rldriver
import config
import random
import os
import tensorflow as tf
import numpy as np
import time
from datetime import datetime

def read_shapenet(filename_queue):
    """Reads and parses examples from ShapeCat data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.

        depth_images: a [width, height, number_of_depth_images] float32 Tensor with the image data
    """

    class ShapenetCatRecord(object):
      pass
    result = ShapenetCatRecord()

    # check glx_main.cc in depth_renderer-egl for input format.
    label_bytes = 4  # int32_t for shapenet
    result.n_images = 12 + 12 + 4 + 1 + 1 # See the ctor of RLDriver
    result.height = config.DEFAULT_RES
    result.width = config.DEFAULT_RES
    result.depth_bytes = 4 # FP32 = 4 bytes
    image_bytes = result.height * result.width * result.depth_bytes
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes * result.n_images

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, record_string = reader.read(filename_queue)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.decode_raw(tf.substr(record_string, 0, label_bytes), tf.int32)

    record_depth_image_string = tf.substr(record_string, label_bytes, image_bytes * result.n_images)
    depth = tf.reshape(
            tf.decode_raw(record_depth_image_string, tf.float32),
            [result.n_images, result.height, result.width, 1])

    result.image = depth

    return result

def distorted_inputs(data_dir):
    """Construct distorted input

    Args:
      data_dir: Path to the training data directory.

    Returns:
      images: Images. 4D tensor of [VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1] size.
      labels: Labels. 1D tensor of [1] size.
    """
    fnq = []
    for root,dirs,names in os.walk(data_dir):
        fnq += [os.path.join(root, name) for name in names]
    #fnq = [os.path.join(data_dir, name) for name in filter(lambda x: x.endswith('train'), os.listdir(data_dir))]
    print('Number of input files: {}'.format(len(fnq)))
    random.shuffle(fnq)
    filename_queue = tf.train.string_input_producer(fnq)
    read_input = read_shapenet(filename_queue)
    read_input.label.set_shape([1])
    print('input: {} {}'.format(read_input.image, read_input.label))

    # return read_input.image, read_input.label
    image_batch, label_batch = tf.train.batch([read_input.image, read_input.label],
            batch_size=config.BATCH_SIZE,
            num_threads=2,
            capacity=3*config.BATCH_SIZE)
    label_batch = tf.squeeze(label_batch, axis=[1])
    return image_batch, label_batch

    '''
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d ShapeNet images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    image_batch, label_batch = tf.train.shuffle_batch(
            [read_input.image, read_input.label],
            batch_size=config.BATCH_SIZE,
            num_threads=4,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    '''

class TrainCat:
    def __init__(self, global_step, ckpt_dir='./cat/ckpt', data_dir='./cat/depth_data'):
        self.ckpt_dir = ckpt_dir
        self.data_dir = data_dir
        self.images, self.labels = distorted_inputs(self.data_dir)
        self.driver = rldriver.RLDriver(['../res/alpha/env-1.2.obj', '../res/alpha/robot.obj'],
                np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                [(30.0, 12), (-30.0, 12), (0, 4), (90, 1), (-90, 1)],
                config.SV_VISCFG,
                config.SV_VISCFG,
                58,
                input_tensor=self.images)
        print('sv_colorfv = {}'.format(self.driver.sv_colorfv.shape))
        print('mv_colorfv = {}'.format(self.driver.mv_colorfv.shape))
        print('final = {}'.format(self.driver.final.shape))
        self.logits = self.driver.final
        self.global_step = global_step
        print('(sparse) labels {}'.format(self.labels.shape))
        print('logits {}'.format(self.logits.shape))
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=self.logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(self.cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        opt = tf.train.AdamOptimizer(1e-4)
        self.train_op = opt.minimize(loss=self.loss, global_step=self.global_step)

    def run(self):
        loss = self.loss
        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                  self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = config.BATCH_SIZE
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.ckpt_dir,
                hooks=[tf.train.StopAtStepHook(last_step=config.MAX_STEPS),
                    tf.train.NanTensorHook(self.loss),
                    _LoggerHook()]) as mon_sess:
            epoch = 0
            while not mon_sess.should_stop():
                mon_sess.run(self.train_op)
                epoch += 1

    def eval_once(self):
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.ckpt_dir,
                save_checkpoint_secs=None) as sess:
            step = 0
            true_count = 0
            total_sample_count = 0
            while total_sample_count < config.NUM_TO_EVALUATE:
                predictions = sess.run([self.top_k_op])
                true_count += np.sum(predictions)
                total_sample_count += config.BATCH_SIZE
                precision = float(true_count) / float(total_sample_count)
                print('%s: precision @ 1 = %.3f (%d in %d)' % (datetime.now(), precision, true_count, total_sample_count))

    def eval(self):
        self.top_k_op = tf.nn.in_top_k(self.logits, self.labels, 1)
        print('top_k_op {}'.format(self.top_k_op.shape))
        self.eval_once()
