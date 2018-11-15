import time
from ops import *
from utils import *
from scipy.misc import imsave

class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200

        self.is_generator_dataset = False
        self.test_only = False
        self.image_post_processing_func = None
        self.advanced_testing = None

        if self.dataset_name == 'alpha_puzzle' :
            self.is_generator_dataset = True
            self.r = load_alpha_puzzle()
            self.img_size = self.r.pbufferWidth
            self.c_dim = 4
            self.label_dim = None
            self.generator = generate_minibatch

        if self.dataset_name == 'double_alpha_puzzle' :
            self.is_generator_dataset = True
            self.r = load_double_alpha_puzzle()
            self.img_size = self.r.pbufferWidth
            self.c_dim = 4
            self.label_dim = None
            self.generator = generate_minibatch

        if self.dataset_name == 'alpha_ntr' :
            self.is_generator_dataset = True
            self.r = load_alpha_ntr()
            self.img_size = self.r.pbufferWidth
            self.c_dim = 4
            self.label_dim = None
            self.generator = generate_minibatch_ntr

        if self.dataset_name == 'alpha_ntr2' :
            self.is_generator_dataset = True
            self.r = load_alpha_ntr()
            self.img_size = self.r.pbufferWidth
            self.c_dim = 4
            self.d_dim = 1
            self.label_dim = None
            self.generator = generate_minibatch_ntr2
            self.image_post_processing_func = feedback_labling

        if self.dataset_name == 'alpha_ntr4' :
            self.is_generator_dataset = True
            self.r = load_alpha_ntr()
            self.img_size = self.r.pbufferWidth
            self.c_dim = 4
            self.d_dim = 1
            self.label_dim = None
            self.generator = generate_minibatch_ntr4
            self.image_post_processing_func = feedback_labling

        if self.dataset_name == 'double_alpha_ntr2' :
            self.is_generator_dataset = True
            self.r = load_double_alpha_puzzle()
            self.img_size = self.r.pbufferWidth
            self.r.uv_feedback = True
            self.c_dim = 4
            self.d_dim = 1
            self.label_dim = None
            self.generator = generate_minibatch_ntr2
            self.image_post_processing_func = feedback_labling
            self.advanced_testing = self._test_double_alpha_ntr2

        if self.dataset_name == 'alpha_ntr2_atex' :
            self.is_generator_dataset = True
            self.r = load_alpha_ntr()
            self.img_size = self.r.pbufferWidth
            self.r.uv_feedback = True
            self.c_dim = 4
            self.d_dim = 1
            self.label_dim = None
            self.generator = generate_minibatch_ntr2
            self.advanced_testing = self._test_double_alpha_ntr2

        if self.dataset_name == 'alpha_ntr3' :
            self.is_generator_dataset = True
            self.r = load_alpha_ntr()
            self.img_size = self.r.pbufferWidth
            self.c_dim = 4
            self.d_dim = 3
            self.label_dim = None
            self.generator = generate_minibatch_ntr3

        if self.dataset_name == 'double_alpha_ntr3' :
            self.is_generator_dataset = True
            self.r = load_double_alpha_puzzle()
            self.img_size = self.r.pbufferWidth
            self.c_dim = 4
            self.d_dim = 3
            self.label_dim = None
            self.generator = generate_minibatch_ntr3
            self.test_only = True

        if not hasattr(self, 'd_dim'):
            self.d_dim = self.c_dim

        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.bootstrap_dir = args.bootstrap_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        if not self.is_generator_dataset:
            self.iteration = len(self.train_x) // self.batch_size
        else:
            assert args.iteration > 0, 'generator dataset requires --iteration'
            self.iteration = args.iteration

        self.init_lr = args.lr
        self.is_ae = args.ae
        self.is_aae = args.aae
        self.is_hg = args.hourglass

        if self.is_ae and not self.is_aae:
            self.train_y = self.train_x
            self.test_y = self.test_x
            self.model_name += 'AE'

        if self.is_hg:
            self.model_name += 'Hg'

        self.out_dir = args.out


    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
                residual_decoder_block = resdecblock
            else :
                residual_block = bottle_resblock
                residual_decoder_block = bottle_resdecblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')
            hg_out_0 = x

            ########################################################################################################

            hg_out_1 = []

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock_0_' + str(i))
                hg_out_1 = [x] + hg_out_1

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock_1_0')

            hg_out_2 = [x]
            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock_1_' + str(i))
                hg_out_2 = [x] + hg_out_2

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock_2_0')

            hg_out_3 = [x]
            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))
                hg_out_3 = [x] + hg_out_3

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            hg_out_4 = [x]
            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))
                hg_out_4 = [x] + hg_out_4

            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            if not self.is_ae and not self.is_hg:
                x = global_avg_pooling(x)
                x = fully_conneted(x, units=self.label_dim, scope='logit')
                return x
            ###################
            # Decoder
            ###################
            print("Bottleneck shape before {}".format(x.shape))
            pre_shape = x.shape.as_list()
            x = tf.layers.flatten(x)
            print("flatten {}".format(x.shape))
            pre_size = int(x.shape[-1])
            x = tf.layers.dense(x, 128)
            print("dense1 {}".format(x.shape))
            x = tf.layers.dense(x, pre_size)
            print("dense2 {}".format(x.shape))
            pre_shape[0] = -1
            x = tf.reshape(x, pre_shape)
            print("Bottleneck shape after {}".format(x.shape))

            ########################################################################################################

            for i in range(1, residual_list[3]) :
                x = x + hg_out_4[i-1] if self.is_hg else x
                x = residual_decoder_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resdecblock_3_' + str(i))

            x = x + hg_out_4[-1] if self.is_hg else x
            x = residual_decoder_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resdecblock_3_0')

            ########################################################################################################

            for i in range(1, residual_list[2]) :
                x = x + hg_out_3[i-1] if self.is_hg else x
                x = residual_decoder_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resdecblock_2_' + str(i))

            x = x + hg_out_3[-1] if self.is_hg else x
            x = residual_decoder_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resdecblock_2_0')

            ########################################################################################################

            for i in range(1, residual_list[1]) :
                x = x + hg_out_2[i-1] if self.is_hg else x
                x = residual_decoder_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resdecblock_1_' + str(i))

            x = x + hg_out_2[-1] if self.is_hg else x
            x = residual_decoder_block(x, channels=ch, is_training=is_training, downsample=True, scope='resdecblock_1_0')

            ########################################################################################################

            for i in range(residual_list[0]) :
                x = x + hg_out_1[i] if self.is_hg else x
                x = residual_decoder_block(x, channels=ch, is_training=is_training, downsample=False, scope='resdecblock_0_' + str(i))

            x = x + hg_out_0 if self.is_hg else x
            x = deconv(x, channels=self.d_dim, kernel=3, stride=1, scope='deconv')
            print('AE output shape {}'.format(x.shape))

            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """

        if not self.is_ae and not self.is_hg:
            self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
            self.test_labels = tf.placeholder(tf.float32, [len(self.test_y), self.label_dim], name='test_labels')
            self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')
            self.test_inptus = tf.placeholder(tf.float32, [len(self.test_x), self.img_size, self.img_size, self.c_dim], name='test_inputs')
        else:
            self.train_inptus = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.c_dim], name='train_inputs')
            self.test_inptus = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.c_dim], name='test_inputs')
            self.train_labels = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.d_dim], name='train_labels')
            self.test_labels = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.d_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)
        self.test_logits = self.network(self.test_inptus, is_training=False, reuse=True)

        if self.is_ae or self.is_hg:
            LOSS = mse_loss
        else:
            LOSS = classification_loss
        self.train_loss, self.train_accuracy = LOSS(logit=self.train_logits, label=self.train_labels)
        self.test_loss, self.test_accuracy = LOSS(logit=self.test_logits, label=self.test_labels)

        """ Training """
        # self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        train_summary_list = [self.summary_train_loss, self.summary_train_accuracy]
        test_summary_list = [self.summary_test_loss, self.summary_test_accuracy]

        if self.is_ae or self.is_hg:
            self.summary_train_ae = tf.summary.image('train_ae', self.train_logits)
            self.summary_test_ae = tf.summary.image('test_ae', self.test_logits)

            train_summary_list.append(self.summary_train_ae)
            test_summary_list.append(self.summary_test_ae)

        self.train_summary = tf.summary.merge(train_summary_list)
        self.test_summary = tf.summary.merge(test_summary_list)

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        assert not self.test_only, "dataset {} is test only".format(self.dataset_name)
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
            '''
            Bootstrap from another pre-trained network
            Note: do not care the returns
            '''
            self.load(self.bootstrap_dir, fullpath=True)

        # loop for epoch
        start_time = time.time()
        if not self.is_generator_dataset:
            test_nbatch = len(self.test_y) // self.batch_size
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                if not self.is_generator_dataset:
                    batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                    batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]
                else:
                    batch_x, batch_y = self.generator(self.r, self.batch_size)

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                if not self.is_generator_dataset:
                    sel = np.random.randint(test_nbatch)
                    test_feed_dict = {
                        self.test_inptus : self.test_x[sel*test_nbatch:(sel+1)*test_nbatch],
                        self.test_labels : self.test_y[sel*test_nbatch:(sel+1)*test_nbatch]
                    }
                else:
                    test_x, test_y = self.generator(self.r, self.batch_size)
                    test_feed_dict = {
                        self.test_inptus : test_x,
                        self.test_labels : test_y
                    }

                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = self.sess.run(
                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if not self.is_hg:
            return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)
        else:
            return "{}{}_{}_hg".format(self.model_name, self.res_n, self.dataset_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir, fullpath=False):
        print(" [*] Reading checkpoints...")
        if not fullpath:
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        if self.bootstrap_dir is not None:
            could_load, checkpoint_counter = self.load(self.bootstrap_dir, fullpath=True)
        else:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if not self.is_generator_dataset:
            test_feed_dict = {
                self.test_inptus: self.test_x,
                self.test_labels: self.test_y
            }

            test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
            print("test_accuracy: {}".format(test_accuracy))
        else:
            assert could_load
            assert self.out_dir is not None
            if self.advanced_testing is not None:
                self.advanced_testing()
                return
            index = 0
            for i in range(self.iteration):
                batch_x, batch_y = self.generator(self.r, self.batch_size)
                test_feed_dict = {
                    self.test_inptus : batch_x,
                    self.test_labels : batch_y
                }
                test_loss, test_y = self.sess.run(
                    [self.test_loss, self.test_logits], feed_dict=test_feed_dict)
                # Input Image, Expected Image and Output Image
                for ii,ei,oi in zip(batch_x, batch_y, test_y):
                    self._imsave(ii, ei, oi)
                    index += 1

    def _imsave(self, index, ii, ei, oi):
        ft = '{}/{}-{}.png'.format(self.out_dir, index, '{}')
        imsave(ft.format('ii'), ii[:,:,:3])
        if self.d_dim == 3:
            imsave(ft.format('ei'), ei[:,:,:3])
            if oi is not None:
                imsave(ft.format('oi'), oi[:,:,:3])
        elif self.d_dim == 1:
            imsave(ft.format('ei'), ei[:,:,0])
            if oi is not None:
                imsave(ft.format('oi'), oi[:,:,0])
        else:
            assert False, "Unrecognized d_dim {}".format(d_dim)
        if self.image_post_processing_func is not None and oi is not None:
            imsave(ft.format('pi'), self.image_post_processing_func(ii, ei, oi))

    def peek(self):
        index = 0
        for i in range(self.iteration):
            batch_x, batch_y = self.generator(self.r, self.batch_size)
            for ii,ei in zip(batch_x, batch_y):
                self._imsave(index, ii, ei, None)
                index += 1

    def _test_double_alpha_ntr2(self):
        tres = 2048
        atex = np.zeros(shape=(tres,tres), dtype=np.float32) # accumulator texture
        index = 0
        for i in range(self.iteration):
            batch_x, batch_y, batch_uv = generate_minibatch_ntr2_withuv(self.r, self.batch_size)
            test_feed_dict = {
                self.test_inptus : batch_x,
                self.test_labels : batch_y
            }
            test_loss, test_y = self.sess.run([self.test_loss, self.test_logits],
                                              feed_dict=test_feed_dict)
            for ii,uvi,labeli in zip(batch_x, batch_uv, test_y):
                # np.clip(labeli, 0.0, 1.0, out=labeli)
                labeli = np.reshape(labeli, (self.img_size,self.img_size))
                nz = np.nonzero(labeli)
                scores = labeli[nz]
                uvs = uvi[nz]
                us = 1.0 - uvs[:,1]
                us = np.array(tres * us, dtype=int)
                vs = uvs[:,0]
                vs = np.array(tres * vs, dtype=int)
                for iu,iv,s in zip(us,vs,scores):
                    if iu < 0 or iu >= tres or iv < 0 or iv > tres:
                        continue
                    atex[iu,iv] += s
        np.savez('{}/atex.npz'.format(self.out_dir), ATEX=atex)
        natex = atex / np.amax(atex)
        imsave('{}/atex.png'.format(self.out_dir), natex)
