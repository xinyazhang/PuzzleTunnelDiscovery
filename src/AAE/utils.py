# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import errno
from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.utils import to_categorical
import numpy as np
import random
from scipy import misc

import sys
sys.path.append(os.getcwd())
import pyosr

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def load_cifar10() :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_cifar100() :
    (train_data, train_labels), (test_data, test_labels) = cifar100.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0
    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 100)
    test_labels = to_categorical(test_labels, 100)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_fashion() :
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_tiny() :
    IMAGENET_MEAN = [123.68, 116.78, 103.94]
    path = './tiny-imagenet-200'
    num_classes = 200

    print('Loading ' + str(num_classes) + ' classes')

    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype=np.float32)
    y_train = np.zeros([num_classes * 500], dtype=np.float32)

    trainPath = path + '/train'

    print('loading training images...')

    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = misc.imread(os.path.join(sChildPath, c), mode='RGB')
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        j += 1
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype=np.float32)
    y_test = np.zeros([num_classes * 50], dtype=np.float32)

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = misc.imread(sChildPath, mode='RGB')
            if len(np.shape(X)) == 2:
                X_test[i] = np.array([X, X, X])
            else:
                X_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    print('finished loading test images : ' + str(i))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    # X_train /= 255.0
    # X_test /= 255.0

    # for i in range(3) :
    #     X_train[:, :, :, i] =  X_train[:, :, :, i] - IMAGENET_MEAN[i]
    #     X_test[:, :, :, i] = X_test[:, :, :, i] - IMAGENET_MEAN[i]

    X_train, X_test = normalize(X_train, X_test)


    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    X_train = np.transpose(X_train, [0, 3, 2, 1])
    X_test = np.transpose(X_test, [0, 3, 2, 1])

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    return X_train, y_train, X_test, y_test

def load_alpha_puzzle():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.avi = True
    res = 224
    r.pbufferWidth = res
    r.pbufferHeight = res
    r.setup()

    keys_w_first_npz = '../res/alpha/alpha-1.2.org.w-first.npz'
    env_wt_fn = '../res/alpha/alpha_env-1.2.wt.obj'
    rob_wt_fn = '../res/alpha/alpha-1.2.wt.obj'
    # rob_wt_fn = '../res/alpha/double-alpha-1.2.wt.obj'
    rob_ompl_center = np.array([16.973146438598633, 1.2278236150741577, 10.204807281494141])
    r.loadModelFromFile(env_wt_fn)
    r.loadRobotFromFile(rob_wt_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.enforceRobotCenter(rob_ompl_center)
    r.views = np.array([[0.0,0.0]], dtype=np.float32)

    return r

def load_double_alpha_puzzle():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.avi = True
    res = 224
    r.pbufferWidth = res
    r.pbufferHeight = res
    r.setup()

    keys_w_first_npz = '../res/alpha/alpha-1.2.org.w-first.npz'
    env_wt_fn = '../res/alpha/alpha_env-1.2.wt.obj'
    #rob_wt_fn = '../res/alpha/double-alpha-1.2.wt.tcp.ply'
    rob_wt_fn = '../res/alpha/double-alpha-1.2.wt.tcp3.obj'
    rob_ompl_center = np.array([16.973146438598633, 1.2278236150741577, 10.204807281494141])
    r.loadModelFromFile(env_wt_fn)
    r.loadRobotFromFile(rob_wt_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.enforceRobotCenter(rob_ompl_center)
    r.views = np.array([[0.0,0.0]], dtype=np.float32)

    return r

def load_alpha_ntr():
    pyosr.init()
    dpy = pyosr.create_display()
    glctx = pyosr.create_gl_context(dpy)
    r = pyosr.Renderer()
    r.avi = True
    res = 224
    r.pbufferWidth = res
    r.pbufferHeight = res
    r.setup()

    keys_w_first_npz = '../res/alpha/alpha-1.2.org.w-first.npz'
    env_wt_fn = '../res/alpha/alpha_env-1.2.wt.obj'
    # rob_wt_fn = '../res/alpha/alpha-1.2.wt.tcp.ply'
    rob_wt_fn = '../res/alpha/alpha-1.2.wt2.tcp.obj'
    # rob_wt_fn = '../res/alpha/double-alpha-1.2.wt.obj'
    rob_ompl_center = np.array([16.973146438598633, 1.2278236150741577, 10.204807281494141])
    r.loadModelFromFile(env_wt_fn)
    r.loadRobotFromFile(rob_wt_fn)
    r.scaleToUnit()
    r.angleModel(0.0, 0.0)
    r.default_depth = 0.0
    r.enforceRobotCenter(rob_ompl_center)
    r.views = np.array([[0.0,0.0]], dtype=np.float32)

    return r

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def get_annotations_map():
    valAnnotationsPath = './tiny-imagenet-200/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch, img_size, dataset_name):
    if dataset_name == 'mnist' :
        batch = _random_crop(batch, [img_size, img_size], 4)

    elif dataset_name =='tiny' :
        batch = _random_flip_leftright(batch)
        batch = _random_crop(batch, [img_size, img_size], 8)

    else :
        batch = _random_flip_leftright(batch)
        batch = _random_crop(batch, [img_size, img_size], 4)
    return batch

from math import sqrt,pi,sin,cos

def random_state(scale=1.0):
    tr = scale * (np.random.rand(3) - 0.5 + 0.25)
    u1,u2,u3 = np.random.rand(3)
    quat = [sqrt(1-u1)*sin(2*pi*u2),
            sqrt(1-u1)*cos(2*pi*u2),
            sqrt(u1)*sin(2*pi*u3),
            sqrt(u1)*cos(2*pi*u3)]
    part1 = np.array(tr, dtype=np.float32)
    part2 = np.array(quat, dtype=np.float32)
    part1_0 = np.array([0.0,0.0,0.0], dtype=np.float32)
    return np.concatenate((part1, part2)), np.concatenate((part1_0, part2))

def generate_minibatch(r, batch_size):
    res = r.pbufferWidth
    rgb_shape = (res,res,3)
    dep_shape = (res,res,1)
    X = []
    Y = []
    for i in range(batch_size):
        q, aq = random_state(0.5)
        r.state = q
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING)
        X.append(np.concatenate((r.mvrgb.reshape(rgb_shape), r.mvdepth.reshape(dep_shape)), axis=2))
        r.state = aq
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING|pyosr.Renderer.HAS_NTR_RENDERING)
        Y.append(np.concatenate((r.mvrgb.reshape(rgb_shape), r.mvdepth.reshape(dep_shape)), axis=2))
    return X, Y

def generate_minibatch_ntr(r, batch_size):
    res = r.pbufferWidth
    rgb_shape = (res,res,3)
    dep_shape = (res,res,1)
    X = []
    Y = []
    for i in range(batch_size):
        q, aq = random_state(0.5)
        r.state = q
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING)
        X.append(np.concatenate((r.mvrgb.reshape(rgb_shape), r.mvdepth.reshape(dep_shape)), axis=2))
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING|pyosr.Renderer.HAS_NTR_RENDERING)
        Y.append(np.concatenate((r.mvrgb.reshape(rgb_shape), r.mvdepth.reshape(dep_shape)), axis=2))
    return X, Y

def generate_minibatch_ntr2(r, batch_size):
    '''
    ntr2: only keeps green channel as indication pixels
    Note: avi should be disabled to eliminiate shadows.
    '''
    res = r.pbufferWidth
    rgb_shape = (res,res,3)
    dep_shape = (res,res,1)
    X = []
    Y = []
    for i in range(batch_size):
        r.avi = True
        q, aq = random_state(0.5)
        r.state = q
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING)
        X.append(np.concatenate((r.mvrgb.reshape(rgb_shape), r.mvdepth.reshape(dep_shape)), axis=2))
        r.avi = False
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING|pyosr.Renderer.HAS_NTR_RENDERING)
        Y.append(np.copy(r.mvrgb.reshape(rgb_shape)[:,:,1:2]))
    return X, Y

def generate_minibatch_ntr2_withuv(r, batch_size):
    '''
    ntr2: only keeps green channel as indication pixels
    Note: avi should be disabled to eliminiate shadows.
    '''
    res = r.pbufferWidth
    rgb_shape = (res,res,3)
    dep_shape = (res,res,1)
    uv_shape = (res,res,2)
    X = []
    Y = []
    UV = []
    for i in range(batch_size):
        r.avi = True
        q, aq = random_state(0.5)
        r.state = q
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING)
        X.append(np.concatenate((r.mvrgb.reshape(rgb_shape), r.mvdepth.reshape(dep_shape)), axis=2))
        r.avi = False
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING|pyosr.Renderer.HAS_NTR_RENDERING)
        Y.append(np.copy(r.mvrgb.reshape(rgb_shape)[:,:,1:2]))
        UV.append(np.copy(r.mvuv.reshape(uv_shape)))
    return X, Y, UV

def generate_minibatch_ntr3(r, batch_size):
    '''
    GT Without AVI, we just want masks
    '''
    res = r.pbufferWidth
    rgb_shape = (res,res,3)
    dep_shape = (res,res,1)
    X = []
    Y = []
    for i in range(batch_size):
        r.avi = True
        q, aq = random_state(0.5)
        r.state = q
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING)
        X.append(np.concatenate((r.mvrgb.reshape(rgb_shape), r.mvdepth.reshape(dep_shape)), axis=2))
        r.avi = False
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING|pyosr.Renderer.HAS_NTR_RENDERING)
        Y.append(np.copy(r.mvrgb.reshape(rgb_shape)))
    return X, Y

def feedback_labling(input_image, expect_image, predict_image):
    red_in = input_image[:,:,0]
    blue_in = input_image[:,:,2]
    pred = predict_image[:,:,0]

    return np.stack([red_in, pred, blue_in], axis=2)

'''
Return a 2xD np.array `bb` to indicate the bounding box
  bb[0]: smallest coefficients
  bb[1]: largest coefficients

  both index tuples are inclusive
'''
def calculate_bb_2d(img):
    nz = np.nonzero(img)
    nz_x = nz[0]
    nz_y = nz[1]
    if nz_x.size == 0 or nz_y.size == 0:
        return None
    return np.array([[min(nz_x), min(nz_y)], [max(nz_x), max(nz_y)]], dtype=np.int32)

def intersect_bb(bb1, bb2):
    if bb1 is None or bb2 is None:
        return False # BB intersect NOT A BB is False (No Intersection)
    x_left = max(bb1[0,0], bb2[0,0])
    y_top = max(bb1[0,1], bb2[0,1])
    x_right = min(bb1[1,0], bb2[1,0])
    y_bottom = min(bb1[1,1], bb2[1,1])

    if x_right < x_left or y_bottom < y_top:
        return False
    return True

def random_bb(res):
    x = np.sort(np.random.randint(res, size=(2)))
    y = np.sort(np.random.randint(res, size=(2)))
    return np.array([[x[0],y[0]],[x[1],y[1]]], dtype=np.int32)

def patch_bb(img, bb, default=0.0):
    mins = bb[0]
    maxs = bb[1]
    # +1 is necessary since numpy's broadcasting is [inclusive, exclusive)
    img[mins[0]:maxs[0]+1, mins[1]:maxs[1]+1,:] = default
    return img

def random_blackout(rgb, green, dep, default_depth):
    red = rgb[:,:,0:1]
    red_bb = calculate_bb_2d(red)
    green_bb = calculate_bb_2d(green)
    res = int(red.shape[0])
    while True:
        bb = random_bb(res)
        '''
        bb must cover red region, and must not cover green region
        '''
        if intersect_bb(bb, green_bb):
            continue
        if not intersect_bb(bb, red_bb):
            continue
        break
    return patch_bb(rgb, bb), patch_bb(dep, bb, default_depth)

def generate_minibatch_ntr4(r, batch_size):
    '''
    ntr2: only keeps green channel as indication pixels
    Note: avi should be disabled to eliminiate shadows.
    '''
    res = r.pbufferWidth
    rgb_shape = (res,res,3)
    dep_shape = (res,res,1)
    X = []
    Y = []
    for i in range(batch_size):
        r.avi = True
        q, aq = random_state(0.5)
        r.state = q
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING)
        red = np.copy(r.mvrgb.reshape(rgb_shape))
        dep = np.copy(r.mvdepth.reshape(dep_shape))
        r.avi = False
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING|pyosr.Renderer.HAS_NTR_RENDERING)
        greend = np.copy(r.mvrgb.reshape(rgb_shape))
        patched_rgb, patched_dep = random_blackout(red, greend[:,:,1:2], dep, r.default_depth)
        X.append(np.concatenate((patched_rgb, patched_dep), axis=2))
        r.avi = False
        r.render_mvrgbd(pyosr.Renderer.NO_SCENE_RENDERING|pyosr.Renderer.HAS_NTR_RENDERING)
        Y.append(np.copy(r.mvrgb.reshape(rgb_shape)[:,:,1:2]))
    return X, Y

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
