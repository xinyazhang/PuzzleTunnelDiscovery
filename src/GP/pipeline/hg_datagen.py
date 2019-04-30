# -*- coding: utf-8 -*-
import numpy as np
# import cv2 # Legacy, not used
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm

import sys
sys.path.append(os.getcwd())
from .import image_augmentation as aug
from .import parse_ompl
import pyosr

# Dataset paths
# import aniconf12_2
# import aniconf10
# import dualconf_tiny
# import dualconf
# import dualconf_g2
# import dualconf_g4

class OsrDataSet(object):

    def __init__(self, env, rob, center=None, res=224, flat_surface=False):
        pyosr.init()
        self.dpy = pyosr.create_display()
        self.glctx = pyosr.create_gl_context(self.dpy)

        r = pyosr.Renderer()
        r.avi = True
        r.flat_surface = flat_surface
        r.pbufferWidth = res
        r.pbufferHeight = res
        r.setup()

        r.loadModelFromFile(env)
        r.loadRobotFromFile(rob)
        r.scaleToUnit()
        r.angleModel(0.0, 0.0)
        r.default_depth=0.0
        if center is not None:
            r.enforceRobotCenter(center)
        r.views = np.array([[0.0,0.0]], dtype=np.float32)

        self.r = r
        self.res = res
        self.rgb_shape = (res,res,3)
        self.dep_shape = (res,res,1)

    def render_rgbd(self, q, flags=0):
        r = self.r
        r.state = q
        r.render_mvrgbd(flags)
        tup = (r.mvrgb.reshape(self.rgb_shape), r.mvdepth.reshape(self.dep_shape))
        return np.concatenate(tup,
                              axis=2)

from math import sqrt,pi,sin,cos

def random_state(scale=1.0):
    tr = np.random.uniform(low=-1.0, high=1.0, size=(3))
    tr *= scale
    #tr = [1.0,0.0,0] # Debugging
    #tr = [1.5,1.5,0]
    u1,u2,u3 = np.random.rand(3)
    quat = [sqrt(1-u1)*sin(2*pi*u2),
            sqrt(1-u1)*cos(2*pi*u2),
            sqrt(u1)*sin(2*pi*u3),
            sqrt(u1)*cos(2*pi*u3)]
    # quat = [1.0, 0.0, 0.0, 0.0] # Debugging
    part1 = np.array(tr, dtype=np.float32)
    part2 = np.array(quat, dtype=np.float32)
    part1_0 = np.array([0.0,0.0,0.0], dtype=np.float32)
    return np.concatenate((part1, part2)), np.concatenate((part1_0, part2))

class NarrowTunnelRegionDataSet(OsrDataSet):
    c_dim = 4
    d_dim = 1

    '''
    Arguments:
        render_flag: choose which geometry to render
    '''
    def __init__(self, rob, env, render_flag, center=None, q_range=1.0, res=256, patch_size=32, aug_patch=False, aug_scaling=0.0, aug_dict={}, flat_surface=False):
        super(NarrowTunnelRegionDataSet, self).__init__(rob=rob,
                                                        env=env,
                                                        center=center,
                                                        res=res,
                                                        flat_surface=flat_surface)
        self.c_dim = 4
        self.d_dim = 1
        self.render_flag = render_flag
        self.patch_size = np.array([patch_size, patch_size], dtype=np.int32)
        self.aug_patch = aug_patch
        self.aug_dict = dict(aug_dict)
        self.aug_scaling = aug_scaling
        self.q_range = q_range

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train', emit_gt=False):
        is_training = True if sample_set == 'train' else False
        '''
        _aux_generator is the only interface used by HourglassModel class
        This generator should yield image, gt and weight (???)

        image: (batch_size, 256, 256, 3)
        '''
        r = self.r
        aug_dict = self.aug_dict
        aug_patch = self.aug_patch
        aug_suppress_hot = aug_dict['suppress_hot'] if 'suppress_hot' in aug_dict else 0.0
        aug_red_noise = aug_dict['red_noise'] if 'red_noise' in aug_dict else 0.0
        aug_suppress_cold = aug_dict['suppress_cold'] if 'suppress_cold' in aug_dict else 0.0
        aug_scaling = self.aug_scaling
        while True:
            train_img = np.zeros((batch_size, self.res, self.res, self.c_dim), dtype = np.float32)
            if emit_gt:
                gt_img = np.zeros((batch_size, self.res, self.res, 3), dtype = np.float32)
            if is_training:
                train_gtmap = np.zeros((batch_size, stacks, self.res//4, self.res//4, self.d_dim), np.float32)
            else:
                self.r.uv_feedback = True
                uv_map = np.zeros((batch_size, self.res, self.res, 2), np.float32)
            train_weights = None
            for i in range(batch_size):
                if np.random.random() < aug_scaling:
                    # Only downscale
                    s = np.random.uniform(low=0.5, high=1.0, size=(3))
                    # Half a chance to upscale
                    if np.random.random() < 0.5:
                        s = 1.0 / s
                    r.final_scaling = s
                else:
                    r.final_scaling = np.array([1.0, 1.0, 1.0])
                r.avi = True
                q,aq = random_state(self.q_range)
                train_img[i] = self.render_rgbd(q, self.render_flag)
                r.avi = False
                r.render_mvrgbd(self.render_flag|pyosr.Renderer.HAS_NTR_RENDERING)
                if is_training:
                    rgbd = r.mvrgb.reshape(self.rgb_shape)
                    if emit_gt:
                        gt_img[i] = rgbd[...,0:3]
                    hm = rgbd[:,:,1:2] # The green region
                    hm = hm.reshape(self.res//4,4,self.res//4,4,self.d_dim).mean(axis=(1,3)) # Downscale to 64x64
                    hm = np.expand_dims(hm, axis=0) # reshape to [1, 64, 64]
                    train_gtmap[i] = np.repeat(hm, stacks, axis=0) # reshape to [4,64,64] thru duplication
                    '''
                    Randomly patch non-NTR retion
                    '''
                    if aug_patch:
                        rnd = np.random.random()
                        aug_func = None
                        # print("rnd {}".format(rnd))
                        if rnd < aug_suppress_hot:
                            # Remove the hot region
                            # print("aug_suppress_hot")
                            patch_tl, patch_size = aug.patch_finder_hot(heatmap=rgbd[:,:,1], margin_pix=16)
                            aug_func = aug.patch_rgb
                        elif rnd < aug_suppress_hot + aug_red_noise:
                            patch_tl, patch_size = aug.patch_finder_hot(heatmap=rgbd[:,:,1], margin_pix=64)
                            # print("aug_red_noise {} {}".format(patch_tl, patch_size))
                            aug_func = aug.red_noise
                        elif rnd < aug_suppress_hot + aug_red_noise + aug_suppress_cold:
                            patch_tl, patch_size = aug.patch_finder_hot(heatmap=rgbd[:,:,1], margin_pix=64)
                            aug_func = aug.focus
                        else:
                            # print("aug_patch")
                            patch_tl = aug.patch_finder_1(coldmap=rgbd[:,:,0], heatmap=rgbd[:,:,1], patch_size=self.patch_size)
                            patch_size = self.patch_size
                            aug_func = aug.patch_rgb
                        if patch_tl is None: # Cannot find a patch, cancel
                            aug_func = None
                        if aug_func is not None:
                            train_img[i] = aug_func(train_img[i], patch_tl, patch_size)
                            if emit_gt:
                                aug.dim_rgb(gt_img[i], patch_tl, patch_size)
                else:
                    uv_map[i] = r.mvuv.reshape((self.res, self.res, 2))
            if is_training:
                to_yield = [train_img, train_gtmap, train_weights]
            else:
                to_yield = [train_img, uv_map, train_weights]
            if emit_gt:
                to_yield.append(gt_img)
            yield to_yield

def create_dataset(ompl_cfg, geo_type, res=256, aug_patch=True, aug_scaling=1.0, aug_dict={}):
    cfg, _ = parse_ompl.parse_simple(ompl_cfg)
    rob = cfg.rob_fn if geo_type == 'rob' else cfg.env_fn
    env = cfg.env_fn
    render_flag = pyosr.Renderer.NO_SCENE_RENDERING
    patch_size=64
    return NarrowTunnelRegionDataSet(rob=rob, env=env,
                                     render_flag=render_flag,
                                     res=res,
                                     patch_size=patch_size,
                                     center=None,
                                     aug_patch=aug_patch,
                                     aug_dict=aug_dict,
                                     aug_scaling=aug_scaling)
