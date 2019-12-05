# -*- coding: utf-8 -*-
import numpy as np
# import cv2 # Legacy, not used
import os
import pathlib
# import matplotlib.pyplot as plt
import random
import time
# from skimage import transform
import scipy.misc as scm

import sys
sys.path.append(os.getcwd())
from . import image_augmentation as aug
from . import parse_ompl
import pyosr
from . import util

class OsrDataSet(object):
    dpy = None

    def __init__(self, env, rob, rob_texfn='', center=None, res=256,
                 flat_surface=False, gen_surface_normal=False):
        if self.dpy is None:
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
        if rob_texfn:
            r.loadRobotTextureImage(rob_texfn)
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
        self.gen_surface_normal = gen_surface_normal

    def render_rgbd(self, q, flags=0):
        r = self.r
        r.state = q
        if self.gen_surface_normal:
            r.render_mvrgbd(flags | pyosr.Renderer.NORMAL_RENDERING)
            tup = (r.mvrgb.reshape(self.rgb_shape)[:,:,0:1],
                   r.mvnormal.reshape(self.rgb_shape),
                   r.mvdepth.reshape(self.dep_shape))
        else:
            r.render_mvrgbd(flags)
            tup = (r.mvrgb.reshape(self.rgb_shape), r.mvdepth.reshape(self.dep_shape))
        return np.concatenate(tup, axis=2)

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

class MultiPuzzleDataSet(object):

    '''
    Arguments:
        render_flag: choose which geometry to render
    '''
    def __init__(self, render_flag, q_range=1.0, res=256,
                 patch_size=32, aug_patch=False, aug_scaling=0.0, aug_dict={},
                 flat_surface=False, gen_surface_normal=False,
                 weighted_loss=False, multichannel=None, use_fp16=False):
        self.gen_surface_normal = gen_surface_normal
        self.c_dim = 1 + 3 + 1 if gen_surface_normal else 4
        self.render_flag = render_flag
        self.res = res
        self.patch_size = np.array([patch_size, patch_size], dtype=np.int32)
        self.aug_patch = aug_patch
        self.aug_dict = dict(aug_dict)
        self.aug_scaling = aug_scaling
        self.q_range = q_range
        self.renders = []
        self.weighted_loss = weighted_loss
        self.multichannel = multichannel
        self.fp_type = np.float16 if use_fp16 else np.float32

    @property
    def d_dim(self):
        return self.multichannel if self.multichannel is not None else 1

    @property
    def number_of_geometries(self):
        return len(self.renders)

    def add_puzzle(self, rob, env, rob_texfn, flat_surface=False):
        ds = OsrDataSet(rob=rob, env=env, rob_texfn=rob_texfn,
                        center=None, res=self.res,
                        flat_surface=flat_surface,
                        gen_surface_normal=self.gen_surface_normal)
        self.renders.append(ds)

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train'):
        is_training = True if sample_set == 'train' else False
        '''
        _aux_generator is the only interface used by HourglassModel class
        This generator should yield image, gt and weight (???)

        image: (batch_size, 256, 256, 3)
        '''
        aug_scaling = self.aug_scaling
        while True:
            train_img = np.zeros((batch_size, self.res, self.res, self.c_dim), dtype = self.fp_type)
            if is_training:
                train_gtmap = np.zeros((batch_size, stacks, self.res, self.res, self.d_dim), self.fp_type)
            else:
                uv_map = np.zeros((batch_size, self.res, self.res, 2), self.fp_type)
            if self.weighted_loss or self.multichannel is not None:
                train_weights = np.zeros((batch_size, self.d_dim), self.fp_type)
            else:
                train_weights = None
            for i in range(batch_size):
                subds_index = random.randint(0, self.number_of_geometries - 1)
                subds = self.renders[subds_index]
                # subds = random.choice(self.renders)
                r = subds.r
                if self.multichannel is not None:
                    train_weights[i, subds_index] = 1
                """
                Scaling
                """
                if np.random.random() < aug_scaling:
                    # Only downscale
                    s = np.random.uniform(low=0.5, high=1.0, size=(3))
                    # Half a chance to upscale
                    if np.random.random() < 0.5:
                        s = 1.0 / s
                    r.final_scaling = s
                else:
                    r.final_scaling = np.array([1.0, 1.0, 1.0])
                """
                Render The Input Image
                """
                r.avi = True
                q, aq = random_state(self.q_range)
                train_img[i] = subds.render_rgbd(q, self.render_flag)
                r.avi = False
                """
                Render the UV coordinates and HeatMap (NTR, i.e., Narrow Tunnel Region)
                """
                if is_training:
                    r.render_mvrgbd(self.render_flag|pyosr.Renderer.HAS_NTR_RENDERING)
                    rgbd = r.mvrgb.reshape(subds.rgb_shape)
                    hm = np.copy(rgbd[:,:,1:2]) # Extract HeatMap from green region
                    if self.weighted_loss:
                        train_weights[i,:] = np.sum(hm, axis=(0,1)) + 0.5 # Ensure the weight is non-zero
                    if self.aug_patch:
                        aug.augment_image(rgbd, self.aug_dict, i, train_img, hm,
                                          random_patch_size=self.patch_size)
                    hm = np.expand_dims(hm, axis=0) # reshape to [1, 256, 256, 1]
                    aug.flip_images(i, train_img, 0, hm)

                    if self.multichannel is not None:
                        train_gtmap[i,:,:,:,subds_index:subds_index+1] = np.repeat(hm, stacks, axis=0) # each hourglass needs an output
                    else:
                        train_gtmap[i] = np.repeat(hm, stacks, axis=0) # each hourglass needs an output
                else:
                    r.render_mvrgbd(self.render_flag|pyosr.Renderer.UV_FEEDBACK)
                    uv_map[i] = r.mvuv.reshape((self.res, self.res, 2))
                    aug.flip_images(i, train_img, i, uv_map)
            if is_training:
                to_yield = [train_img, train_gtmap, train_weights]
            else:
                to_yield = [train_img, uv_map, train_weights]
            yield to_yield

def create_multidataset(ompl_cfgs, geo_type, res=256,
                        aug_patch=True, aug_scaling=1.0, aug_dict={},
                        gen_surface_normal=False, weighted_loss=False,
                        multichannel=None, params={}
                        ):
    render_flag = pyosr.Renderer.NO_SCENE_RENDERING
    patch_size=64
    ds = MultiPuzzleDataSet(render_flag=render_flag,
                            res=res,
                            patch_size=patch_size,
                            aug_patch=aug_patch,
                            aug_dict=aug_dict,
                            aug_scaling=aug_scaling,
                            gen_surface_normal=gen_surface_normal,
                            weighted_loss=weighted_loss,
                            multichannel=multichannel,
                            use_fp16=params['fp16']
                            )
    def gen_from_geo_type(cfg, geo_type):
        p = pathlib.Path(cfg.rob_fn).parents[0]
        if geo_type == 'rob':
            yield cfg.rob_fn, cfg.env_fn, str(p.joinpath('rob_chart_screened_uniform.png'))
        elif geo_type == 'env':
            yield cfg.env_fn, cfg.env_fn, str(p.joinpath('env_chart_screened_uniform.png'))
        elif geo_type == 'both':
            yield cfg.rob_fn, cfg.env_fn, str(p.joinpath('rob_chart_screened_uniform.png'))
            yield cfg.env_fn, cfg.env_fn, str(p.joinpath('env_chart_screened_uniform.png'))
        else:
            assert False
    for ompl_cfg in ompl_cfgs:
        cfg, _ = parse_ompl.parse_simple(ompl_cfg)
        for rob, env, rob_texfn in gen_from_geo_type(cfg, geo_type):
            if not os.path.isfile(rob_texfn):
                util.warn(f'{ompl_cfgs} does not contain ground truth file {rob_texfn}')
                continue
            ds.add_puzzle(rob=rob, env=env, rob_texfn=rob_texfn)
    return ds

def craft_dict(params):
    dic = {}
    for k in ['suppress_hot', 'red_noise', 'suppress_cold']:
        if k in params:
            dic[k] = float(params[k])
        else:
            dic[k] = 0.0
    return dic

def create_dataset_from_params(params):
    assert 'all_ompl_configs' in params, "all_ompl_configs is mandatory"
    geo_type = params['what_to_render']
    aug_dict = craft_dict(params)
    gen_surface_normal = bool(params['training_data_include_surface_normal']) if 'training_data_include_surface_normal' in params else False
    print(f"create_dataset_from_params with {params}")
    nchannel = None
    if params['multichannel']:
        params['joint_list'] = list(params['all_puzzle_names'])
        nchannel = len(params['joint_list'])
    dataset = create_multidataset(params['all_ompl_configs'],
                                  geo_type=geo_type,
                                  aug_patch=params['enable_augmentation'],
                                  aug_scaling=0.5,
                                  aug_dict=aug_dict,
                                  gen_surface_normal=gen_surface_normal,
                                  weighted_loss=params['weighted_loss'],
                                  multichannel=nchannel,
                                  params=params
                                  )
    if params['multichannel']:
        assert dataset.d_dim == nchannel
        params['num_joints'] = dataset.d_dim
    return dataset

