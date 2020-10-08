#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

import sys
import datagen
from scipy.misc import imsave
import numpy as np

def usage():
    print('Usage: hg_peek_dataset.py <datase name> <output dir>')

def main(args):
    ds_name = args[0]
    if ds_name in ['-h', 'help']:
        usage()
        return
    io_dir = args[1]
    aug_dict = { 'suppress_hot' : 0.3, 'red_noise' : 0.3, 'suppress_cold': 0.3 }
    # aug_dict = {}
    # aug_dict = { 'red_noise' : 1.0 }
    ds = datagen.create_dataset(ds_name, res=256, aug_patch=True, aug_scaling=0.0, aug_dict=aug_dict) # Use higher resolution for preview
    index = 0
    gen = ds._aux_generator(stacks=1, emit_gt=True)
    accum_img = None
    for tup in gen:
        train_imgs = tup[0]
        train_gtmaps = tup[1]
        gt_imgs = tup[3]
        for img,hm,gt in zip(train_imgs, train_gtmaps, gt_imgs):
            print('img {} hm {} gt {}'.format(img.shape, hm.shape, gt.shape))
            if accum_img is None:
                accum_img = img
            else:
                accum_img += img
            imsave('{}/{}-img.png'.format(io_dir, index), img[...,:3])
            imsave('{}/{}-hm.png'.format(io_dir, index), hm[0,:,:,0])
            imsave('{}/{}-gt.png'.format(io_dir, index), gt[...,:3])
            index += 1
        if index > 16:
            break
    accum_img[...,1] = np.mean(accum_img)
    imsave('{}/sum-img.png'.format(io_dir, index), accum_img[...,:3])

if __name__ == '__main__':
    main(sys.argv[1:])
