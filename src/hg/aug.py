# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
import numpy as np

def _clip_imgcoord_inplace(img_coord, img_shape):
    img_coord[0] = np.clip(img_coord[0], 0, img_shape[0])
    img_coord[1] = np.clip(img_coord[1], 0, img_shape[1])
    return img_coord

def _calc_maxs(img_shape, tl, size):
    maxs = tl + size
    return _clip_imgcoord_inplace(maxs, img_shape)

'''
patch_finder_1:
    Randomly sample a square patch that
    1. masks the cool region defined by `coldmap`
    2. does not mask the hot region defined by `heatmap`
    3. its size should be `patch_size`

    Returns:
        An np.array(shape=(2), dtype=np.int) object, indicating the top left corner of the patch.
        or
        None, when failed to find a patch within `max_trial` iterations
'''
def patch_finder_1(coldmap, heatmap, patch_size, max_trial=32):
    cold_x, = np.nonzero(np.sum(coldmap, axis=1))
    cold_y, = np.nonzero(np.sum(coldmap, axis=0))
    if len(cold_x) == 0 or len(cold_y) == 0:
        return None
    tl = None
    for i in range(max_trial):
        tl_x = np.random.choice(cold_x)
        tl_y = np.random.choice(cold_y)
        tl = np.array([tl_x, tl_y], dtype=np.int32)
        maxs = _calc_maxs(coldmap.shape, tl, patch_size)
        '''
        Check if covers something in coldmap
        '''
        if np.sum(coldmap[tl[0]:maxs[0], tl[1]:maxs[1]]) == 0:
            continue
        '''
        Check if covers anything in heatmap
        We leave 2 pix margin

        FIXME: if 0 in tl, we'll have the margin of two pixels in the bottom/right side
        '''
        tl = np.clip(tl - 2, 0, None)
        maxs = _calc_maxs(coldmap.shape, tl, patch_size+4)
        if np.sum(heatmap[tl[0]:maxs[0], tl[1]:maxs[1]]) == 0:
            break
    return tl

def patch_finder_hot(heatmap, margin_pix):
    hot_x, = np.nonzero(np.sum(heatmap, axis=1))
    hot_y, = np.nonzero(np.sum(heatmap, axis=0))
    if len(hot_x) == 0 or len(hot_y) == 0:
        return np.zeros(shape=2, dtype=np.int32), np.zeros(shape=2, dtype=np.int32)
    tl = np.array([np.min(hot_x), np.min(hot_y)], dtype=np.int32)
    br = np.array([np.max(hot_x), np.max(hot_y)], dtype=np.int32)
    tl -= margin_pix
    br += margin_pix
    _clip_imgcoord_inplace(tl, heatmap.shape)
    _clip_imgcoord_inplace(br, heatmap.shape)
    return tl, br - tl

def patch_rgb(img, tl, size, default=0):
    maxs = _calc_maxs(img.shape, tl, size)
    img[tl[0]:maxs[0], tl[1]:maxs[1], :] = default
    return img

def dim_rgb(img, tl, size, factor=0.5):
    maxs = _calc_maxs(img.shape, tl, size)
    img[tl[0]:maxs[0], tl[1]:maxs[1], :] *= factor
    return img

def red_noise(img, tl, size):
    #maxs = _calc_maxs(img.shape, tl, size)
    #bak = np.copy(img[tl[0]:maxs[0], tl[1]:maxs[1], :])
    # Poisson distribution
    # lam is the expectation
    img[:,:,0] += 255.0 * np.random.poisson(0.05, size=(img.shape[0:2]))
    #img[tl[0]:maxs[0], tl[1]:maxs[1], :] = bak
    return img

def focus(img, tl, size):
    maxs = _calc_maxs(img.shape, tl, size)
    old = img
    img = np.zeros(shape=img.shape, dtype=img.dtype)
    img[tl[0]:maxs[0], tl[1]:maxs[1], :] = old[tl[0]:maxs[0], tl[1]:maxs[1], :]
    return img
