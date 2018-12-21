import numpy as np

def _calc_maxs(img_shape, tl, size):
    maxs = tl + size
    maxs[0] = np.clip(maxs[0], 0, img_shape[0])
    maxs[1] = np.clip(maxs[1], 0, img_shape[1])
    return maxs

'''
patch_finder_1:
    Randomly sample a square patch that
    1. masks the cool region defined by `coldmap`
    2. does not mask the hot region defined by `heatmap`
    3. its size should be `patch_size`

    Returns:
        An np.array(shape=(2), dtype=np.int) object, indicating the top left corner of the patch.
'''
def patch_finder_1(coldmap, heatmap, patch_size):
    cold_x, = np.nonzero(np.sum(coldmap, axis=1))
    cold_y, = np.nonzero(np.sum(coldmap, axis=0))
    while True:
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
        We leave 1 pix margin

        FIXME: if 0 in tl, we'll have the margin of two pixels in the bottom/right side
        '''
        tl = np.clip(tl - 1, 0, None)
        maxs = _calc_maxs(coldmap.shape, tl, patch_size+2)
        if np.sum(heatmap[tl[0]:maxs[0], tl[1]:maxs[1]]) == 0:
            break
    return tl

def patch_rgb(img, tl, size, default=0):
    maxs = _calc_maxs(img.shape, tl, size)
    img[tl[0]:maxs[0], tl[1]:maxs[1], :] = default

def dim_rgb(img, tl, size, factor=0.5):
    maxs = _calc_maxs(img.shape, tl, size)
    img[tl[0]:maxs[0], tl[1]:maxs[1], :] *= factor
