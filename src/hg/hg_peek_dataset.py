#!/usr/bin/env python2

import sys
import datagen
from scipy.misc import imsave

def usage():
    print('Usage: hg_peek_dataset.py <datase name> <output dir>')

def main(args):
    ds_name = args[0]
    if ds_name in ['-h', 'help']:
        usage()
        return
    io_dir = args[1]
    aug_dict = { 'suppress_hot' : 0.4, 'red_noise' : 0.4 }
    # aug_dict = { 'red_noise' : 1.0 }
    ds = datagen.create_dataset(ds_name, res=256, aug_patch=True, aug_scaling=0.5, aug_dict=aug_dict) # Use higher resolution for preview
    index = 0
    gen = ds._aux_generator(stacks=1, emit_gt=True)
    for tup in gen:
        train_imgs = tup[0]
        train_gtmaps = tup[1]
        gt_imgs = tup[3]
        for img,hm,gt in zip(train_imgs, train_gtmaps, gt_imgs):
            print('img {} hm {} gt {}'.format(img.shape, hm.shape, gt.shape))
            imsave('{}/{}-img.png'.format(io_dir, index), img[...,:3])
            imsave('{}/{}-hm.png'.format(io_dir, index), hm[0,:,:,0])
            imsave('{}/{}-gt.png'.format(io_dir, index), gt[...,:3])
            index += 1
        if index > 64:
            break

if __name__ == '__main__':
    main(sys.argv[1:])
