#!/usr/bin/env python3
# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later

import sys, os, pathlib
# print(os.path.realpath(__file__))
# print(sys.path)
sys.path.append(str(pathlib.Path.home().joinpath('bin')))
sys.path.append(os.getcwd())
# print(sys.path)
import glfw
import argparse
import numpy as np
from imageio import imread
from pyvistexture import TextureViewer

class AtexViewer(TextureViewer):
    def __init__(self):
        super().__init__()
        self.mc_tex = None
        self.ch_sel = 0
        self.total_channel = 1

    def key_up(self, key, mod):
        print(f"key {key} mod {mod}")
        if glfw.KEY_PAGE_UP == key:
            self.ch_sel = self.ch_sel + 1
            self.ch_sel = self.ch_sel % self.total_channel
            self.update_tex()
            return False
        if glfw.KEY_PAGE_DOWN == key:
            self.ch_sel = self.ch_sel - 1
            self.ch_sel = self.ch_sel % self.total_channel
            self.update_tex()
            return False
        return True

    def add_sc_tex(self, tex):
        assert len(tex.shape) == 2
        mc_tex = np.expand_dims(tex, 2)
        self.add_mc_tex(mc_tex)

    def add_mc_tex(self, tex):
        self.mc_tex = tex
        self.total_channel = self.mc_tex.shape[-1]
        self.update_tex()

    def update_tex(self):
        print(f"update_tex. current channel {self.ch_sel}")
        tex = self.mc_tex[:,:,self.ch_sel]
        # print(f'mc_tex {self.mc_tex.shape}')
        # print(f'tex {tex.shape}')
        w = tex.shape[0]
        h = tex.shape[1]
        r_ch = np.full((w, h), 32, dtype=np.uint8)
        g_ch = np.empty((w, h), dtype=np.uint8)
        # print(tex.shape)
        ma = np.max(tex)
        mi = np.min(tex)
        ntex = np.array(tex - mi, dtype=np.float32)
        if ma - mi > 0:
            print(f"ntex {ntex.shape} {ntex.dtype} ma {ma} mi {mi}")
            ntex /= ma - mi
        g_ch[...] = ntex[...] * 255.0
        b_ch = np.full((w, h), 32, dtype=np.uint8)
        self.update_texture(r_ch, g_ch, b_ch)

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('obj', help='.OBJ file')
    p.add_argument('tex', help='.png/.npz texture file')
    args = p.parse_args()
    v = AtexViewer()
    v.load_geometry(args.obj)
    v.init_viewer()
    if args.tex.endswith('.png'):
        tex = imread(args.tex)
    else:
        tex = np.load(args.tex)['ATEX']
    tex = np.flipud(tex)
    if len(tex.shape) == 2:
        v.add_sc_tex(tex)
    elif len(tex.shape) == 3:
        v.add_mc_tex(tex)
    else:
        assert False
    v.run()

if __name__ == '__main__':
    main()
