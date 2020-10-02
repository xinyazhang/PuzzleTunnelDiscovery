# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import config
import pyosr
import sys
import os
import numpy as np

def render_everything_to(src, dst, label):
    pyosr.init()
    pyosr.create_gl_context(pyosr.create_display())
    modelId = 0
    r=pyosr.Renderer()
    r.setup()
    r.default_depth = 0.0
    view_array = []
    for angle,ncam in config.VIEW_CFG:
        view_array += [ [angle,float(i)] for i in np.arange(0.0, 360.0, 360.0/float(ncam)) ]
    r.views = np.array(view_array)
    w = r.pbufferWidth
    h = r.pbufferHeight
    for root, dirs, files in os.walk(src):
        for fn in files:
            # print(fn)
            _,ext = os.path.splitext(fn)
            if ext not in ['.dae', '.obj', '.ply']:
                continue
            ffn = os.path.join(root, fn)
            r.loadModelFromFile(ffn)
            r.scaleToUnit()
            r.angleModel(0.0, 0.0)
            dep = r.render_mvdepth_to_buffer()
            dep = dep.reshape(r.views.shape[0],w,h,1)
            outfn = os.path.join(dst, '%07d.train' % modelId)
            with open(outfn, 'w') as f:
                if label >= 0:
                    labelbytes = np.array([label],dtype=np.int32).tobytes()
                    f.write(labelbytes)
                f.write(dep.tobytes())
            print('{} -> {}'.format(ffn, outfn))
            modelId += 1

if __name__ == '__main__':
    # print(sys.argv)
    src,dst,label = sys.argv[1:4]
    render_everything_to(src, dst, int(label))
