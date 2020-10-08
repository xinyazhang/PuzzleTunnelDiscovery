#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fn = sys.argv[1]
d = np.load(fn)
if 'Q' not in d:
    # V,D dataset
    Q = d['V']
    V = d['D']
else:
    # Q,V dataset
    Q = d['Q']
    V = d['V']

xs = Q[:, 0]
ys = Q[:, 1]
zs = Q[:, 2]
ax.scatter(xs, ys, zs, c=V)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
