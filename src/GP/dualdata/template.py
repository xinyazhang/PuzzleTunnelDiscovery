# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
import numpy as np

'''
Store invariants across puzzles
'''

THICKNESS = 4.57
FILLISTER_DEPTH = THICKNESS - 3.76
FILLISTER_LENGTH = 3.58
FILLISTER_MARGIN = 0.02

HOLLOW_SQUARE_SIZE = 14.00
TRI_STICK_LENGTH = 61.60
STICK_WIDTH = (TRI_STICK_LENGTH - 3 * HOLLOW_SQUARE_SIZE) / 4.0
STICK_HEIGHT = THICKNESS

# STICK_LENGTH is a variable

UC_V = np.array(
[[1,0,0],
[1,1,1],
[0,1,1],
[1,0,1],
[1,1,0],
[0,0,1],
[0,0,0],
[0,1,0]], dtype=np.float64)

UC_F = np.array(
[
[4,0,6],
[4,6,7],
[1,2,5],
[1,5,3],
[4,1,3],
[4,3,0],
[0,3,5],
[0,5,6],
[6,5,2],
[6,2,7],
[1,4,7],
[1,7,2]], dtype=np.int32)
