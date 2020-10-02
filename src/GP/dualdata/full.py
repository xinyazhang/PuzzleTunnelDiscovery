# Copyright (C) 2020 The University of Texas at Austin
# SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
from .template import *

STICK_LENGTH = 61.60

# Caliper gives different numbers for different bars.
# Hence we decided to assume the width is uniform

# Describe the sticks
#   origin: the bottom left cornor
#   up: fillister locations on upside surface (0: beginning at the first hollow, 1: ending at the first hollow, ...)
#   down: fillister locations on downside surface
STICKS_X_DESC = [
        {
            'origin': (0,0),
            'len' : STICK_LENGTH,
            'up': [0,3],
            'down': [0]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0),
            'len' : STICK_LENGTH,
            'up': [2,3,4],
            'down': [1,2,3]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 2.0),
            'len' : STICK_LENGTH,
            'up': [0,2,3],
            'down': [2,4]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 3.0),
            'len' : STICK_LENGTH,
            'up': [],
            'down': [0,2,3,5]
        }
]

STICKS_Y_DESC = [
        {
            'origin': (0,0),
            'len' : STICK_LENGTH,
            'up': [],
            'down': [0,2,3],
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0, 0.0),
            'len' : STICK_LENGTH,
            'up': [0,2,3],
            'down': [2,5]
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 2.0, 0.0),
            'len' : STICK_LENGTH,
            'up': [2,3,5],
            'down': [0,3]
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 3.0, 0.0),
            'len' : STICK_LENGTH,
            'up': [],
            'down': [2,3,5]
        },
]
