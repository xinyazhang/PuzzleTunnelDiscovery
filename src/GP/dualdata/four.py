from template import *

STICK_LENGTH_X = 2 * HOLLOW_SQUARE_SIZE + 3 * STICK_WIDTH
# Note, we need additional grids to block easy solution
STICK_LENGTH_Y = 3 * HOLLOW_SQUARE_SIZE + 4 * STICK_WIDTH

STICKS_X_DESC = [
        {
            'origin': (0,0),
            'len' : STICK_LENGTH_X,
            'up': [0,3],
            'down': [0]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0),
            'len' : STICK_LENGTH_X,
            'up': [2,3],
            'down': [1,2,3]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 2.0),
            'len' : STICK_LENGTH_X,
            'up': [],
            'down': []
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 3.0),
            'len' : STICK_LENGTH_X,
            'up': [],
            'down': []
        },
]

STICKS_Y_DESC = [
        {
            'origin': (0,0),
            'len' : STICK_LENGTH_Y,
            'up': [],
            'down': [0,2,3],
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0, 0.0),
            'len' : STICK_LENGTH_Y,
            'up': [0,2,3],
            'down': [2]
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 2.0, 0.0),
            'len' : STICK_LENGTH_Y,
            'up': [],
            'down': []
        },
]

