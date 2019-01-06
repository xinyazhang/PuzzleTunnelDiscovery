from template import *

STICK_LENGTH_X = 2 * HOLLOW_SQUARE_SIZE + 3 * STICK_WIDTH
STICK_LENGTH_Y = HOLLOW_SQUARE_SIZE + 2 * STICK_WIDTH

STICKS_X_DESC = [
        {
            'origin': (0,0),
            'len' : STICK_LENGTH_X,
            'up': [3],
            'down': [0]
        },
        {
            'origin': (0, (HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0),
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
            'down': [],
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 1.0, 0.0),
            'len' : STICK_LENGTH_Y,
            'up': [],
            'down': []
        },
        {
            'origin': ((HOLLOW_SQUARE_SIZE + STICK_WIDTH) * 2.0, 0.0),
            'len' : STICK_LENGTH_Y,
            'up': [],
            'down': []
        },
]

