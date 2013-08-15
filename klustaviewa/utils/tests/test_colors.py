"""Unit tests for colors module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

import numpy as np
from galry import figure, imshow, show, ylim, rectangles

from kwiklib.utils.colors import (COLORS_COUNT, COLORMAP, COLORMAP_TEXTURE,
    next_color, SHIFTLEN)
    

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def hsv_rect(hsv, coords):
    col = hsv_to_rgb(hsv)
    col = np.clip(col, 0, 1)
    rgb_rect(col, coords)

def rgb_rect(rgb, coords):
    x0, y0, x1, y1 = coords
    a = 2./len(rgb)
    c = np.zeros((len(rgb), 4))
    c[:,0] = np.linspace(x0, x1-a, len(rgb))
    c[:,1] = y0
    c[:,2] = np.linspace(x0+a, x1, len(rgb))
    c[:,3] = y1
    rectangles(coordinates=c, color=rgb)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_colors_1():
    for c in xrange(1, COLORS_COUNT):
        assert next_color(c) == c + 1
    assert next_color(COLORS_COUNT) == 1
    
def test_color_galry():
    autodestruct = True
    if autodestruct:
        autodestruct = 100

    figure(constrain_navigation=False, toolbar=False, 
        autodestruct=autodestruct,
        )
    for i in xrange(SHIFTLEN):
        y0 = 1 - 2 * i / float(SHIFTLEN)
        y1 = 1 - 2 * (i + 1) / float(SHIFTLEN)
        rgb_rect(COLORMAP_TEXTURE[i, ...], (-1, y0, 1, y1))
    ylim(-1,1)
    show()
    