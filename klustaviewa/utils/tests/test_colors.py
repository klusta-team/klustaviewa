"""Unit tests for colors module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

from klustaviewa.utils.colors import COLORMAP, COLORS_COUNT, next_color


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_colors_1():
    for c in xrange(1, COLORS_COUNT):
        assert next_color(c) == c + 1
    
    assert next_color(COLORS_COUNT) == 1
    
    