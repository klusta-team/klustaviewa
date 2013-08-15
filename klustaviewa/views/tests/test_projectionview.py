"""Unit tests for projection view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.selection import select
from kwiklib.dataio.tools import check_dtype, check_shape
from klustaviewa import USERPREF
from klustaviewa.views import ProjectionView
from klustaviewa.views.tests.utils import show_view, get_data, assert_fun


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_projectionview():
        
    kwargs = {}
    kwargs['operators'] = [
        lambda self: assert_fun(self.view.get_projection(0) == (0, 0)),
        lambda self: assert_fun(self.view.get_projection(1) == (0, 1)),
        lambda self: self.view.select_channel(0, 5),
        lambda self: self.view.select_feature(0, 1),
        lambda self: self.view.select_channel(1, 32),
        lambda self: self.view.select_feature(1, 2),
        lambda self: assert_fun(self.view.get_projection(0) == (5, 1)),
        lambda self: assert_fun(self.view.get_projection(1) == (32, 2)),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    kwargs['fetdim'] = 3
    kwargs['nchannels'] = 32
    kwargs['nextrafet'] = 3
    
    # Show the view.
    show_view(ProjectionView, **kwargs)
    
    