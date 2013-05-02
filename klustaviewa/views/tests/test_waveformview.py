"""Unit tests for waveform view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.io.tests.mock_data import (setup, teardown,
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from klustaviewa.io.loader import KlustersLoader
from klustaviewa.io.selection import select
from klustaviewa.io.tools import check_dtype, check_shape
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.views import WaveformView
from klustaviewa.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_waveformview():
    
    keys = ('waveforms,clusters,cluster_colors,clusters_selected,masks,'
            'geometrical_positions'
            ).split(',')
           
    data = get_data()
    kwargs = {k: data[k] for k in keys}
    
    
    # kwargs['clusters'] = 
    
    operators = [
        lambda self: self.view.toggle_mask(),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(WaveformView, operators=operators, **kwargs)
    
    