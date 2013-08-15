"""Unit tests for waveform view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.views.tests.mock_data import (setup, teardown,
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.selection import select
from kwiklib.dataio.tools import check_dtype, check_shape
from klustaviewa import USERPREF
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
    
    operators = [
        lambda self: self.view.toggle_mask(),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(WaveformView, operators=operators, **kwargs)
    
    