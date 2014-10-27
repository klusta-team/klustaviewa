"""Unit tests for raw data waveform visualisation view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.views.tests.mock_data import (setup, teardown, 
    create_similarity_matrix, nspikes, nclusters, nsamples, nchannels, fetdim, 
    ncorrbins, create_trace, freq)
from klustaviewa import USERPREF
from klustaviewa.views import TraceView
from klustaviewa.views.tests.utils import show_view, get_data
import tables

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_traceview():
    
    trace = create_trace(int(freq * 60), nchannels)
    ignored_channels = np.arange(5)
    
    kwargs = {}
    kwargs['trace'] = trace
    kwargs['freq'] = freq
    kwargs['spiketimes'] = np.array([1, 30, 100])
    kwargs['ignored_channels'] = ignored_channels
    kwargs['operators'] = [
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(TraceView, **kwargs)
    
