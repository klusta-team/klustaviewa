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
    ncorrbins, create_rawdata, freq)
from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.selection import select
from kwiklib.dataio.tools import check_dtype, check_shape
from klustaviewa import USERPREF
from klustaviewa.views import RawDataView
from klustaviewa.views.tests.utils import show_view, get_data
import tables

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_rawdataview():
    
    rawdata = create_rawdata(int(freq * 60), nchannels)
    dead_channels = np.arange(5)
    
    kwargs = {}
    kwargs['rawdata'] = rawdata
    kwargs['freq'] = freq
    kwargs['dead_channels'] = dead_channels
    kwargs['operators'] = [
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(RawDataView, **kwargs)
    
