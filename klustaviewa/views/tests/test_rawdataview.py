"""Unit tests for raw data waveform visualisation view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.views.tests.mock_data import (setup, teardown, create_similarity_matrix,
        nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from klustaviewa.dataio import KlustersLoader
from klustaviewa.dataio.selection import select
from klustaviewa.dataio.tools import check_dtype, check_shape
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.views import RawDataView
from klustaviewa.views.tests.utils import show_view, get_data
import tables

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_rawdataview():
    
    # for testing - we open a sample h5 file. This will be replaced by dynamically
    # generated raw data.
    
    dir = os.path.dirname(os.path.abspath(__file__))
    try:
        filename = r"datatest/n6mab031109.h5"
        f = tables.openFile(os.path.join(dir, filename))
    except:
        filename = r"datatest/n6mab031109.trim.h5"
        f = tables.openFile(os.path.join(dir, filename))
    try:
        data = f.root.RawData
    except:
        data = f.root.raw_data

    nsamples, nchannels = data.shape
    total_size = nsamples
    freq = 20000.
    dt = 1. / freq
    duration = (data.shape[0] - 1) * dt

    duration_initial = 5.

    x = np.tile(np.linspace(0., duration, nsamples // 500), (nchannels, 1))
    y = np.zeros_like(x)+ np.linspace(-.9, .9, nchannels).reshape((-1, 1))
    
    kwargs = {}
    kwargs['rawdata'] = data
    kwargs['freq'] = freq
    kwargs['operators'] = [
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(RawDataView, **kwargs)
    
if __name__ == "__main__":
    setup()
    test_rawdataview()
    teardown()