"""Unit tests for loader module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
from collections import Counter

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil
from nose.tools import with_setup

from klustaviewa.dataio.tests.mock_data import (
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from klustaviewa.dataio import (HDF5Loader, HDF5Writer, select, get_indices,
    check_dtype, check_shape, get_array, load_text)
from klustaviewa.utils.userpref import USERPREF


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_hdf5_loader1():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    filename = os.path.join(dir, 'test.xml')
    
    # Convert in HDF5.
    with HDF5Writer(filename) as writer:
        writer.convert()
        
    # Open the HDF5 file.
    filename = os.path.join(dir, 'test.main.h5')
    l = HDF5Loader(filename=filename)
    
    print l.cluster_info
    print l.group_info
    
    
    l.close()
    
    