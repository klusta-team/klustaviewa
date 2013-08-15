"""Unit tests for kldtools module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
import tables as tb

from kwiklib.dataio import (save_binary, create_kld, write_raw_data, 
    close_kld, dat_to_kld, read_dat)
from kwiklib.dataio.tests import create_rawdata, duration, freq, nchannels

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------
nsamples = int(duration * freq)

def setup():
    # Create mock directory if needed.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    dat = create_rawdata(nsamples, nchannels)
    
    # Create mock DAT file.
    save_binary(os.path.join(dir, 'test.dat'), dat)
    

# -----------------------------------------------------------------------------
# KLD tests
# -----------------------------------------------------------------------------
def test_kld_1():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    filename_dat = os.path.join(dir, 'test.dat')
    filename_kld = os.path.join(dir, 'test.kld')
    
    # Convert the DAT file in KLD.
    dat_to_kld(filename_dat, filename_kld, nchannels, 
        nsamples=nsamples)
    
    # Load DAT file (memmap).
    dat = read_dat(filename_dat, nchannels)
    assert dat.shape == (nsamples, nchannels)
    
    # Load KLD file.
    file_kld = tb.openFile(filename_kld)
    kld = file_kld.root.data[:]
    assert kld.shape == (nsamples, nchannels)

    # Check they are identical.
    assert np.array_equal(dat, kld)
    
    # Close the KLD file.
    close_kld(file_kld)
    
    
