"""Unit tests for dataio.tools module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

import numpy as np

from kwiklib.dataio import (normalize, find_filename, save_text, 
    MemMappedText, load_text, save_binary, read_dat)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_normalize():
    data = np.array([.5, .75, 1.])
    
    # normalization
    data_normalized = normalize(data)
    assert np.array_equal(data_normalized, [-1, 0, 1])
    
    # normalization with a different range
    data_normalized = normalize(data, range=(0, 1))
    assert np.array_equal(data_normalized, [0, 0.5, 1])
    
    # symmetric normalization (0 stays 0)
    data_normalized = normalize(data, symmetric=True)
    assert np.array_equal(data_normalized, data)

def test_memmap_text():
    folder = tempfile.gettempdir()
    filename = os.path.join(folder, 'memmap')
    
    x = np.random.randint(size=(MemMappedText.BUFFER_SIZE + 1000, 10), 
        low=0, high=100)
    save_text(filename, x)
    
    m = MemMappedText(filename, np.int32)
    
    l = m.next()
    i = 0
    while l is not None:
        assert np.array_equal(l, x[i, :])
        i += 1
        l = m.next()
        
def test_memmap_numpy():
    folder = tempfile.gettempdir()
    filename = os.path.join(folder, 'memmapb')
    
    dtype = np.int16
    freq = 20000.
    duration = 10.
    nchannels = 32
    nsamples = int(freq * duration)
    
    x = np.random.randint(size=(nsamples, nchannels), 
        low=0, high=1000).astype(dtype)
    save_binary(filename, x)
    
    m = read_dat(filename, nchannels=nchannels, dtype=dtype)
    
    slices = (slice(1000, 10000, 4), slice(2, 30, 3))
    
    assert m.shape == x.shape
    np.testing.assert_equal(x[slices], m[slices])
    
    
    