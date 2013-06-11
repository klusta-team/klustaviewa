"""Unit tests for dataio.tools module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from klustaviewa.dataio.tools import normalize, find_filename


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

def test_find_filename():
    dir = '/my/path/'
    extension_requested = 'spk'
    files = [
        'blabla.aclu.1',
        'blabla_test.aclu.1',
        'blabla_test2.aclu.1',
        'blabla_test3.aclu.3',
        'blabla.spk.1',
        'blabla_test.spk.1',
        'blabla_test.spk.1',
        ]
    spkfile = find_filename('/my/path/blabla.clu.1', extension_requested,
        files=files, dir=dir)
    assert spkfile == dir + 'blabla.spk.1'
        
    spkfile = find_filename('/my/path/blabla_test.clu.1', extension_requested,
        files=files, dir=dir)
    assert spkfile == dir + 'blabla_test.spk.1'
        
    spkfile = find_filename('/my/path/blabla_test2.clu.1', extension_requested,
        files=files, dir=dir)
    assert spkfile == dir + 'blabla_test.spk.1'
        
    spkfile = find_filename('/my/path/blabla_test3.clu.1', extension_requested,
        files=files, dir=dir)
    assert spkfile == dir + 'blabla_test.spk.1'
        
    spkfile = find_filename('/my/path/blabla_test3.clu.3', extension_requested,
        files=files, dir=dir)
    assert spkfile == None
    
    
    
    