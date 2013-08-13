"""This module provides functions used to read and write KLD files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import tables
import numpy as np

from tools import MemMappedArray


# -----------------------------------------------------------------------------
# Conversion functions
# -----------------------------------------------------------------------------
def create_kld(filename_kld):
    file_kld = tables.openFile(filename_kld, mode='w')
    file_kld.createGroup('/', 'metadata')
    file_kld.setNodeAttr('/', 'VERSION', 1)
    return file_kld

def write_metadata(file_kld, metadata):
    # TODO
    pass
    
def write_raw_data(file_kld, filename_dat, nchannels, 
        nsamples=None):
    # Create the EArray.
    data = file_kld.createEArray('/', 'data', tables.Int16Atom(), 
        (0, nchannels), expectedrows=nsamples)
    
    # Open the DAT file.
    dat = MemMappedArray(filename_dat, np.int16)
    chunk_nrows = 1000
    chunk_pos = 0
    while True:
        # Read the chunk from the DAT file.
        i0, i1 = (chunk_pos * nchannels), (chunk_pos + chunk_nrows) * nchannels
        chunk = dat[i0:i1]
        chunk = chunk.reshape((-1, nchannels))
        if chunk.size == 0:
            break
        # Write the chunk in the EArray.
        data.append(chunk)
        chunk_pos += chunk.shape[0]
    return data

def close_kld(file_kld):
    file_kld.flush()
    file_kld.close()
    
def dat_to_kld(filename_dat, filename_kld, nchannels, nsamples=None,
        metadata=None):
    if os.path.exists(filename_kld):
        return
    file_kld = create_kld(filename_kld)
    # TODO
    write_metadata(file_kld, metadata)
    write_raw_data(file_kld, filename_dat, nchannels, nsamples=None)
    
    close_kld(file_kld)
    
    