"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tables
import time

import numpy as np
import matplotlib.pyplot as plt

from klustersloader import (find_filenames, find_index, read_xml,
    read_clusters, read_cluster_info, read_group_info, read_probe,)
from tools import MemMappedText, MemMappedBinary


# -----------------------------------------------------------------------------
# HDF5 helper functions
# -----------------------------------------------------------------------------
def open_klusters(filename):
    filenames = find_filenames(filename)
    fileindex = find_index(filename)
    
    # Open small Klusters files.
    data = {}
    metadata = read_xml(filenames['xml'], fileindex)
    data['clu'] = read_clusters(filenames['clu'])
    if 'aclu' in filenames and os.path.exists(filenames['aclu']):
        data['aclu'] = read_clusters(filenames['aclu'])
    if 'acluinfo' in filenames and os.path.exists(filenames['acluinfo']):
        data['acluinfo'] = read_cluster_info(filenames['acluinfo'])
    if 'groupinfo' in filenames and os.path.exists(filenames['groupinfo']):
        data['groupinfo'] = read_group_info(filenames['groupinfo'])
    if 'probe' in filenames:
        data['probe'] = read_probe(filenames['probe'])

    # Find out the number of columns in the .fet file.
    with open(filenames['fet'], 'r') as f:
        f.readline()
        # Get the number of non-empty columns in the .fet file.
        data['fetcol'] = len([col for col in f.readline().split(' ') if col.strip() != ''])
    
    metadata['nspikes'] = len(data['clu'])
    # data['nchannels'] = metadata['nchannels']
    # data['nsamples'] = metadata['nsamples']

    # Open big Klusters files.
    data['fet'] = MemMappedText(filenames['fet'], np.int64, skiprows=1)
    data['spk'] = MemMappedBinary(filenames['spk'], np.int16, 
        rowsize=metadata['nchannels'] * metadata['nsamples'])
    if 'uspk' in filenames and os.path.exists(filenames['uspk'] or ''):
        data['uspk'] = MemMappedBinary(filenames['uspk'], np.int16, 
            rowsize=metadata['nchannels'] * metadata['nsamples'])
    if 'mask' in filenames and os.path.exists(filenames['mask'] or ''):
        data['mask'] = MemMappedText(filenames['mask'], np.float32, skiprows=1)

    data.update(metadata)
        
    return filenames, data
    
def create_hdf5_files(filenames, data):
    fetcol = data['fetcol']
    nsamples = data['nsamples']
    nchannels = data['nchannels']
    
    hdf5 = {}
    
    # Create the HDF5 file.
    hdf5['main_file'] = tables.openFile(filenames['hdf5_main'], mode='w')
    hdf5['main_file'].createGroup('/', 'shanks')
    hdf5['main_file'].createGroup('/shanks', 'shank0')

    hdf5['wave_file'] = tables.openFile(filenames['hdf5_wave'], mode='w')
    hdf5['wave_file'].createGroup('/', 'shanks')
    hdf5['wave_file'].createGroup('/shanks', 'shank0')
    
    spikes_description = dict(
        time=tables.UInt64Col(),
        # mask_binary=tables.BoolCol(shape=(fetcol,)),
        # mask_float=tables.Int8Col(shape=(fetcol,)),
        features=tables.Float32Col(shape=(fetcol,)),
        cluster=tables.UInt32Col(),)
    if 'mask' in data:
        spikes_description['mask'] = tables.UInt8Col(shape=(fetcol,))
    waves_description = dict(
            wave=tables.Float32Col(shape=(nsamples * nchannels)),)
    if 'uspk' in data:
        waves_description['wave_unfiltered'] = tables.Float32Col(
            shape=(nsamples * nchannels))

    hdf5['spike_table'] = hdf5['main_file'].createTable(
        '/shanks/shank0', 'spikes', spikes_description)
    hdf5['wave_table'] = hdf5['wave_file'].createTable(
        '/shanks/shank0', 'waves', waves_description)
    hdf5['main_file'].createExternalLink(
        '/shanks/shank0', 'waveforms', hdf5['wave_table'])

    # TODO: write metadata
        
    return hdf5
    
   
# -----------------------------------------------------------------------------
# HDF5 writer
# ----------------------------------------------------------------------------- 
class HDF5Writer(object):
    def __init__(self, filename=None):
        self._progress_callback = None
        self.filename = filename
        
    def __enter__(self):
        if self.filename:
            self.open(self.filename)
        return self
            
    def open(self, filename=None):
        if filename is not None:
            self.filename = filename
        self.filenames, self.klusters_data = open_klusters(self.filename)
        self.hdf5_data = create_hdf5_files(self.filenames, self.klusters_data)
        self.nspikes = self.klusters_data['nspikes']
        self.nchannels = self.klusters_data['nchannels']
        self.nsamples = self.klusters_data['nsamples']
        self.spike = 0
        
    def read_next_spike(self):
        if self.spike >= self.nspikes:
            return {}
        data = self.klusters_data
        read = {}
        read['cluster'] = data['clu'][self.spike]
        read['fet'] = data['fet'].next()
        read['time'] = read['fet'][-1]
        read['spk'] = data['spk'].next()
        if 'mask' in data:
            read['mask'] = data['mask'].next()
        self.spike += 1
        return read
        
    def write_spike(self, read):
        
        # Create the rows.
        row_main = self.hdf5_data['spike_table'].row
        row_wave = self.hdf5_data['wave_table'].row

        # Fill the main row.
        row_main['cluster'] = read['cluster']
        row_main['features'] = read['fet']# * 1e-5
        row_main['time'] = read['time']
        if 'mask' in read:
            row_main['mask'] = (read['mask'] * 255).astype(np.uint8)
        row_main.append()
        
        # Fill the wave row.
        row_wave['wave'] = read['spk']# * 1e-5
        row_wave.append()

    def convert(self):
        """Convert the old file format to the new HDF5-based format."""
        read = self.read_next_spike()
        self._progress_callback(self.spike, self.nspikes)
        while read:
            self.write_spike(read)
            read = self.read_next_spike()
            self._progress_callback(self.spike, self.nspikes)
        
    def progress_report(self, fun):
        self._progress_callback = fun
        return fun
        
    def close(self):
        """Close all files."""
        
        # Close the memory-mapped Klusters files.
        for data in self.klusters_data:
            if isinstance(data, (MemMappedBinary, MemMappedText)):
                data.close()
        
        # Close the HDF5 files.
        if self.hdf5_data['main_file'].isopen:
            self.hdf5_data['main_file'].flush()
            self.hdf5_data['main_file'].close()

        if self.hdf5_data['wave_file'].isopen:
            self.hdf5_data['wave_file'].flush()
            self.hdf5_data['wave_file'].close()

    def __del__(self):
        self.close()
        
    def __exit__(self, exception_type, exception_val, trace):
        self.close()
        



