"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables
import time

import numpy as np
import matplotlib.pyplot as plt

from klustersloader import (find_filenames, find_index, read_xml,
    filename_to_triplet, triplet_to_filename, find_indices,
    find_hdf5_filenames,
    read_clusters, read_cluster_info, read_group_info, read_probe,)
from tools import MemMappedText, MemMappedBinary


# -----------------------------------------------------------------------------
# HDF5 helper functions
# -----------------------------------------------------------------------------
def open_klusters_oneshank(filename):
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
    data['fileindex'] = fileindex

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
    
    return data
    
def open_klusters(filename):
    indices = find_indices(filename)
    triplet = filename_to_triplet(filename)
    filenames = {}
    for index in indices:
        filenames[index] = triplet_to_filename(triplet[:2] + (index,))
    klusters_data = {index: open_klusters_oneshank(filename) 
        for index, filename in filenames.iteritems()}
    return klusters_data
    
def get_spikes_description(fetcol=None, has_mask=None):
    spikes_description = dict(
        time=tables.UInt64Col(),
        features=tables.Float32Col(shape=(fetcol,)),
        cluster=tables.UInt32Col(),)
    if has_mask:
        spikes_description['mask'] = tables.UInt8Col(shape=(fetcol,))
    return spikes_description
    
def get_waves_description(nsamples=None, nchannels=None, has_umask=None):
    waves_description = dict(
        wave=tables.Float32Col(shape=(nsamples * nchannels)),)
    if has_umask:
        waves_description['wave_unfiltered'] = tables.Float32Col(
            shape=(nsamples * nchannels))
    return waves_description
    
def create_hdf5_files(filename, klusters_data):
    hdf5 = {}
    
    hdf5_filenames = find_hdf5_filenames(filename)
    
    # Create the HDF5 file.
    hdf5['main_file'] = tables.openFile(hdf5_filenames['hdf5_main'], mode='w')
    hdf5['wave_file'] = tables.openFile(hdf5_filenames['hdf5_wave'], mode='w')
    
    # Metadata.
    for file in [hdf5['main_file'], hdf5['wave_file']]:
        file.createGroup('/', 'shanks')
        metadata_group = file.createGroup('/', 'metadata')
        file.setNodeAttr(metadata_group, 'probe', '')#json.dumps(probe.probes))
    
    # Create groups and tables for each shank.
    for shank, data in klusters_data.iteritems():
        
        hdf5['main_file'].createGroup('/shanks', 'shank{0:d}'.format(shank)) 
        hdf5['wave_file'].createGroup('/shanks', 'shank{0:d}'.format(shank)) 

        hdf5['spike_table', shank] = hdf5['main_file'].createTable(
            '/shanks/shank{0:d}'.format(shank), 'spikes', 
            get_spikes_description(
                fetcol=data['fetcol'],
                has_mask=('mask' in data)))
                
        hdf5['wave_table', shank] = hdf5['wave_file'].createTable(
            '/shanks/shank{0:d}'.format(shank), 'waves', 
            get_waves_description(
                nsamples=data['nsamples'],
                nchannels=data['nchannels'],
                has_umask=('uspk' in data)))
                
        hdf5['main_file'].createExternalLink(
            '/shanks/shank{0:d}'.format(shank), 
            'waveforms', 
            hdf5['wave_table', shank])

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
        self.klusters_data = open_klusters(self.filename)
        # self.filenames, self.klusters_data
        self.hdf5_data = create_hdf5_files(self.filename, self.klusters_data)
        self.shanks = sorted(self.klusters_data.keys())
        self.shank = self.shanks[0]
        self.spike = 0
        
    def read_next_spike(self):
        if self.spike >= self.klusters_data[self.shank]['nspikes']:
            return {}
        data = self.klusters_data[self.shank]
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
        row_main = self.hdf5_data['spike_table', self.shank].row
        row_wave = self.hdf5_data['wave_table', self.shank].row

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

    def report_progress(self):
        if self._progress_callback:
            self._progress_callback(
                self.spike, 
                self.klusters_data[self.shank]['nspikes'], 
                self.shanks.index(self.shank),
                len(self.shanks))
        
    def convert(self):
        """Convert the old file format to the new HDF5-based format."""
        for self.shank in self.shanks:
            self.spike = 0
            read = self.read_next_spike()
            self.report_progress()
            while read:
                self.write_spike(read)
                read = self.read_next_spike()
                self.report_progress()
        
    def progress_report(self, fun):
        self._progress_callback = fun
        return fun
        
    def close(self):
        """Close all files."""
        
        # Close the memory-mapped Klusters files.
        for shank in self.shanks:
            for data in self.klusters_data[shank]:
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




