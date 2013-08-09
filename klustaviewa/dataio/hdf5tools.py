"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables
import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from probe import probe_to_json
from params import params_to_json
from klustersloader import (find_filenames, find_index, read_xml,
    filename_to_triplet, triplet_to_filename, find_indices,
    find_hdf5_filenames, find_filename, find_any_filename,
    read_clusters, read_cluster_info, read_group_info, read_probe,)
from loader import (default_cluster_info, default_group_info)
from klatools import kla_to_json
from tools import MemMappedText, MemMappedBinary


# Table descriptions.
# -------------------
def get_spikes_description(fetcol=None):
    spikes_description = OrderedDict([
        ('time', tables.UInt64Col()),
        ('features', tables.Float32Col(shape=(fetcol,))),
        ('cluster_auto', tables.UInt32Col()),
        ('cluster_manual', tables.UInt32Col()),
        ('masks', tables.UInt8Col(shape=(fetcol,))),])
    return spikes_description
    
def get_waveforms_description(nsamples=None, nchannels=None, has_umask=None):
    waveforms_description = OrderedDict([
        ('waveform_filtered', tables.Int16Col(shape=(nsamples * nchannels))),
        ('waveform_unfiltered', tables.Int16Col(shape=(nsamples * nchannels))),
        ])
    return waveforms_description
    
def get_clusters_description():
    clusters_description = OrderedDict([
        ('cluster', tables.UInt32Col()),
        # ('color', tables.UInt8Col()),
        ('group', tables.UInt8Col()),])
    return clusters_description
    
def get_groups_description():
    groups_description = OrderedDict([
        ('group', tables.UInt8Col()),
        # ('color', tables.UInt8Col()),
        ('name', tables.StringCol(64)),])
    return groups_description
    

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
    
    # Read .aclu data.
    if 'aclu' in filenames and os.path.exists(filenames['aclu']):
        data['aclu'] = read_clusters(filenames['aclu'])
    else:
        data['aclu'] = data['clu']
        
    # Read .acluinfo data.
    if 'acluinfo' in filenames and os.path.exists(filenames['acluinfo']):
        data['acluinfo'] = read_cluster_info(filenames['acluinfo'])
    else:
        data['acluinfo'] = default_cluster_info(np.unique(data['aclu']))
        
    # Read group info.
    if 'groupinfo' in filenames and os.path.exists(filenames['groupinfo']):
        data['groupinfo'] = read_group_info(filenames['groupinfo'])
    else:
        data['groupinfo'] = default_group_info()
    
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

    # data['metadata'] = metadata
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
            
    # Load probe file.
    filename_probe = (find_filename(filename, 'probe') or
                          find_any_filename(filename, 'probe'))
    if filename_probe:
        probe_ns = {}
        execfile(filename_probe, {}, probe_ns)
        klusters_data['probe'] = probe_ns
    
    # Read the metadata.
    filenames = find_filenames(filename)
    metadata = read_xml(filenames['xml'], 1)
    klusters_data['metadata'] = metadata
    
    return klusters_data

def create_hdf5_files(filename, klusters_data):
    hdf5 = {}
    
    hdf5_filenames = find_hdf5_filenames(filename)
    
    # Get the data corresponding to the first shank.
    # klusters_data_first = klusters_data.itervalues().next()
    
    # Create the HDF5 file.
    hdf5['klx'] = tables.openFile(hdf5_filenames['hdf5_klx'], mode='w')
    hdf5['kla'] = open(hdf5_filenames['hdf5_kla'], mode='w')
    
    # Metadata.
    # for file in [hdf5['klx'], hdf5['wave_file']]:
    file = hdf5['klx']
    file.createGroup('/', 'shanks')
    file.createGroup('/', 'metadata')
    
    # Get the old probe information, convert it to JSON, and save it in
    # /metadata
    if 'probe' in klusters_data:
        probe_text = probe_to_json(klusters_data['probe'])
    else:
        probe_text = ''
    file.setNodeAttr('/metadata', 'PRB_JSON', probe_text)
    
    # Read the old XML metadata and save the JSON parameters string.
    params_text = params_to_json(klusters_data['metadata'])
    file.setNodeAttr('/metadata', 'PRM_JSON', params_text)
    
    # Create groups and tables for each shank.
    shanks = sorted([key for key in klusters_data.keys() 
        if isinstance(key, (int, long))])
    for shank in shanks:
        data = klusters_data[shank]
        
        shank_path = '/shanks/shank{0:d}'.format(shank)
        
        # Create the /shanks/shank<X> groups in each file.
        # for file in [hdf5['klx'], hdf5['wave_file']]:
        file = hdf5['klx']
        file.createGroup('/shanks', 'shank{0:d}'.format(shank))
        # file.createGroup(shank_path, 'metadata')
        # file.setNodeAttr(shank_path + '/metadata',
            # 'nchannels', data['nchannels'],)
        # file.setNodeAttr(shank_path + '/metadata',
            # 'nsamples', data['nsamples'],)
        # file.setNodeAttr(shank_path + '/metadata',
            # 'fetdim', data['fetdim'],)
        
                
        # Create the cluster table.
        # -------------------------
        hdf5['cluster_table', shank] = hdf5['klx'].createTable(
            shank_path, 'clusters', 
            get_clusters_description())
            
        # Fill the table.
        if 'acluinfo' in data:
            for cluster, clusterinfo in data['acluinfo'].iterrows():
                row = hdf5['cluster_table', shank].row
                row['cluster'] = cluster
                # row['color'] = clusterinfo['color']
                row['group'] = clusterinfo['group']
                row.append()
            
            
        # Create the group table.
        # -----------------------
        hdf5['group_table', shank] = hdf5['klx'].createTable(
            shank_path, 'groups_of_clusters', 
            get_groups_description())
            
        # Fill the table.
        if 'groupinfo' in data:
            for group, groupinfo in data['groupinfo'].iterrows():
                row = hdf5['group_table', shank].row
                row['group'] = group
                # row['color'] = groupinfo['color']
                row['name'] = groupinfo['name']
                row.append()
               
               
        # Create the spike table.
        # -----------------------
        hdf5['spike_table', shank] = hdf5['klx'].createTable(
            shank_path, 'spikes', 
            get_spikes_description(
                fetcol=data['fetcol'],
                ))
                
                
        # Create the wave table.
        # ----------------------
        hdf5['wave_table', shank] = hdf5['klx'].createTable(
            shank_path, 'waveforms', 
            get_waveforms_description(
                nsamples=data['nsamples'],
                nchannels=data['nchannels'],
                # has_umask=('uspk' in data)
                ))
        
        # Create the link in the main file, to the wave table.
        # hdf5['klx'].createExternalLink(
            # shank_path, 'waveforms', 
            # hdf5['wave_table', shank])

    return hdf5

def klusters_to_hdf5(filename, progress_report=None):
    with HDF5Writer(filename) as f:
        # Callback function for progress report.
        if progress_report is not None:
            f.progress_report(progress_report)
        f.convert()
    
   
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
        self.shanks = sorted([key for key in self.klusters_data.keys() 
            if isinstance(key, (int, long))])
        self.shank = self.shanks[0]
        self.spike = 0
        
    def read_next_spike(self):
        if self.spike >= self.klusters_data[self.shank]['nspikes']:
            return {}
        data = self.klusters_data[self.shank]
        read = {}
        read['cluster'] = data['aclu'][self.spike]
        read['fet'] = data['fet'].next()
        read['time'] = read['fet'][-1]
        read['spk'] = data['spk'].next()
        if 'mask' in data:
            read['mask'] = data['mask'].next()
        else:
            read['mask'] = np.ones_like(read['fet'])
        self.spike += 1
        return read
        
    def write_spike(self, read):
        
        # Create the rows.
        row_main = self.hdf5_data['spike_table', self.shank].row
        row_wave = self.hdf5_data['wave_table', self.shank].row

        # Fill the main row.
        # WARNING: when converting from Klusters, we have no way to know 
        # if the clustering is automatic or manual, so we set it to manual
        # and put a fixed value in auto.
        row_main['cluster_manual'] = read['cluster']
        row_main['cluster_auto'] = 2
        row_main['features'] = read['fet']
        row_main['time'] = read['time']
        # if 'mask' in read:
        row_main['masks'] = (read['mask'] * 255).astype(np.uint8)
        row_main.append()
        
        # Fill the wave row.
        row_wave['waveform_filtered'] = read['spk']
        if 'uspk' in read:
            row_wave['waveform_unfiltered'] = read['uspk']
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
            # Create the KLA file.
            kla_json = kla_to_json(dict(
                cluster_colors=self.klusters_data[self.shank]['acluinfo']['color'],
                group_colors=self.klusters_data[self.shank]['groupinfo']['color'],
            ))
            self.hdf5_data['kla'].write(kla_json)
            
            # Convert in HDF5 by going through all spikes.
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
        if hasattr(self, 'shanks'):
            for shank in self.shanks:
                for data in self.klusters_data[shank]:
                    if isinstance(data, (MemMappedBinary, MemMappedText)):
                        data.close()
        
        # Close the KLA file.
        if self.hdf5_data['kla']:
            self.hdf5_data['kla'].close()
        
        # Close the HDF5 files.
        if self.hdf5_data['klx'].isopen:
            self.hdf5_data['klx'].flush()
            self.hdf5_data['klx'].close()
        
    def __del__(self):
        self.close()
        
    def __exit__(self, exception_type, exception_val, trace):
        self.close()




