"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables
import time
import shutil
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from probe import probe_to_json, all_to_all_probe
from params import params_to_json
from klustersloader import (find_filenames, find_index, read_xml,
    filename_to_triplet, triplet_to_filename, find_indices,
    find_hdf5_filenames, find_filename, find_any_filename, 
    find_filename_or_new,
    read_clusters, read_cluster_info, read_group_info,)
from loader import (default_cluster_info, default_group_info)
from klatools import kla_to_json, write_kla
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
    # If the ACLUINFO does not exist, try CLUINFO (older file extension)
    elif 'cluinfo' in filenames and os.path.exists(filenames['cluinfo']):
        data['acluinfo'] = read_cluster_info(filenames['cluinfo'])
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
    if 'spk' in filenames and os.path.exists(filenames['spk'] or ''):
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
    filenames_shanks = {}
    for index in indices:
        filenames_shanks[index] = triplet_to_filename(triplet[:2] + (index,))
    klusters_data = {index: open_klusters_oneshank(filename) 
        for index, filename in filenames_shanks.iteritems()}
    shanks = filenames_shanks.keys()
           
    # Find the dataset filenames and load the metadata.
    filenames = find_filenames(filename)
    # Metadata common to all shanks.
    metadata = read_xml(filenames['xml'], 1)
    # Metadata specific to each shank.
    metadata.update({shank: read_xml(filenames['xml'], shank)
        for shank in shanks})
    metadata['shanks'] = sorted(shanks)
    
    klusters_data['metadata'] = metadata
    klusters_data['filenames'] = filenames

    # Load probe file.
    filename_probe = filenames['probe']
    # It no probe file exists, create a default, linear probe with the right
    # number of channels per shank.
    if not filename_probe:
        # Generate a probe filename.
        filename_probe = find_filename_or_new(filename, 'default.probe',
            have_file_index=False)
        shanks = {shank: klusters_data[shank]['nchannels']
            for shank in filenames_shanks.keys()}
        probe_python = all_to_all_probe(shanks)
        with open(filename_probe, 'w') as f:
            f.write(probe_python)
        
    probe_ns = {}
    execfile(filename_probe, {}, probe_ns)
    klusters_data['probe'] = probe_ns

    return klusters_data

def create_hdf5_files(filename, klusters_data):
    hdf5 = {}
    
    hdf5_filenames = find_hdf5_filenames(filename)
    
    # Get the data corresponding to the first shank.
    # klusters_data_first = klusters_data.itervalues().next()
    
    # Create the HDF5 file.
    hdf5['klx'] = tables.openFile(hdf5_filenames['hdf5_klx'], mode='w')
    
    # Metadata.
    # for file in [hdf5['klx'], hdf5['wave_file']]:
    file = hdf5['klx']
    file.createGroup('/', 'shanks')
    file.createGroup('/', 'metadata')
    
    # Put the version number.
    file.setNodeAttr('/', 'VERSION', 1)
    
    # Get the old probe information, convert it to JSON, and save it in
    # /metadata
    # if 'probe' in klusters_data:
    # WARNING: the .probe file is mandatory.
    probe_text = probe_to_json(klusters_data['probe'])
    # else:
        # probe_text = ''
    file.setNodeAttr('/metadata', 'PRB_JSON', probe_text)
    
    # Read the old XML metadata and save the JSON parameters string.
    params_text = params_to_json(klusters_data['metadata'])
    file.setNodeAttr('/metadata', 'PRM_JSON', params_text)
    
    # Get the list of shanks.
    shanks = sorted([key for key in klusters_data.keys() 
        if isinstance(key, (int, long))])
    file.setNodeAttr('/metadata', 'SHANKS', np.unique(shanks))
            
    # Create groups and tables for each shank.
    for shank in shanks:
        data = klusters_data[shank]
        
        shank_path = '/shanks/shank{0:d}'.format(shank)
        
        # Create the /shanks/shank<X> groups in each file.
        file = hdf5['klx']
        file.createGroup('/shanks', 'shank{0:d}'.format(shank))
        
        
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
        self.filenames = self.klusters_data['filenames']
        
        # Backup the original CLU file.
        filename_clu_original = find_filename_or_new(self.filename, 'clu_original')
        shutil.copyfile(self.filenames['clu'], filename_clu_original)
        
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
        if 'spk' in data:
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
        # We set both manual and auto clustering to the current CLU file.
        row_main['cluster_manual'] = read['cluster']
        row_main['cluster_auto'] = read['cluster']
        row_main['features'] = read['fet']
        row_main['time'] = read['time']
        # if 'mask' in read:
        row_main['masks'] = (read['mask'] * 255).astype(np.uint8)
        row_main.append()
        
        # Fill the wave row.
        if 'spk' in read:
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
        
    def write_kla(self):
        kla = {
            shank: dict(
                cluster_colors=self.klusters_data[shank]['acluinfo']['color'],
                group_colors=self.klusters_data[shank]['groupinfo']['color'],
            ) for shank in self.shanks
        }
        write_kla(self.filenames['hdf5_kla'], kla)
        
    def convert(self):
        """Convert the old file format to the new HDF5-based format."""
        # Write the KLA file.
        self.write_kla()
        # Convert in HDF5 by going through all spikes.
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
        if hasattr(self, 'shanks'):
            for shank in self.shanks:
                for data in self.klusters_data[shank]:
                    if isinstance(data, (MemMappedBinary, MemMappedText)):
                        data.close()
        
        # # Close the KLA file.
        # if self.hdf5_data['kla']:
            # self.hdf5_data['kla'].close()
        
        # Close the HDF5 files.
        if self.hdf5_data['klx'].isopen:
            self.hdf5_data['klx'].flush()
            self.hdf5_data['klx'].close()
        
    def __del__(self):
        self.close()
        
    def __exit__(self, exception_type, exception_val, trace):
        self.close()




