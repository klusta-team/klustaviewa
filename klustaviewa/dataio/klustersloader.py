"""This module provides utility classes and functions to load spike sorting
data sets."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import os.path
import re
from collections import Counter

import numpy as np
import pandas as pd
from galry import QtGui, QtCore

from loader import (Loader, default_group_info, reorder, renumber_clusters,
    default_cluster_info)
from tools import (find_filename, find_index, load_text, load_xml, normalize,
    load_binary, load_pickle, save_text, get_array, find_any_filename,
    first_row, load_binary_memmap)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_some_spikes, get_indices)
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.utils.settings import SETTINGS
from klustaviewa.utils.logger import (debug, info, warn, exception, FileLogger,
    register, unregister)
from klustaviewa.utils.colors import COLORS_COUNT, generate_colors


# -----------------------------------------------------------------------------
# File reading functions
# -----------------------------------------------------------------------------
def read_xml(filename_xml, fileindex):
    """Read the XML file associated to the current dataset,
    and return a metadata dictionary."""
    
    params = load_xml(filename_xml, fileindex=fileindex)
    
    # klusters tests
    metadata = dict(
        nchannels=params['nchannels'],
        nsamples=params['nsamples'],
        fetdim=params['fetdim'],
        freq=params['rate'])
    
    return metadata

# Features.
def process_features(features, fetdim, nchannels, freq, nfet=None):
    features = np.array(features, dtype=np.float32)
    nspikes, ncol = features.shape
    if nfet is not None:
        nextrafet = nfet - fetdim * nchannels
    else:
        nextrafet = ncol - fetdim * nchannels
            
    # get the spiketimes
    spiketimes = features[:,-1].copy()
    spiketimes *= (1. / freq)
    
    # normalize normal features while keeping symmetry
    features_normal = normalize(features[:,:fetdim * nchannels],
                                        symmetric=True)
    # features_time = normalize(features[:,[-1]],
                                        # symmetric=False)
    features_time = spiketimes.reshape((-1, 1)) * 1. / spiketimes[-1] * 2 - 1
    # normalize extra features without keeping symmetry
    if nextrafet > 1:
        features_extra = normalize(features[:,-nextrafet:-1],
                                            symmetric=False)
        features = np.hstack((features_normal, features_extra, features_time))
    else:
        features = np.hstack((features_normal, features_time))
    return features, spiketimes
    
def read_features(filename_fet, nchannels, fetdim, freq):
    """Read a .fet file and return the normalize features array,
    as well as the spiketimes."""
    try:
        features = load_text(filename_fet, np.int32, skiprows=1, delimiter=' ')
    except ValueError:
        features = load_text(filename_fet, np.float32, skiprows=1, delimiter='\t')
    return process_features(features, fetdim, nchannels, freq, 
        nfet=first_row(filename_fet))
    
# Clusters.
def process_clusters(clusters):
    return clusters[1:]

def read_clusters(filename_clu):
    clusters = load_text(filename_clu, np.int32)
    return process_clusters(clusters)

# RES file.
def process_res(spiketimes, freq=None):
    if freq is None:
        return spiketimes
    else:
        return spiketimes * 1. / freq

def read_res(filename_res, freq=None):
    res = load_text(filename_res, np.int32)
    return process_res(res, freq)

# Cluster info.
def process_cluster_info(cluster_info):
    cluster_info = pd.DataFrame({'color': cluster_info[:, 1], 
        'group': cluster_info[:, 2]}, dtype=np.int32, index=cluster_info[:, 0])
    return cluster_info
    
def read_cluster_info(filename_clusterinfo):
    # For each cluster (absolute indexing): cluster index, color index, 
    # and group index
    cluster_info = load_text(filename_clusterinfo, np.int32)
    return process_cluster_info(cluster_info)
    
# Group info.
def process_group_info(group_info):
    group_info = pd.DataFrame(
        {'color': group_info[:, 1].astype(np.int32),
         'name': group_info[:, 2]}, index=group_info[:, 0].astype(np.int32))
    return group_info

def read_group_info(filename_groups):
    # For each group (absolute indexing): color index, and name
    group_info = load_text(filename_groups, str, delimiter='\t')
    return process_group_info(group_info)
    
# Masks.
def process_masks(masks_full, fetdim):
    masks = masks_full[:,:-1:fetdim]
    return masks, masks_full

def read_masks(filename_mask, fetdim):
    masks_full = load_text(filename_mask, np.float32, skiprows=1)
    return process_masks(masks_full, fetdim)
    
# Waveforms.
def process_waveforms(waveforms, nsamples, nchannels):
    waveforms = np.array(waveforms, dtype=np.float32)
    waveforms = normalize(waveforms, symmetric=True)
    waveforms = waveforms.reshape((-1, nsamples, nchannels))
    return waveforms

def read_waveforms(filename_spk, nsamples, nchannels):
    waveforms = np.array(load_binary(filename_spk), dtype=np.float32)
    n = waveforms.size
    if n % nsamples != 0 or n % nchannels != 0:
        waveforms = load_text(filename_spk, np.float32)
    return process_waveforms(waveforms, nsamples, nchannels)
    
# DAT.
def read_dat(filename_dat, nchannels):
    nsamples = os.path.getsize(filename_dat) // nchannels
    return load_binary_memmap(filename_dat, dtype=np.int16,
                             shape=(nsamples, nchannels))

# Probe.
def process_probe(probe):
    return normalize(probe)

def read_probe(filename_probe):
    if os.path.exists(filename_probe):
        # Try the text-flavored probe file.
        try:
            probe = load_text(filename_probe, np.float32)
        except:
            # Or try the Python-flavored probe file (SpikeDetekt, with an
            # extra field 'geometry').
            ns = {}
            execfile(filename_probe, ns)
            probe = ns['geometry']
            probe = np.array([probe[i] for i in sorted(probe.keys())],
                                dtype=np.float32)
        return process_probe(probe)


# -----------------------------------------------------------------------------
# File saving functions
# -----------------------------------------------------------------------------
def save_cluster_info(filename_cluinfo, cluster_info):
    cluster_info_array = np.hstack((cluster_info.index.reshape((-1, 1)), 
        cluster_info.values))
    save_text(filename_cluinfo, cluster_info_array)
    
def save_group_info(filename_groupinfo, group_info):
    group_info_array = np.hstack((group_info.index.reshape((-1, 1)), 
        group_info.values))
    save_text(filename_groupinfo, group_info_array, fmt='%s', delimiter='\t')
    
def save_clusters(filename_clu, clusters):
    save_text(filename_clu, clusters, header=len(np.unique(clusters)))

def convert_to_clu(clusters, cluster_info):
    cluster_groups = cluster_info['group']
    clusters_new = np.array(clusters, dtype=np.int32) + 2
    for i in (0, 1):
        clusters_new[cluster_groups.ix[clusters] == i] = i
    clusters_unique = np.unique(set(clusters_new).union(set([0, 1])))
    clusters_renumbered = reorder(clusters_new, clusters_unique)
    return clusters_renumbered

    
# -----------------------------------------------------------------------------
# Klusters Loader
# -----------------------------------------------------------------------------
class KlustersLoader(Loader):
    def open(self, filename):
        """Open a file."""
        self.filename = filename
        # Find the file index associated to the filename, or 1 by default.
        self.fileindex = find_index(filename) or 1
        self.find_filenames()
        self.save_original_clufile()
        self.read()
        
    def find_filenames(self):
        """Find the filenames of the different files for the current
        dataset."""
        self.filename_xml = find_filename(self.filename, 'xml')
        self.filename_fet = find_filename(self.filename, 'fet')
        self.filename_clu = (find_filename(self.filename, 'clu') or
            self.filename_fet.replace('.fet.', '.clu.'))
        filename = self.filename_fet or self.filename_clu.replace('.clu.', '.fet.')
        self.filename_res = find_filename(self.filename, 'res')
        self.filename_aclu = (find_filename(self.filename, 'aclu') or
            filename.replace('.fet.', '.aclu.'))
        # TODO: improve this bad looking code
        self.filename_clusterinfo = find_filename(self.filename, 
            'acluinfo') or filename.replace(
            '.fet.', '.acluinfo.')
        self.filename_groups = (find_filename(self.filename, 'groupinfo') or 
            filename.replace('.fet.', '.groupinfo.'))
        # fmask or mask file
        self.filename_mask = find_filename(self.filename, 'fmask')
        if not self.filename_mask:
            self.filename_mask = find_filename(self.filename, 'mask')
        self.filename_spk = find_filename(self.filename, 'spk')
        self.filename_dat = find_filename(self.filename, 'dat')
        self.filename_probe = (find_filename(self.filename, 'probe') or 
            find_any_filename(self.filename, 'probe'))
        
    def save_original_clufile(self):
        filename_clu_original = find_filename(self.filename, 'clu_original')
        if filename_clu_original is None:
            if os.path.exists(self.filename_clu):
                # Save the original clu file if it does not exist yet.
                with open(self.filename_clu, 'r') as f:
                    clu = f.read()
                with open(self.filename_clu.replace('.clu.', 
                    '.clu_original.'), 'w') as f:
                    f.write(clu)
            if os.path.exists(self.filename_aclu):
                # Save the original clu file if it does not exist yet.
                with open(self.filename_aclu, 'r') as f:
                    clu = f.read()
                with open(self.filename_aclu.replace('.aclu.', 
                    '.aclu_original.'), 'w') as f:
                    f.write(clu)
    
    
    # Internal read methods.
    # ----------------------
    def read_metadata(self):
        try:
            self.metadata = read_xml(self.filename_xml, self.fileindex)
        except:
            # Die if no XML file is available for this dataset, as it contains
            # critical metadata.
            raise IOError("The XML file is missing.")
            
        self.nsamples = self.metadata.get('nsamples')
        self.nchannels = self.metadata.get('nchannels')
        self.fetdim = self.metadata.get('fetdim')
        self.freq = self.metadata.get('freq')
        
    def read_probe(self):
        if self.filename_probe is None:
            info("No probe file has been found.")
            self.probe = None
        else:
            try:
                self.probe = read_probe(self.filename_probe)
                info("Successfully loaded {0:s}".format(self.filename_probe))
            except Exception as e:
                info(("There was an error while loading the probe file "
                          "'{0:s}' : {1:s}").format(self.filename_probe,
                            e.message))
                self.probe = None
    
    def read_features(self):
        try:
            self.features, self.spiketimes = read_features(self.filename_fet,
                self.nchannels, self.fetdim, self.freq)
            info("Successfully loaded {0:s}".format(self.filename_fet))
        except IOError:
            raise IOError("The FET file is missing.")
        # Convert to Pandas.
        self.features = pd.DataFrame(self.features, dtype=np.float32)
        self.duration = self.spiketimes[-1]
        self.spiketimes = pd.Series(self.spiketimes, dtype=np.float32)
        
        # Count the number of spikes and save it in the metadata.
        self.nspikes = self.features.shape[0]
        self.metadata['nspikes'] = self.nspikes
        self.nextrafet = self.features.shape[1] - self.nchannels * self.fetdim
    
    def read_res(self):
        try:
            self.spiketimes_res = read_res(self.filename_res, self.freq)
            self.spiketimes_res = pd.Series(self.spiketimes_res, dtype=np.float32)
        except IOError:
            warn("The RES file is missing.")
    
    def read_clusters(self):
        try:
            # Try reading the ACLU file, or fallback on the CLU file.
            if os.path.exists(self.filename_aclu):
                self.clusters = read_clusters(self.filename_aclu)
                info("Successfully loaded {0:s}".format(self.filename_aclu))
            else:
                self.clusters = read_clusters(self.filename_clu)
                info("Successfully loaded {0:s}".format(self.filename_clu))
        except IOError:
            warn("The CLU file is missing.")
            # Default clusters if the CLU file is not available.
            self.clusters = np.zeros(self.nspikes, dtype=np.int32)
        # Convert to Pandas.
        self.clusters = pd.Series(self.clusters, dtype=np.int32)
        
        # Count clusters.
        self._update_data()
    
    def read_cluster_info(self):
        try:
            self.cluster_info = read_cluster_info(self.filename_clusterinfo)
            info("Successfully loaded {0:s}".format(self.filename_clusterinfo))
        except IOError:
            info("The CLUINFO file is missing, generating a default one.")
            self.cluster_info = default_cluster_info(self.clusters_unique)
                
        if not np.array_equal(self.cluster_info.index, self.clusters_unique):
            info("The CLUINFO file does not correspond to the loaded CLU file.")
            self.cluster_info = default_cluster_info(self.clusters_unique)
            
        self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)
        
    def read_group_info(self):
        try:
            self.group_info = read_group_info(self.filename_groups)
            info("Successfully loaded {0:s}".format(self.filename_groups))
        except IOError:
            info("The GROUPINFO file is missing, generating a default one.")
            self.group_info = default_group_info()
        
        # Convert to Pandas.
        self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']
        
    def read_masks(self):
        try:
            self.masks, self.masks_full = read_masks(self.filename_mask,
                                                     self.fetdim)
            info("Successfully loaded {0:s}".format(self.filename_mask))
        except IOError:
            warn("The MASKS/FMASKS file is missing.")
            # Default masks if the MASK/FMASK file is not available.
            self.masks = np.ones((self.nspikes, self.nchannels))
            self.masks_full = np.ones(self.features.shape)
        self.masks = pd.DataFrame(self.masks)
        self.masks_full = pd.DataFrame(self.masks_full)
    
    def read_waveforms(self):
        try:
            self.waveforms = read_waveforms(self.filename_spk, self.nsamples,
                                            self.nchannels)
            info("Successfully loaded {0:s}".format(self.filename_spk))
        except IOError:
            warn("The SPK file is missing.")
            self.waveforms = np.zeros((self.nspikes, self.nsamples, 
                self.nchannels))
        # Convert to Pandas.
        self.waveforms = pd.Panel(self.waveforms, dtype=np.float32)
    
    def read_dat(self):
        try:
            self.dat = read_dat(self.filename_dat, self.nchannels)
        except IOError:
            warn("The DAT file is missing.")
    
    def read_fil(self):
        try:
            self.fil = read_dat(self.filename_fil, self.nchannels)
        except IOError:
            warn("The FIL file is missing.")
    
    def read_stats(self):
        self.ncorrbins = SETTINGS.get('correlograms.ncorrbins', 100)
        self.corrbin = SETTINGS.get('correlograms.corrbin', .001)

        
    # Log file.
    # ---------
    def initialize_logfile(self):
        filename = self.filename_fet.replace('.fet.', '.kvwlg.')
        self.logfile = FileLogger(filename, name='datafile', 
            level=USERPREF['loglevel_file'])
        # Register log file.
        register(self.logfile)
        
    
    # Public methods.
    # ---------------
    def read(self):
        self.initialize_logfile()
        # Load the similarity measure chosen by the user in the preferences
        # file: 'gaussian' or 'kl'.
        # Refresh the preferences file when a new file is opened.
        USERPREF.refresh()
        self.similarity_measure = USERPREF['similarity_measure'] or 'gaussian'
        info("Similarity measure: {0:s}.".format(self.similarity_measure))
        info("Opening {0:s}.".format(self.filename))
        self.report_progress(0, 5)
        self.read_metadata()
        self.read_probe()
        self.report_progress(1, 5)
        self.read_features()
        self.report_progress(2, 5)
        self.read_res()
        self.read_clusters()
        self.report_progress(3, 5)
        self.read_cluster_info()
        self.read_group_info()
        self.read_masks()
        self.report_progress(4, 5)
        self.read_waveforms()
        self.report_progress(5, 5)
        self.read_stats()
    
    def save(self, renumber=False):
        self.update_cluster_info()
        self.update_group_info()
        
        if renumber:
            self.renumber()
            clusters = get_array(self.clusters_renumbered)
            cluster_info = self.cluster_info_renumbered
        else:
            clusters = get_array(self.clusters)
            cluster_info = self.cluster_info
        
        # Save both ACLU and CLU files.
        save_clusters(self.filename_aclu, clusters)
        save_clusters(self.filename_clu, 
            convert_to_clu(clusters, cluster_info))
        
        # Save CLUINFO and GROUPINFO files.
        save_cluster_info(self.filename_clusterinfo, cluster_info)
        save_group_info(self.filename_groups, self.group_info)
    
    def close(self):
        if hasattr(self, 'logfile'):
            unregister(self.logfile)
            
    def __del__(self):
        self.close()
        
    
# -----------------------------------------------------------------------------
# Memory Loader
# -----------------------------------------------------------------------------
class MemoryLoader(Loader):
    def __init__(self, parent=None, **kwargs):
        super(MemoryLoader, self).__init__(parent)
        self.read(**kwargs)
    
    
    # Internal read methods.
    # ----------------------
    def read_metadata(self, nsamples=None, nchannels=None, fetdim=None,
        freq=None):
        self.nsamples = nsamples
        self.nchannels = nchannels
        self.fetdim = fetdim
        self.freq = freq
        
    def read_probe(self, probe):
        try:
            self.probe = process_probe(probe)
        except Exception as e:
            info(("There was an error while loading the probe: "
                      "'{0:s}'").format(e.message))
            self.probe = None
    
    def read_features(self, features):
        self.features, self.spiketimes = process_features(features,
            self.nchannels, self.fetdim, self.freq)
        # Convert to Pandas.
        self.features = pd.DataFrame(self.features, dtype=np.float32)
        self.duration = self.spiketimes[-1]
        self.spiketimes = pd.Series(self.spiketimes, dtype=np.float32)
        
        # Count the number of spikes and save it in the metadata.
        self.nspikes = self.features.shape[0]
        self.nextrafet = self.features.shape[1] - self.nchannels * self.fetdim
    
    def read_clusters(self, clusters):
        self.clusters = process_clusters(clusters)
        # Convert to Pandas.
        self.clusters = pd.Series(self.clusters, dtype=np.int32)
        # Count clusters.
        self._update_data()
    
    def read_cluster_info(self, cluster_info):
        self.cluster_info = process_cluster_info(cluster_info)
                
        assert np.array_equal(self.cluster_info.index, self.clusters_unique), \
            "The CLUINFO file does not correspond to the loaded CLU file."
            
        self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)
        
    def read_group_info(self, group_info):
        self.group_info = process_group_info(group_info)
        # Convert to Pandas.
        self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']
        
    def read_masks(self, masks):
        self.masks, self.masks_full = process_masks(masks, self.fetdim)
        self.masks = pd.DataFrame(self.masks)
        self.masks_full = pd.DataFrame(self.masks_full)
    
    def read_waveforms(self, waveforms):
        self.waveforms = process_waveforms(waveforms, self.nsamples,
                                        self.nchannels)
        # Convert to Pandas.
        self.waveforms = pd.Panel(self.waveforms, dtype=np.float32)
    
    def read_stats(self):
        self.ncorrbins = SETTINGS.get('correlograms.ncorrbins', 100)
        self.corrbin = SETTINGS.get('correlograms.corrbin', .001)
    
    
    # Public methods.
    # ---------------
    def read(self, nsamples=None, nchannels=None, fetdim=None,
            freq=None, probe=None, features=None, clusters=None,
            cluster_info=None, group_info=None, masks=None,
            waveforms=None):
        self.read_metadata(nsamples=nsamples, nchannels=nchannels,
            fetdim=fetdim, freq=freq)
        self.read_probe(probe)
        self.read_features(features)
        self.read_clusters(clusters)
        self.read_cluster_info(cluster_info)
        self.read_group_info(group_info)
        self.read_masks(masks)
        self.read_waveforms(waveforms)
        self.read_stats()
    
    
    