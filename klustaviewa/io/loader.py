"""This module provides utility classes and functions to load spike sorting
data sets."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os.path
import re
from collections import Counter

import numpy as np
import pandas as pd

from tools import (find_filename, find_index, load_text, load_xml, normalize,
    load_binary, load_pickle, save_text, get_array, find_any_filename)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_indices)
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.utils.logger import debug, info, warn, exception
from klustaviewa.utils.colors import COLORS_COUNT


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

# def read_clusters_info(filename_cluinfo, fileindex):
    # info = load_pickle(filename_cluinfo)
    # return info
    
def read_features(filename_fet, nchannels, fetdim, freq):
    """Read a .fet file and return the normalize features array,
    as well as the spiketimes."""
    
    features = load_text(filename_fet, np.int32, skiprows=1)
    features = np.array(features, dtype=np.float32)
    
    # HACK: There are either 1 or 5 dimensions more than fetdim*nchannels
    # we can't be sure so we first try 1, if it does not work we try 5.
    for nextrafet in [1, 5]:
        try:
            features = features.reshape((-1,
                                         fetdim * nchannels + nextrafet))
            # if the features array could be reshape, directly break the loop
            break
        except ValueError:
            features = None
    if features is None:
        raise ValueError("""The number of columns in the feature matrix
        is not fetdim (%d) x nchannels (%d) + 1 or 5.""" % 
            (fetdim, nchannels))
    
    # get the spiketimes
    spiketimes = features[:,-1].copy()
    spiketimes *= (1. / freq)
    
    # count the number of extra features
    nextrafet = features.shape[1] - nchannels * fetdim
    
    # normalize normal features while keeping symmetry
    features[:,:-nextrafet] = normalize(features[:,:-nextrafet],
                                        symmetric=True)
    # normalize extra features without keeping symmetry
    features[:,-nextrafet:] = normalize(features[:,-nextrafet:],
                                        symmetric=False)
    
    return features, spiketimes
    
def read_clusters(filename_clu):
    clusters = load_text(filename_clu, np.int32)
    clusters = clusters[1:]
    return clusters

def read_cluster_info(filename_clusterinfo):
    # For each cluster (absolute indexing): cluster index, color index, 
    # and group index
    cluster_info = load_text(filename_clusterinfo, np.int32)
    cluster_info = pd.DataFrame({'color': cluster_info[:, 1], 
        'group': cluster_info[:, 2]}, dtype=np.int32, index=cluster_info[:, 0])
    return cluster_info
    
def read_group_info(filename_groups):
    # For each group (absolute indexing): color index, and name
    group_info = load_text(filename_groups, str, delimiter='\t')
    group_info = pd.DataFrame(
        {'color': group_info[:, 1].astype(np.int32),
         'name': group_info[:, 2]}, index=group_info[:, 0].astype(np.int32))
    return group_info
    
def read_masks(filename_mask, fetdim):
    masks_full = load_text(filename_mask, np.float32, skiprows=1)
    masks = masks_full[:,:-1:fetdim]
    return masks, masks_full
    
def read_waveforms(filename_spk, nsamples, nchannels):
    waveforms = np.array(load_binary(filename_spk), dtype=np.float32)
    waveforms = normalize(waveforms)
    waveforms = waveforms.reshape((-1, nsamples, nchannels))
    return waveforms

def read_probe(filename_probe):
    return normalize(np.array(load_text(filename_probe, np.int32),
        dtype=np.float32))


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
    save_text(filename_clu, clusters, header=len(Counter(clusters)))


# -----------------------------------------------------------------------------
# Generic Loader class
# -----------------------------------------------------------------------------
class Loader(object):
    
    # Initialization methods
    # ----------------------
    def __init__(self, filename=None):
        """Initialize a Loader object for loading Klusters-formatted files.
        
        Arguments:
          * filename: the full path of any file belonging to the same
            dataset.
        
        """
        self.spikes_selected = None
        self.clusters_selected = None
        if filename:
            self.open(filename)
    
    def open(self, filename):
        pass
    
    
    # Input-Output methods
    # --------------------
    def read(self):
        pass
    
    def save(self):
        pass
    
    
    # Access to the data: spikes
    # --------------------------
    def select(self, spikes=None, clusters=None):
        pass
    
    def get_clusters_selected(self):
        return self.clusters_selected
        
    def get_clusters_unique(self):
        return self.clusters_unique
    
    def get_features(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        return select(self.features, spikes)
    
    def get_spiketimes(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        return select(self.spiketimes, spikes)
    
    def get_clusters(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        return select(self.clusters, spikes)
    
    def get_masks(self, spikes=None, full=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        if not full:
            masks = self.masks
        else:
            masks = self.masks_full
        return select(masks, spikes)
    
    def get_waveforms(self, spikes=None, clusters=None):
        if spikes is not None:
            return select(self.waveforms, spikes)
        else:
            if clusters is None:
                clusters = self.clusters_selected
            if clusters is not None:
                spikes = get_some_spikes_in_clusters(clusters, self.clusters,
                    nspikes_max_expected=USERPREF['waveforms_nspikes_max_expected'],
                    nspikes_per_cluster_min=USERPREF['waveforms_nspikes_per_cluster_min'])
            else:
                spikes = self.spikes_selected
        return select(self.waveforms, spikes)
    
    def get_spikes(self, clusters=None):
        if clusters is None:
            clusters = self.clusters_selected
        return get_indices(self.get_clusters(clusters=clusters))
    
    
    # Access to the data: clusters
    # ----------------------------
    def get_cluster_colors(self, clusters=None):
        if clusters is None:
            clusters = self.clusters_selected
        return select(self.cluster_colors, clusters)
    
    def get_cluster_groups(self, clusters=None):
        if clusters is None:
            clusters = self.clusters_selected
        return select(self.cluster_groups, clusters)
    
    def get_group_colors(self, groups=None):
        return select(self.group_colors, groups)
    
    def get_group_names(self, groups=None):
        return select(self.group_names, groups)
    
    def get_cluster_sizes(self, clusters=None):
        if clusters is None:
            clusters = self.clusters_selected
        counter = Counter(self.clusters)
        sizes = pd.Series(counter, dtype=np.int32)
        return select(sizes, clusters)
    
        
    # Access to the data: misc
    # ------------------------
    def get_probe(self):
        return self.probe
    
    def get_new_clusters(self, n=1):
        return self.clusters.max() + np.arange(1, n + 1, dtype=np.int32)
    
    def get_new_group(self):
        groups = get_indices(self.group_names).values
        if len(groups) > 0:
            return groups.max() + 1
        else:
            return 0
    
    
    # Control methods
    # ---------------
    def update_clusters_unique(self):
        counter = Counter(self.clusters)
        self.clusters_unique = np.array(sorted(counter.keys()))
        
    # Set.
    def set_cluster(self, spikes, cluster):
        self.clusters.ix[spikes] = cluster
        self.update_clusters_unique()
        
    def set_cluster_groups(self, clusters, group):
        self.cluster_groups.ix[clusters] = group
        
    def set_cluster_colors(self, clusters, color):
        self.cluster_colors.ix[clusters] = color
        
    def set_group_names(self, groups, name):
        self.group_names.ix[groups] = name
        
    def set_group_colors(self, groups, color):
        self.group_colors.ix[groups] = color
        
    # Add.
    def add_cluster(self, cluster, group, color):
        if cluster not in self.cluster_groups.index:
            self.cluster_groups = self.cluster_groups.append(
                pd.Series([group], index=[cluster])).sort_index()
        if cluster not in self.cluster_colors.index:
            self.cluster_colors = self.cluster_colors.append(
                pd.Series([color], index=[cluster])).sort_index()
        
    def add_group(self, group, name, color):
        if group not in self.group_colors.index:
            self.group_colors = self.group_colors.append(
                pd.Series([color], index=[group])).sort_index()
        if group not in self.group_names.index:
            self.group_names = self.group_names.append(
                pd.Series([name], index=[group])).sort_index()
        
    # Remove.
    def remove_cluster(self, cluster):
        if np.any(np.in1d(cluster, self.clusters)):
            raise ValueError(("Cluster {0:d} is not empty and cannot "
            "be removed.").format(cluster))
        if cluster in self.cluster_groups.index:
            self.cluster_groups = self.cluster_groups.drop(cluster)
        if cluster in self.cluster_colors.index:
            self.cluster_colors = self.cluster_colors.drop(cluster)
            
    def remove_group(self, group):
        if np.any(np.in1d(group, self.cluster_groups)):
            raise ValueError(("Group {0:d} is not empty and cannot "
            "be removed.").format(group))
        if group in self.group_colors.index:
            self.group_colors = self.group_colors.drop(group)
        if group in self.group_names.index:
            self.group_names = self.group_names.drop(group)
    
    def remove_empty_clusters(self):
        clusters_all = self.cluster_groups.index
        clusters_in_data = self.clusters_unique
        clusters_empty = sorted(set(clusters_all) - set(clusters_in_data))
        if len(clusters_empty) > 0:
            debug("Removing empty clusters {0:s}.".
                format(str(clusters_empty)))
            for cluster in clusters_empty:
                self.remove_cluster(cluster)
        return clusters_empty
        
    
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
        self.filename_clu = find_filename(self.filename, 'clu')
        self.filename_clu_klustaviewa = find_filename(self.filename, 'clu')
        # self.filename_clu_klustaviewa = self.filename_clu.replace(
            # '.clu.', '.clu_klustaviewa.')
        self.filename_clusterinfo = find_filename(self.filename, 
            'cluinfo') or self.filename_clu.replace(
            '.clu.', '.cluinfo.')
        self.filename_groups = (find_filename(self.filename, 'groupinfo') or 
            self.filename_clu.replace('.clu.', '.groupinfo.'))
        # fmask or mask file
        self.filename_mask = find_filename(self.filename, 'fmask')
        if not self.filename_mask:
            self.filename_mask = find_filename(self.filename, 'mask')
        self.filename_spk = find_filename(self.filename, 'spk')
        self.filename_probe = (find_filename(self.filename, 'probe') or 
            find_any_filename(self.filename, 'probe'))
        
    def save_original_clufile(self):
        filename_clu_original = find_filename(self.filename, 'clu_original')
        if filename_clu_original is None:
            # Save the original clu file if it does not exist yet.
            with open(self.filename_clu, 'r') as f:
                clu = f.read()
            with open(self.filename_clu.replace('.clu.', 
                '.clu_original.'), 'w') as f:
                f.write(clu)
    
    
    # Internal read methods.
    # ----------------------
    def read_metadata(self):
        try:
            self.metadata = read_xml(self.filename_xml, self.fileindex)
        except IOError:
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
            except IOError as e:
                info(("There was an error while loading the probe file "
                          "'{0:s}' : {1:s}").format(self.filename_probe,
                            e.message))
                self.probe = None
    
    def read_features(self):
        try:
            self.features, self.spiketimes = read_features(self.filename_fet,
                self.nchannels, self.fetdim, self.freq)
        except IOError:
            raise IOError("The FET file is missing.")
        # Convert to Pandas.
        self.features = pd.DataFrame(self.features, dtype=np.float32)
        self.spiketimes = pd.Series(self.spiketimes, dtype=np.float32)
        
        # Count the number of spikes and save it in the metadata.
        self.nspikes = self.features.shape[0]
        self.metadata['nspikes'] = self.nspikes
        self.nextrafet = self.features.shape[1] - self.nchannels * self.fetdim
    
    def read_clusters(self):
        try:
            self.clusters = read_clusters(self.filename_clu)
        except IOError:
            warn("The CLU file is missing.")
            # Default clusters if the CLU file is not available.
            self.clusters = np.zeros(self.nspikes + 1, dtype=np.int32)
            self.clusters[0] = 1
        # Convert to Pandas.
        self.clusters = pd.Series(self.clusters, dtype=np.int32)
        
        # Counter clusters.
        counter = Counter(self.clusters)
        self.nclusters = len(counter)
        self.clusters_unique = np.array(sorted(counter.keys()))
    
    def read_cluster_info(self):
        try:
            self.cluster_info = read_cluster_info(self.filename_clusterinfo)
        except IOError:
            info("The CLUINFO file is missing, generating a default one.")
            n = len(self.clusters_unique)
            self.cluster_info = np.zeros((n, 3), dtype=np.int32)
            self.cluster_info[:, 0] = self.clusters_unique
            self.cluster_info[:, 1] = np.mod(np.arange(n, 
                dtype=np.int32), COLORS_COUNT) + 1
            # First column: color index, second column: group index (2 by
            # default)
            self.cluster_info[:, 2] = 2 * np.ones(n)
                
            self.cluster_info = pd.DataFrame({
                'color': self.cluster_info[:, 1],
                'group': self.cluster_info[:, 2]},
                dtype=np.int32, index=self.cluster_info[:, 0])
                
        assert np.array_equal(self.cluster_info.index, self.clusters_unique), \
            "The CLUINFO file does not correspond to the loaded CLU file."
            
        self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)
        
    def read_group_info(self):
        try:
            self.group_info = read_group_info(self.filename_groups)
        except IOError:
            info("The GROUPINFO file is missing, generating a default one.")
            self.group_info = np.zeros((3, 3), dtype=object)
            self.group_info[:, 0] = np.arange(3)
            self.group_info[:, 1] = (#np.array(
                np.mod(np.arange(3), COLORS_COUNT) + 1)#, dtype=str)
            self.group_info[:, 2] = np.array(['Noise', 'MUA', 'Good'],
                dtype=object)
            self.group_info = pd.DataFrame(
                {'color': self.group_info[:, 1].astype(np.int32),
                 'name': self.group_info[:, 2]},
                 index=self.group_info[:, 0].astype(np.int32))
        # Convert to Pandas.
        # self.group_info = pd.DataFrame(self.group_info)
        # self.group_info[0] = self.group_info[0].astype(np.int32)
        self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']
        
    def read_masks(self):
        try:
            self.masks, self.masks_full = read_masks(self.filename_mask,
                                                     self.fetdim)
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
        except IOError:
            warn("The SPK file is missing.")
            self.waveforms = np.zeros((self.nspikes, self.nsamples, 
                self.nchannels))
        # Convert to Pandas.
        self.waveforms = pd.Panel(self.waveforms, dtype=np.float32)
    
    def read_stats(self):
        self.ncorrbins = 50
        self.corrbin = .001
    
    
    # Internal save methods.
    # ----------------------
    def update_cluster_info(self):
        cluster_info = {
            'color': self.cluster_colors,
            'group': self.cluster_groups,
        }
        self.cluster_info = pd.DataFrame(cluster_info, dtype=np.int32)
    
    def update_group_info(self):
        group_info = {
            'color': self.group_colors,
            'name': self.group_names,
        }
        self.group_info = pd.DataFrame(group_info)
    
    def save_clusters(self):
        save_clusters(self.filename_clu_klustaviewa, get_array(self.clusters))
        
    def save_cluster_info(self):
        self.update_cluster_info()
        save_cluster_info(self.filename_clusterinfo, self.cluster_info)
    
    def save_group_info(self):
        self.update_group_info()
        save_group_info(self.filename_groups, self.group_info)
    
    
    # Public methods.
    # ---------------
    def read(self):
        info("Opening {0:s}.".format(self.filename))
        self.read_metadata()
        self.read_probe()
        self.read_features()
        self.read_clusters()
        self.read_cluster_info()
        self.read_group_info()
        self.read_masks()
        self.read_waveforms()
        self.read_stats()
    
    def save(self):
        self.save_clusters()
        self.save_cluster_info()
        self.save_group_info()
    
    def select(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)    
        self.spikes_selected = spikes
        self.clusters_selected = clusters

    def unselect(self):
        self.select(spikes=None, clusters=None)
        