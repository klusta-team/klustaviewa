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
import tables as tb
from galry import QtGui, QtCore

from loader import (Loader, default_group_info, reorder, renumber_clusters,
    default_cluster_info)
from klustersloader import find_filenames
from hdf5tools import klusters_to_hdf5
from tools import (load_text, load_xml, normalize,
    load_binary, load_pickle, save_text, get_array,
    first_row, load_binary_memmap)
from probe import (load_probe_json)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_some_spikes, get_indices, pandaize)
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.utils.settings import SETTINGS
from klustaviewa.utils.logger import (debug, info, warn, exception, FileLogger,
    register, unregister)
from klustaviewa.utils.colors import COLORS_COUNT, generate_colors


# -----------------------------------------------------------------------------
# HDF5 Loader
# -----------------------------------------------------------------------------
class HDF5Loader(Loader):
    # Read functions.
    # ---------------
    def open(self, filename):
        """Open a file."""
        filenames = find_filenames(filename)
        filename_main = filenames['hdf5_main']
        self.filename_log = filenames['kvwlg']
        # Conversion if the main HDF5 file does not exist.
        if not os.path.exists(filename_main):
            klusters_to_hdf5(filename, self.klusters_to_hdf5_progress_report)
        self.filename = filename_main
        self.read()
       
    def klusters_to_hdf5_progress_report(self, spike, nspikes, shank, nshanks):
        count = 100 * nshanks
        index = int((shank + spike * 1. / nspikes) * 100)
        self.report_progress(index, count)
       
    def read(self):
        """Open a HDF5 main file."""
        
        self.initialize_logfile()
        # Load the similarity measure chosen by the user in the preferences
        # file: 'gaussian' or 'kl'.
        # Refresh the preferences file when a new file is opened.
        USERPREF.refresh()
        self.similarity_measure = USERPREF['similarity_measure'] or 'gaussian'
        info("Similarity measure: {0:s}.".format(self.similarity_measure))
        info("Opening {0:s}.".format(self.filename))
        
        self.main = tb.openFile(self.filename)
        # Get the list of shanks.
        self.shanks = [int(re.match("shank([0-9]+)", 
            shank._v_name).group(1)[0])
                for shank in self.main.listNodes('/shanks')]
        self.read_metadata()
        # By default, read the first available shank.
        self.set_shank(self.shanks[0])
        
    
    # Shank functions.
    # ----------------
    def read_shank(self, shank=None):
        """Read the tables corresponding to a given shank."""
        if shank is not None:
            self.shank = shank
        # Get the tables.
        self.spike_table = self.main.getNode(
            self.shank_path + '/spikes')
        # self.has_masks = 'mask' in self.spike_table.coldescrs
        # self.fetcol = self.spike_table.coldescrs['features'].shape[0]
        self.wave_table = self.main.getNode(
            self.shank_path + '/waveforms')
        self.clusters_table = self.main.getNode(
            self.shank_path + '/clusters')
        self.groups_table = self.main.getNode(
            self.shank_path + '/groups')
        # Get the contents.
        self.read_shank_metadata()
        self.read_clusters()
        self.read_spiketimes()
        self.read_cluster_info()
        self.read_group_info()
        self.read_arrays()
        
    def get_shanks(self):
        """Return the list of shanks available in the file."""
        return self.shanks
        
    def set_shank(self, shank):
        """Change the current shank and read the corresponding tables."""
        if not shank in self.shanks:
            warn("Shank {0:d} is not in the list of shanks: {1:s}".format(
                shank, str(self.shanks)))
        self.shank = shank
        self.shank_path = '/shanks/shank{0:d}'.format(self.shank)
        self.read_shank()
    
    
    # Read contents.
    # --------------
    def read_metadata(self):
        """Read the metadata in /metadata."""
        self.freq = self.main.getNodeAttr('/metadata', 'freq')
        probe_json = self.main.getNodeAttr('/metadata', 'probe') or None
        if probe_json:
            self.probe = load_probe_json(probe_json)
        else:
            self.probe = None
        
    def get_probe_geometry(self):
        if self.probe:
            return self.probe[self.shank]['geometry_array']
        else:
            return None
        
    def read_shank_metadata(self):
        """Read the per-shank metadata in /shanks/shank<X>/metadata."""
        self.fetdim = self.main.getNodeAttr(
            self.shank_path + '/metadata', 'fetdim')
        self.nsamples = self.main.getNodeAttr(
            self.shank_path + '/metadata', 'nsamples')
        self.nchannels = self.main.getNodeAttr(
            self.shank_path + '/metadata', 'nchannels')
    
    def read_clusters(self):
        clusters = self.spike_table.col('cluster')
        self.nspikes = clusters.shape[0]
        # Convert to Pandas.
        self.clusters = pd.Series(clusters, dtype=np.int32)
        # Count clusters.
        self._update_data()
    
    def read_spiketimes(self):
        spiketimes = self.spike_table.col('time') * (1. / self.freq)
        # Convert to Pandas.
        self.spiketimes = pd.Series(spiketimes, dtype=np.float32)
        self.duration = spiketimes[-1]
    
    def read_cluster_info(self):
        # Read the cluster info.
        clusters = self.clusters_table.col('cluster')
        cluster_colors = self.clusters_table.col('color')
        cluster_groups = self.clusters_table.col('group')
        
        # Create the cluster_info DataFrame.
        self.cluster_info = pd.DataFrame(dict(
            color=cluster_colors,
            group=cluster_groups,
            ), index=clusters)
        self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)
        
    def read_group_info(self):
        # Read the group info.
        groups = self.groups_table.col('group')
        group_colors = self.groups_table.col('color')
        group_names = self.groups_table.col('name')
        
        # Create the group_info DataFrame.
        self.group_info = pd.DataFrame(dict(
            color=group_colors,
            name=group_names,
            ), index=groups)
        self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']
    
    
    # Read and process arrays.
    # ------------------------
    def process_features(self, y):
        x = y.copy()
        x[:,:-1] *= self.background_features_normalization
        x[:,-1] *= (1. / (self.duration * self.freq))
        x[:,-1] = 2 * x[:,-1] - 1
        return x
    
    def process_masks_full(self, masks_full):
        return (masks_full * 1. / 255).astype(np.float32)
    
    def process_masks(self, masks_full):
        return (masks_full[:,:-1:self.fetdim] * 1. / 255).astype(np.float32)
    
    def process_waveforms(self, waveforms):
        return (waveforms * 1e-5).astype(np.float32).reshape((-1, self.nsamples, self.nchannels))
    
    def read_arrays(self):
        self.nextrafet = (self.spike_table.cols.features.shape[1] - 
            self.nchannels * self.fetdim)
            
        self.features = self.spike_table, 'features', self.process_features
        self.masks_full = self.spike_table, 'masks', self.process_masks_full
        self.masks = self.spike_table, 'masks', self.process_masks
        # For the waveforms, need to dereference with __call__ as it
        # is an external link.
        self.waveforms = self.wave_table(), 'waveform', self.process_waveforms
        
        # Background spikes
        # -----------------
        # Used for:
        #   * background feature view
        #   * similarity matrix
        # Load background spikes for FeatureView.
        step = max(1, int(self.nspikes / (1000. * self.nclusters)))
        self.background_spikes = slice(0, self.nspikes, step)
        self.background_table = self.spike_table[self.background_spikes]
        self.background_features = self.background_table['features']
        # Find normalization factor for features.
        self.background_features_normalization = 1. / np.abs(
            self.background_features[:,:-1]).max()
        self.background_features = self.process_features(
            self.background_features)
        # self.background_features_pandas = pandaize(
            # self.background_features, self.background_spikes)
        self.background_masks = self.process_masks_full(
            self.background_table['masks'])
        self.background_clusters = self.background_table['cluster']
        

    # Access to the data: spikes
    # --------------------------
    def select(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        # HDD access here: get the portion of the table with the requested 
        # clusters (cache). It is very quick to access the different columns
        # from this in-memory table later.
        self.spikes_selected_table = self.spike_table[spikes]
        self.spikes_selected = spikes
        self.clusters_selected = clusters

    def get_features_background(self):
        # return self.background_features_pandas
        return pandaize(self.background_features, self.background_spikes)
    
    def get_features(self, spikes=None, clusters=None):
        # Special case: return the already-selected values from the cache.
        if spikes is None and clusters is None:
            features = self.spikes_selected_table['features']
            values = self.process_features(features)
            return pandaize(values, self.spikes_selected)
        # Normal case.
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        return select(self.features, spikes)
    
    # def get_some_features(self):#, clusters=None):
        # """Return the features for a subset of all spikes: a large number
        # of spikes from any cluster, and a controlled subset of the selected 
        # clusters."""
        # # Merge background features and all features.
        # features_bg = self.get_background_features()
        # features = self.get_features()
        # return pd.concat([features, features_bg])
        
    def get_masks(self, spikes=None, full=None, clusters=None):
        # Special case: return the already-selected values from the cache.
        if spikes is None and clusters is None:
            masks = self.spikes_selected_table['masks']
            if full:
                values = self.process_masks_full(masks)
            else:
                values = self.process_masks(masks)
            return pandaize(values, self.spikes_selected)
        # Normal case.
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
                    counter=self.counter,
                    nspikes_max_expected=USERPREF['waveforms_nspikes_max_expected'],
                    nspikes_per_cluster_min=USERPREF['waveforms_nspikes_per_cluster_min'])
            else:
                spikes = self.spikes_selected
        return select(self.waveforms, spikes)
    
    
    # Log file.
    # ---------
    def initialize_logfile(self):
        # filename = os.path.splitext(self.filename)[0] + '.kvwlg'
        self.logfile = FileLogger(self.filename_log, name='datafile', 
            level=USERPREF['loglevel_file'])
        # Register log file.
        register(self.logfile)
        
    
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
        
        # TODO
        # # Save both ACLU and CLU files.
        # save_clusters(self.filename_aclu, clusters)
        # save_clusters(self.filename_clu, 
            # convert_to_clu(clusters, cluster_info))
        
        # # Save CLUINFO and GROUPINFO files.
        # save_cluster_info(self.filename_acluinfo, cluster_info)
        # save_group_info(self.filename_groupinfo, self.group_info)
    
    
    # Close functions.
    # ----------------
    def close(self):
        """Close the main HDF5 file."""
        if hasattr(self, 'wave_table'):
            self.wave_table.umount()
        if hasattr(self, 'main'):
            self.main.flush()
            self.main.close()
        if hasattr(self, 'logfile'):
            unregister(self.logfile)
       
    # def __del__(self):
        # self.close()
        