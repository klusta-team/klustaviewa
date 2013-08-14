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

from tools import (load_text, normalize,
    load_binary, load_pickle, save_text, get_array, 
    first_row, load_binary_memmap)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_some_spikes, get_indices)
# from klustaviewa import USERPREF
# from klustaviewa import SETTINGS
from klustaviewa.utils.logger import (debug, info, warn, exception, FileLogger,
    register, unregister)
from klustaviewa.utils.colors import COLORS_COUNT, generate_colors


# -----------------------------------------------------------------------------
# Default cluster/group info
# -----------------------------------------------------------------------------
def default_cluster_info(clusters_unique):
    n = len(clusters_unique)
    cluster_info = pd.DataFrame({
        'color': generate_colors(n),
        'group': 3 * np.ones(n)},
        dtype=np.int32,
        index=clusters_unique)
    # Put cluster 0 in group 0 (=noise), cluster 1 in group 1 (=MUA)
    if 0 in clusters_unique:
        cluster_info['group'][0] = 0
    if 1 in clusters_unique:
        cluster_info['group'][1] = 1
    return cluster_info

def default_group_info():
    group_info = np.zeros((4, 3), dtype=object)
    group_info[:, 0] = np.arange(4)
    group_info[:, 1] = generate_colors(group_info.shape[0])
    group_info[:, 2] = np.array(['Noise', 'MUA', 'Good', 'Unsorted'],
        dtype=object)
    group_info = pd.DataFrame(
        {'color': group_info[:, 1].astype(np.int32),
         'name': group_info[:, 2]},
         index=group_info[:, 0].astype(np.int32))
    return group_info


# -----------------------------------------------------------------------------
# Cluster renumbering
# -----------------------------------------------------------------------------
def reorder(x, order):
    x_reordered = np.zeros_like(x)
    for i, o in enumerate(order):
        x_reordered[x == o] = i
    return x_reordered

def renumber_clusters(clusters, cluster_info):
    clusters_unique = get_array(get_indices(cluster_info))
    nclusters = len(clusters_unique)
    assert np.array_equal(clusters_unique, np.unique(clusters))
    clusters_array = get_array(clusters)
    groups = get_array(cluster_info['group'])
    colors = get_array(cluster_info['color'])
    groups_unique = np.unique(groups)
    # Reorder clusters according to the group.
    clusters_unique_reordered = np.hstack(
        [sorted(clusters_unique[groups == group]) for group in groups_unique])
    # WARNING: there's a +2 offset to avoid conflicts with the old convention
    # cluster 0 = noise, cluster 1 = MUA.
    clusters_renumbered = reorder(clusters_array, clusters_unique_reordered) + 2
    cluster_permutation = reorder(clusters_unique_reordered, clusters_unique)
    # Reorder cluster info.
    groups_reordered = groups[cluster_permutation]
    colors_reordered = colors[cluster_permutation]
    # Recreate new cluster info.
    cluster_info_reordered = pd.DataFrame({'color': colors_reordered, 
        'group': groups_reordered}, dtype=np.int32, 
        index=(np.arange(nclusters) + 2))
    return clusters_renumbered, cluster_info_reordered


# -----------------------------------------------------------------------------
# Generic Loader class
# -----------------------------------------------------------------------------
class Loader(QtCore.QObject):
    progressReported = QtCore.pyqtSignal(int, int)
    saveProgressReported = QtCore.pyqtSignal(int, int)
    
    # Progress report
    # ---------------
    def report_progress(self, index, count):
        self.progressReported.emit(index, count)
        
    def report_progress_save(self, index, count):
        self.saveProgressReported.emit(index, count)
        
    
    # Initialization methods
    # ----------------------
    def __init__(self, parent=None, filename=None, userpref=None):
        """Initialize a Loader object for loading Klusters-formatted files.
        
        Arguments:
          * filename: the full path of any file belonging to the same
            dataset.
        
        """
        super(Loader, self).__init__()
        self.spikes_selected = None
        self.clusters_selected = None
        self.override_color = False
        
        if not userpref:
            # HACK: if no UserPref is given in argument to the loader,
            # use a mock dictionary returning None all the time.
            class MockDict(object):
                def __getitem__(self, name):
                    return None
            userpref = MockDict()
        self.userpref = userpref
        
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
    
    def close(self):
        pass
    
    
    # Access to the data: spikes
    # --------------------------
    def select(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)    
        self.spikes_selected = spikes
        self.clusters_selected = clusters

    def unselect(self):
        self.select(spikes=None, clusters=None)
        
    def get_clusters_selected(self):
        return self.clusters_selected
        
    def has_selection(self):
        return self.clusters_selected is not None and len(self.clusters_selected) > 0
        
    def get_clusters_unique(self):
        return self.clusters_unique
    
    def get_features(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        return select(self.features, spikes)
    
    def get_features_background(self):
        return self.features
        
    def get_some_features(self, clusters=None):
        """Return the features for a subset of all spikes: a large number
        of spikes from any cluster, and a controlled subset of the selected 
        clusters."""
        if clusters is None:
            clusters = self.clusters_selected
        if clusters is not None:
            spikes_background = get_some_spikes(self.clusters,
                nspikes_max=self.userpref['features_nspikes_background_max'],)
            spikes_clusters = get_some_spikes_in_clusters(
                clusters,
                self.clusters,
                counter=self.counter,
                nspikes_max_expected=self.userpref[
                    'features_nspikes_selection_max'],
                nspikes_per_cluster_min=self.userpref[
                    'features_nspikes_per_cluster_min'])
            spikes = np.union1d(spikes_background, spikes_clusters)
        else:
            spikes = self.spikes_selected
        return select(self.features, spikes)
        
    def get_spiketimes(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        spiketimes = getattr(self, 'spiketimes', getattr(self, 'spiketimes_res', None))
        return select(spiketimes, spikes)
    
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
                    counter=self.counter,
                    nspikes_max_expected=self.userpref['waveforms_nspikes_max_expected'],
                    nspikes_per_cluster_min=self.userpref['waveforms_nspikes_per_cluster_min'])
            else:
                spikes = self.spikes_selected
        return select(self.waveforms, spikes)
    
    def get_dat(self):
        return self.dat
    
    def get_spikes(self, clusters=None):
        if clusters is None:
            clusters = self.clusters_selected
        return get_indices(self.get_clusters(clusters=clusters))
    
    def get_duration(self):
        return self.duration
    
    
    # Access to the data: clusters
    # ----------------------------
    def get_cluster_colors(self, clusters=None, can_override=True,
            ):
        if clusters is None:
            clusters = self.clusters_selected
        if can_override and self.override_color:
            group_colors = get_array(self.get_group_colors('all'))
            groups = get_array(self.get_cluster_groups('all'))
            colors = pd.Series(group_colors[groups], 
                index=self.get_clusters_unique())
        else:
            colors = self.cluster_colors
        return select(colors, clusters)
    
    def get_cluster_color(self, cluster):
        try:
            return select(self.cluster_colors, cluster)
        except IndexError:
            return 0
    
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
        # counter = Counter(self.clusters)
        sizes = pd.Series(self.counter, dtype=np.int32)
        return select(sizes, clusters)
    
        
    # Access to the data: misc
    # ------------------------
    def get_probe(self):
        return self.probe
    
    def get_probe_geometry(self):
        return self.probe
    
    def get_new_clusters(self, n=1):
        return self.clusters.max() + np.arange(1, n + 1, dtype=np.int32)
    
    def get_next_cluster(self, cluster):
        cluster_groups = self.get_cluster_groups('all')
        group = self.get_cluster_groups(cluster)
        clusters = get_indices(cluster_groups)
        cluster_groups = get_array(cluster_groups)
        samegroup = (cluster_groups == group) & (clusters > cluster)
        i = np.nonzero(samegroup)[0]
        if len(i) > 0:
            return clusters[i[0]]
        else:
            return cluster
    
    def get_new_group(self):
        groups = get_indices(self.group_names).values
        if len(groups) > 0:
            return groups.max() + 1
        else:
            return 0
    
    # def get_correlogram_window(self):
        # return self.ncorrbins * self.corrbin
    
    def set_override_color(self, override_color):
        self.override_color = override_color
    
    
    # Control methods
    # ---------------
    def _update_data(self,):
        """Update internal variables."""
        clusters_array = get_array(self.clusters)
        self.clusters_unique = np.unique(clusters_array)
        self.nclusters = len(self.clusters_unique)
        bincount = np.bincount(clusters_array)
        self.counter = {key: bincount[key] for key in np.nonzero(bincount)[0]}
        
    # Set.
    def set_cluster(self, spikes, cluster):
        self.clusters.ix[spikes] = cluster
        self._update_data()
        
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
    
    # Cluster and group info.
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
    
    # Renumber.
    def renumber(self):
        self.clusters_renumbered, self.cluster_info_renumbered = \
            renumber_clusters(self.clusters, self.cluster_info)
        
        
