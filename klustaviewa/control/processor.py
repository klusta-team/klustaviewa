"""The Controller offers high-level methods to change the data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import inspect
from collections import Counter

import numpy as np
import pandas as pd

import klustaviewa.utils.logger as log
from klustaviewa.io.selection import get_indices, select
from klustaviewa.io.tools import get_array
from klustaviewa.utils.colors import next_color


# -----------------------------------------------------------------------------
# Processor
# -----------------------------------------------------------------------------
class Processor(object):
    """Implement actions.
    
    An Action object is:
        
        (method_name, args, kwargs)
    
    """
    def __init__(self, loader):
        self.loader = loader
    
    
    # Actions.
    # --------
    # Merge.
    def merge_clusters(self, clusters_old, cluster_groups, cluster_colors,
        cluster_merged):
        # Get spikes in clusters to merge.
        # spikes = self.loader.get_spikes(clusters=clusters_to_merge)
        spikes = get_indices(clusters_old)
        clusters_to_merge = get_indices(cluster_groups)
        group = get_array(cluster_groups)[0]
        color_old = get_array(cluster_groups)[0]
        color_new = next_color(color_old)
        self.loader.add_cluster(cluster_merged, group, color_new)
        # Set the new cluster to the corresponding spikes.
        self.loader.set_cluster(spikes, cluster_merged)
        # Remove old clusters.
        for cluster in clusters_to_merge:
            self.loader.remove_cluster(cluster)
        self.loader.unselect()
        return dict(to_select=cluster_merged,
            to_invalidate=sorted(set(clusters_to_merge).union(
                set([cluster_merged]))),
            to_compute=cluster_merged)
        
    def merge_clusters_undo(self, clusters_old, cluster_groups, 
        cluster_colors, cluster_merged):
        # Get spikes in clusters to merge.
        spikes = self.loader.get_spikes(clusters=cluster_merged)
        clusters_to_merge = get_indices(cluster_groups)
        # Add old clusters.
        for cluster, group, color in zip(
                clusters_to_merge, cluster_groups, cluster_colors):
            self.loader.add_cluster(cluster, group, color)
        # Set the new clusters to the corresponding spikes.
        self.loader.set_cluster(spikes, clusters_old)
        # Remove merged cluster.
        self.loader.remove_cluster(cluster_merged)
        self.loader.unselect()
        return dict(to_select=clusters_to_merge,
            to_invalidate=sorted(set(clusters_to_merge).union(
                set([cluster_merged]))),
            to_compute=clusters_to_merge)
        
        
    # Split.
    def split_clusters(self, clusters_old, cluster_groups, 
        cluster_colors, clusters_new):
        spikes = get_indices(clusters_old)
        # Find groups and colors of old clusters.
        cluster_indices_old = np.unique(clusters_old)
        cluster_indices_new = np.unique(clusters_new)
        # Get group and color of the new clusters, from the old clusters.
        groups = self.loader.get_cluster_groups(cluster_indices_old)
        colors = self.loader.get_cluster_colors(cluster_indices_old)
        # Add clusters.
        for cluster_new, group, color in zip(cluster_indices_new, 
                groups, colors):
            self.loader.add_cluster(cluster_new, group, next_color(color))
        # Set the new clusters to the corresponding spikes.
        self.loader.set_cluster(spikes, clusters_new)
        # Remove empty clusters.
        clusters_empty = self.loader.remove_empty_clusters()
        self.loader.unselect()
        clusters_to_select = sorted(set(cluster_indices_old).union(
                set(cluster_indices_new)) - set(clusters_empty))
        return dict(
            to_select=clusters_to_select,
            to_invalidate=sorted(set(cluster_indices_old).union(
                set(cluster_indices_new))),
            to_compute=clusters_to_select)
        
    def split_clusters_undo(self, clusters_old, cluster_groups, 
        cluster_colors, clusters_new):
        spikes = get_indices(clusters_old)
        # Find groups and colors of old clusters.
        cluster_indices_old = np.unique(clusters_old)
        cluster_indices_new = np.unique(clusters_new)
        # Add clusters that were removed after the split operation.
        clusters_empty = sorted(set(cluster_indices_old) - 
            set(cluster_indices_new))
        for cluster in clusters_empty:
            self.loader.add_cluster(cluster, select(cluster_groups, cluster),
                select(cluster_colors, cluster))
        # Set the new clusters to the corresponding spikes.
        self.loader.set_cluster(spikes, clusters_old)
        # Remove clusters.
        # Remove empty clusters.
        clusters_empty = self.loader.remove_empty_clusters()
        self.loader.unselect()
        clusters_to_select = cluster_indices_old
        return dict(
            to_select=clusters_to_select,
            to_invalidate=sorted(set(cluster_indices_old).union(
                set(cluster_indices_new))),
            to_compute=clusters_to_select)
        
        
    # Change cluster color.
    def change_cluster_color(self, cluster, color_old, color_new,
            clusters_selected):
        self.loader.set_cluster_colors(cluster, color_new)
        return dict(to_select=clusters_selected)
        
    def change_cluster_color_undo(self, cluster, color_old, color_new,
            clusters_selected):
        self.loader.set_cluster_colors(cluster, color_old)
        return dict(to_select=clusters_selected)
        
        
    # Move clusters.
    def move_clusters(self, clusters, groups_old, group_new):
        # Get next cluster to select.
        next_cluster = self.loader.get_next_cluster(clusters[-1])
        self.loader.set_cluster_groups(clusters, group_new)
        # to_compute=[] to force refreshing the correlation matrix
        return dict(to_select=[next_cluster], to_compute=[])
        
    def move_clusters_undo(self, clusters, groups_old, group_new):
        self.loader.set_cluster_groups(clusters, groups_old)
        # to_compute=[] to force refreshing the correlation matrix
        return dict(to_select=clusters, to_compute=[])
      
      
    # Rename group.
    def rename_group(self, group, name_old, name_new):
        self.loader.set_group_names(group, name_new)
        
    def rename_group_undo(self, group, name_old, name_new):
        self.loader.set_group_names(group, name_old)
    
    
    # Change group color.
    def change_group_color(self, group, color_old, color_new):
        self.loader.set_group_colors(group, color_new)
        return dict(groups_to_select=[group])
        
    def change_group_color_undo(self, group, color_old, color_new):
        self.loader.set_group_colors(group, color_old)
        return dict(groups_to_select=[group])
    
    
    # Add group.
    def add_group(self, group, name, color):
        self.loader.add_group(group, name, color)
        
    def add_group_undo(self, group, name, color):
        self.loader.remove_group(group)
    
    
    # Remove group.
    def remove_group(self, group, name, color):
        self.loader.remove_group(group)
        
    def remove_group_undo(self, group, name, color):
        self.loader.add_group(group, name, color)
    
    