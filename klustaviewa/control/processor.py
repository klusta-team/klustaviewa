"""The Controller offers high-level methods to change the data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import inspect

import numpy as np
import pandas as pd

from kwiklib.utils import logger as log
from kwiklib.dataio.selection import get_indices, select
from kwiklib.dataio.tools import get_array
from kwiklib.utils.colors import random_color


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
        group = np.max(get_array(cluster_groups))
        # color_old = get_array(cluster_colors)[0]
        color_new = random_color()
        self.loader.add_cluster(cluster_merged, group, color_new)
        # Set the new cluster to the corresponding spikes.
        self.loader.set_cluster(spikes, cluster_merged)
        # Remove old clusters.
        for cluster in clusters_to_merge:
            self.loader.remove_cluster(cluster)
        self.loader.unselect()
        return dict(clusters_to_merge=clusters_to_merge,
                    cluster_merged=cluster_merged,
                    cluster_merged_colors=(color_new, color_new),)

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
        color_old = self.loader.get_cluster_color(clusters_to_merge[0])
        color_old2 = self.loader.get_cluster_color(clusters_to_merge[1])
        return dict(clusters_to_merge=clusters_to_merge,
                    cluster_merged=cluster_merged,
                    cluster_to_merge_colors=(color_old, color_old2),
                    )


    # Split.
    def split_clusters(self, clusters, clusters_old, cluster_groups,
        cluster_colors, clusters_new):
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        spikes = get_indices(clusters_old)
        # Find groups and colors of old clusters.
        cluster_indices_old = np.unique(clusters_old)
        cluster_indices_new = np.unique(clusters_new)
        # Get group and color of the new clusters, from the old clusters.
        groups = self.loader.get_cluster_groups(cluster_indices_old)
        # colors = self.loader.get_cluster_colors(cluster_indices_old)
        # Add clusters.
        self.loader.add_clusters(cluster_indices_new,
            # HACK: take the group of the first cluster for all new clusters
            get_array(groups)[0]*np.ones(len(cluster_indices_new)),
            )
        # Set the new clusters to the corresponding spikes.
        self.loader.set_cluster(spikes, clusters_new)
        # Remove empty clusters.
        clusters_empty = self.loader.remove_empty_clusters()
        self.loader.unselect()
        clusters_to_select = sorted(set(cluster_indices_old).union(
                set(cluster_indices_new)) - set(clusters_empty))
        return dict(clusters_to_split=clusters,
                    clusters_split=get_array(cluster_indices_new),
                    clusters_empty=clusters_empty)

    def split_clusters_undo(self, clusters, clusters_old, cluster_groups,
        cluster_colors, clusters_new):
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        spikes = get_indices(clusters_old)
        # Find groups and colors of old clusters.
        cluster_indices_old = np.unique(clusters_old)
        cluster_indices_new = np.unique(clusters_new)
        # Add clusters that were removed after the split operation.
        clusters_empty = sorted(set(cluster_indices_old) -
            set(cluster_indices_new))
        self.loader.add_clusters(
            clusters_empty,
            select(cluster_groups, clusters_empty),
            # select(cluster_colors, clusters_empty),
            )
        # Set the new clusters to the corresponding spikes.
        self.loader.set_cluster(spikes, clusters_old)
        # Remove empty clusters.
        clusters_empty = self.loader.remove_empty_clusters()
        self.loader.unselect()
        return dict(clusters_to_split=clusters,
                    clusters_split=get_array(cluster_indices_new),
                    # clusters_empty=clusters_empty
                    )


    # Change cluster color.
    def change_cluster_color(self, cluster, color_old, color_new,
            clusters_selected):
        self.loader.set_cluster_colors(cluster, color_new)
        return dict(clusters=clusters_selected, cluster=cluster,
            color_old=color_old, color_new=color_new)

    def change_cluster_color_undo(self, cluster, color_old, color_new,
            clusters_selected):
        self.loader.set_cluster_colors(cluster, color_old)
        return dict(clusters=clusters_selected, cluster=cluster,
            color_old=color_old, color_new=color_new)


    # Move clusters.
    def move_clusters(self, clusters, groups_old, group_new):
        # Get next cluster to select.
        next_cluster = self.loader.get_next_cluster(clusters[-1])
        self.loader.set_cluster_groups(clusters, group_new)
        # to_compute=[] to force refreshing the correlation matrix
        # return dict(to_select=[next_cluster], to_compute=[])
        return dict(clusters=clusters, groups_old=groups_old, group=group_new,
            next_cluster=next_cluster)

    def move_clusters_undo(self, clusters, groups_old, group_new):
        self.loader.set_cluster_groups(clusters, groups_old)
        # to_compute=[] to force refreshing the correlation matrix
        # return dict(to_select=clusters, to_compute=[])
        return dict(clusters=clusters, groups_old=groups_old, group=group_new)


    # Change group color.
    def change_group_color(self, group, color_old, color_new):
        self.loader.set_group_colors(group, color_new)
        return dict(groups=[group])

    def change_group_color_undo(self, group, color_old, color_new):
        self.loader.set_group_colors(group, color_old)
        return dict(groups=[group])


    # Add group.
    def add_group(self, group, name, color):
        self.loader.add_group(group, name, color)

    def add_group_undo(self, group, name, color):
        self.loader.remove_group(group)


    # Rename group.
    def rename_group(self, group, name_old, name_new):
        self.loader.set_group_names(group, name_new)

    def rename_group_undo(self, group, name_old, name_new):
        self.loader.set_group_names(group, name_old)


    # Remove group.
    def remove_group(self, group, name, color):
        self.loader.remove_group(group)

    def remove_group_undo(self, group, name, color):
        self.loader.add_group(group, name, color)


