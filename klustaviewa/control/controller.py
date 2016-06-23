"""The Controller offers high-level methods to change the data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import inspect

import numpy as np
import pandas as pd

from klustaviewa.control.processor import Processor
from klustaviewa.control.stack import Stack
from kwiklib.utils import logger as log
from kwiklib.dataio.selection import get_indices, select
from kwiklib.dataio.tools import get_array


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_pretty_arg(item):
    if isinstance(item, (pd.Series)):
        if item.size == 0:
            return '[]'
        elif item.size == 1:
            return '[{0:s}]'.format(str(item.values[0]))
        else:
            return '[{0:s}, ..., {1:s}]'.format(*map(str, item.values[[0, -1]]))
    if isinstance(item, (pd.Int64Index, pd.Index)):
        if item.size == 0:
            return '[]'
        elif item.size == 1:
            return '[{0:s}]'.format(str(item.values[0]))
        else:
            return '[{0:s}, ..., {1:s}]'.format(*map(str, item.values[[0, -1]]))
    return str(item).replace('\n', '')

def get_pretty_action(method_name, args, kwargs, verb='Process'):
    args_str = ', '.join(map(get_pretty_arg, args))
    kwargs_str = ', '.join([key + '=' + str(val)
        for key, val in kwargs.iteritems()])
    if kwargs_str:
        kwargs_str = ', ' + kwargs_str
    return '{3:s} action {0:s}({1:s}{2:s})'.format(
        method_name, args_str, kwargs_str, verb)

def log_action(action, prefix=''):
    method_name, args, kwargs = action
    description = kwargs.get('_description', 
        get_pretty_action(*action))
    log.info(prefix + description)

def call_action(processor, action, suffix=''):
    method_name, args, kwargs = action
    kwargs = kwargs.copy()
    kwargs.pop('_description', None)
    return getattr(processor, method_name + suffix)(*args, **kwargs)


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------
class Controller(object):
    """Implement actions that can be undone and redone.
    
    An Action object is:
        
        (method_name, args, kwargs)
    
    """
    def __init__(self, loader):
        self.loader = loader
        self.processor = Processor(loader)
        # Create the action stack.
        self.stack = Stack(maxsize=20)
    
    
    # Internal action methods.
    # ------------------------
    
    def _process(self, method_name, *args, **kwargs):
        """Create, register, and process an action."""
        # Create the action.
        action = (method_name, args, kwargs)
        # Add the action to the stack.
        self.stack.add(action)
        # Log the action.
        log_action(action)
        # Process the action.
        output = call_action(self.processor, action)
        return method_name, output or {}
    
    
    # Public action methods.
    # ----------------------
    def merge_clusters(self, clusters):
        clusters_to_merge = clusters
        cluster_merged = self.loader.get_new_clusters(1)[0]
        clusters_old = self.loader.get_clusters(clusters=clusters_to_merge)
        cluster_groups = self.loader.get_cluster_groups(clusters_to_merge)
        cluster_colors = self.loader.get_cluster_colors(clusters_to_merge)
        return self._process('merge_clusters', clusters_old, cluster_groups, 
            cluster_colors, cluster_merged, 
            _description='Merged clusters {0:s} into {1:s}'.format(
                get_pretty_arg(list(clusters)), 
                get_pretty_arg(cluster_merged)))
        
    def split_clusters(self, clusters, spikes):
        # Old clusters for all spikes to split.
        clusters_old = self.loader.get_clusters(spikes=spikes)
        assert np.all(np.in1d(clusters_old, clusters))
        # Old cluster indices.
        cluster_indices_old = np.unique(clusters_old)
        nclusters = len(cluster_indices_old)
        # New clusters indices.
        clusters_indices_new = self.loader.get_new_clusters(nclusters)
        # Generate new clusters array.
        clusters_new = clusters_old.copy()
        # Assign new clusters.
        for cluster_old, cluster_new in zip(cluster_indices_old,
                clusters_indices_new):
            clusters_new[clusters_old == cluster_old] = cluster_new
        cluster_groups = self.loader.get_cluster_groups(cluster_indices_old)
        cluster_colors = self.loader.get_cluster_colors(cluster_indices_old)
        return self._process('split_clusters', clusters, 
            clusters_old, cluster_groups, cluster_colors, clusters_new, 
            _description='Split clusters {0:s} into {1:s}'.format(
                get_pretty_arg(list(cluster_indices_old)),
                get_pretty_arg(list(clusters_indices_new)),
                ))

    def split2_clusters(self, spikes, clusters):
        # clusters is new
        # Old clusters for all spikes to split.
        clusters_old = self.loader.get_clusters(spikes=spikes)
        # assert np.all(np.in1d(clusters_old, clusters))
        # Old cluster indices.
        cluster_indices_old = np.unique(clusters_old)
        nclusters = len(cluster_indices_old)

        
        # Renumber output of klustakwik.
        clu_idx = np.unique(clusters)
        nclusters_new = len(clu_idx)
        # Get new clusters indices.
        clusters_indices_new = self.loader.get_new_clusters(nclusters_new)
        clu_renumber = np.zeros(clu_idx.max() + 1, dtype=np.int32)
        clu_renumber[clu_idx] = clusters_indices_new
        clusters_new = clu_renumber[clusters]

        cluster_groups = self.loader.get_cluster_groups(cluster_indices_old)
        cluster_colors = self.loader.get_cluster_colors(cluster_indices_old)
        return self._process('split_clusters', get_array(cluster_indices_old), 
            clusters_old, cluster_groups, cluster_colors, clusters_new, 
            _description='Split2')
        
    def change_cluster_color(self, cluster, color):
        color_old = self.loader.get_cluster_color(cluster) # Sven edit self.loader.get_cluster_colors(cluster) removed s
        color_new = color
        clusters_selected = self.loader.get_clusters_selected()
        return self._process('change_cluster_color', cluster, color_old, 
            color_new, clusters_selected, 
            _description='Changed cluster color of {0:s}'.format(get_pretty_arg(cluster)))
        
    def move_clusters(self, clusters, group):
        groups_old = self.loader.get_cluster_groups(clusters)
        group_new = group
        return self._process('move_clusters', clusters, groups_old, group_new, 
            _description='Moved clusters {0:s} to {1:s}'.format(
                get_pretty_arg(clusters), get_pretty_arg(group)))
      
    def rename_group(self, group, name):
        name_old = self.loader.get_group_names(group)
        name_new = name
        return self._process('rename_group', group, name_old, name_new, 
            _description='Renamed group {0:s} to {1:s}'.format(
                get_pretty_arg(group), get_pretty_arg(name)))
        
    def change_group_color(self, group, color):
        color_old = self.loader.get_group_colors(group)
        color_new = color
        return self._process('change_group_color', group, color_old, color_new, 
            _description='Changed color of group {0:s}'.format(get_pretty_arg(group)))
    
    def add_group(self, group, name, color):
        return self._process('add_group', group, name, color, 
            _description='Added group {0:s}'.format(get_pretty_arg(name)))
        
    def remove_group(self, group):
        name = self.loader.get_group_names(group)
        color = None # Sven changed because group color is giving errors
        return self._process('remove_group', group, name, color, 
            _description='Removed group {0:s}'.format(get_pretty_arg(group)))
        
    
    
    # Stack methods.
    # --------------
    def undo(self):
        """Undo an action if possible."""
        action = self.stack.undo()
        if action is None:
            return None, None
        # Get the undone action.
        method_name, args, kwargs = action
        # Log the action.
        log_action(action, prefix='Undo: ')
        # Undo the action.
        output = call_action(self.processor, action, suffix='_undo')
        return method_name + '_undo', output or {}
        
    def redo(self):
        action = self.stack.redo()
        if action is None:
            return
        # Get the redo action.
        method_name, args, kwargs = action
        # Log the action.
        log_action(action, prefix='Redo: ')
        # Redo the action.
        output = call_action(self.processor, action)
            
        return method_name, output or {}
        
    def can_undo(self):
        return self.stack.can_undo()
        
    def can_redo(self):
        return self.stack.can_redo()

