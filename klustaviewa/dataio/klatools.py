"""This module provides functions used to load KLA (klailiary) files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables
import time

import numpy as np
import matplotlib.pyplot as plt

from selection import get_indices


# -----------------------------------------------------------------------------
# Probe file functions
# -----------------------------------------------------------------------------
def kla_to_json(kla_dict):
    """Convert a KLA dictionary to JSON.
    cluster_colors and group_colors are pandas.Series objects."""
    kla_full = {}
    for shank, kla in kla_dict.iteritems():
        cluster_colors = kla['cluster_colors']
        group_colors = kla['group_colors']
        clusters = get_indices(cluster_colors)
        groups = get_indices(group_colors)
        kla_shank = dict(
            clusters=[{'cluster': str(cluster), 'color': str(cluster_colors[cluster])}
                for cluster in clusters],
            groups_of_clusters=[{'group': str(group), 'color': str(group_colors[group])}
                for group in groups],
        )
        kla_full[shank] = kla_shank
    return json.dumps(kla_full, indent=4)

def load_kla_json(kla_json):
    """Convert from KLA JSON into two NumPy arrays with the cluster colors and group colors."""
    if not kla_json:
        return None
    kla_dict = json.loads(kla_json)

    shank_all = {}
    
    # load list of cluster and group colors for each shank
    for shank, kla in kla_dict.iteritems():
        cluster_colors = [int(o['color']) for o in kla['clusters']]
        group_colors = [int(o['color']) for o in kla['groups_of_clusters']]
        shank_all[int(shank)] = dict(cluster_colors=cluster_colors, 
            group_colors=group_colors,)
        
    return shank_all
    
def write_kla(filename_kla, kla):
    kla_json = kla_to_json(kla)
    with open(filename_kla, 'w') as f:
        f.write(kla_json)
    

