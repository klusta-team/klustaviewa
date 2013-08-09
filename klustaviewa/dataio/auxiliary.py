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
def kla_to_json(kla):
    """cluster_colors and group_colors are pandas.Series objects."""
    cluster_colors = kla['cluster_colors']
    group_colors = kla['group_colors']
    clusters = get_indices(cluster_colors)
    groups = get_indices(group_colors)
    kla = dict(
        clusters=[{'cluster': str(cluster), 'color': str(cluster_colors.ix[cluster])}
            for cluster in clusters],
        groups_of_clusters=[{'group': str(group), 'color': str(group_colors.ix[group])}
            for group in groups],
    )
    return json.dumps(kla)

def load_kla_json(kla_json):
    """Convert from KLA JSON into two NumPy arrays with the cluster colors and group colors."""
    if not kla_json:
        return None
    kla_dict = json.loads(kla_json)
    
    cluster_colors = [o['color'] for o in kla_dict['clusters']]
    group_colors = [o['color'] for o in kla_dict['groups_of_clusters']]
    
    return dict(cluster_colors=cluster_colors, 
        group_colors=group_colors,)
    

    

