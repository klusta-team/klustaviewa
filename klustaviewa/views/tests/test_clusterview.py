"""Unit tests for cluster view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.views.tests.mock_data import (setup, teardown,
    create_similarity_matrix,
    nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from klustaviewa.dataio.loader import KlustersLoader
from klustaviewa.dataio.selection import select, get_indices
from klustaviewa.dataio.tools import check_dtype, check_shape
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.views import ClusterView
from klustaviewa.views.tests.utils import show_view, get_data, assert_fun


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_clusterview():
    keys = ('cluster_groups,group_colors,group_names,'
            'cluster_sizes').split(',')
    data = get_data()
    kwargs = {k: data[k] for k in keys}
    
    kwargs['cluster_colors'] = data['cluster_colors_full']
    
    clusters = get_indices(data['cluster_sizes'])
    quality = pd.Series(np.random.rand(len(clusters)), index=clusters)
    
    kwargs['cluster_quality'] = quality
    
    kwargs['operators'] = [
        # lambda self: self.view.select([2,4]),
        # lambda self: self.view.add_group("MyGroup", [2,3,6]),
        # lambda self: self.view.rename_group(3, "New group"),
        # lambda self: self.view.change_group_color(3, 2),
        # lambda self: self.view.change_cluster_color(3, 4),
        # lambda self: self.view.move_to_noise(3),
        # lambda self: self.view.unselect(),
        lambda self: self.view.set_quality(quality),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    window = show_view(ClusterView, **kwargs)
    
    
    