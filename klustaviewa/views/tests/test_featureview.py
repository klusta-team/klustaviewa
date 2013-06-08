"""Unit tests for feature view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.views.tests.mock_data import (setup, teardown,
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from klustaviewa.dataio.loader import KlustersLoader
from klustaviewa.dataio.selection import select
from klustaviewa.dataio.tools import check_dtype, check_shape
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.views import FeatureView
from klustaviewa.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_featureview():
        
    keys = ('features,masks,clusters,clusters_selected,cluster_colors,'
            'fetdim,nchannels,nextrafet,duration,freq').split(',')
           
    data = get_data()
    kwargs = {k: data[k] for k in keys}
    
    kwargs['features'] = data['features_full']
    
    kwargs['operators'] = [
        lambda self: self.view.toggle_mask(),
        lambda self: self.view.set_wizard_target(2, 1),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(FeatureView, **kwargs)
    
    