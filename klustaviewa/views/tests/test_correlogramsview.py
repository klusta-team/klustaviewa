"""Unit tests for correlograms view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.views.tests.mock_data import (setup, teardown,
        nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins, corrbin,
        create_baselines, create_correlograms)
from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.selection import select
from kwiklib.dataio.tools import check_dtype, check_shape
from klustaviewa import USERPREF
from klustaviewa.views import CorrelogramsView
from klustaviewa.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_correlogramsview():
    keys = ('clusters_selected,cluster_colors').split(',')
           
    data = get_data()
    kwargs = {k: data[k] for k in keys}
    
    kwargs['correlograms'] = create_correlograms(kwargs['clusters_selected'], 
        ncorrbins)
    kwargs['baselines'] = create_baselines(kwargs['clusters_selected'])
    kwargs['ncorrbins'] = ncorrbins
    kwargs['corrbin'] = corrbin
    
    kwargs['operators'] = [
        lambda self: self.view.change_normalization('uniform'),
        lambda self: self.view.change_normalization('row'),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(CorrelogramsView, **kwargs)
    
    