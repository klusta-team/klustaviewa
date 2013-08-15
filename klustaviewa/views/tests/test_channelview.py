"""Unit tests for channel view."""

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
    nspikes, nchannels, nsamples, nchannels, fetdim, ncorrbins)
from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.selection import select, get_indices
from kwiklib.dataio.tools import check_dtype, check_shape
from klustaviewa import USERPREF
from klustaviewa.views import ChannelView
from klustaviewa.views.tests.utils import show_view, get_data, assert_fun


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_channelview():
    # keys = ('channel_names','channel_colors','channel_groups','channel_group_colors','channel_group_names')
    # data = get_data()
    # kwargs = {k: data[k] for k in keys}
    
    kwargs['channel_colors'] = data['channel_colors_full']
    # kwargs['background'] = {5: 1, 7: 2}
    
    kwargs['operators'] = [
        lambda self: self.view.set_quality(quality),
        lambda self: self.view.set_background({5: 1, 7: 2}),
        lambda self: self.view.set_background({6: 3}),
        lambda self: self.view.set_background({}),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    window = show_view(ChannelView, **kwargs)
    
def create_channel_names(nchannels):
    return ["Channel {0:d}".format(channel) for channel in xrange(nchannels)]

def create_channel_colors(nchannels):
    return np.mod(np.arange(nchannels, dtype=np.int32), COLORS_COUNT) + 1

def create_channel_group_names(nchannelgroups):
    return ["Group {0:d}".format(channelgroup) for channelgroup in xrange(nchannelgroups)]

def create_channel_groups(nchannels):
    return np.array(np.random.randint(size=nchannels, low=0, high=4), 
        dtype=np.int32)

def create_channel_group_colors(nchannelgroups):
    return np.mod(np.arange(nchannelgroups, dtype=np.int32), COLORS_COUNT) + 1
    