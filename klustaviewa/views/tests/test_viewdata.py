"""Unit tests for the viewdata module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import time
import tempfile

import numpy as np

from klustaviewa.views.tests.mock_data import (ncorrbins, corrbin,
        create_baselines, create_correlograms)
from spikedetekt2.dataio import *
from klustaviewa.views.viewdata import *
from klustaviewa.views.tests.utils import show_view
from klustaviewa.views import (WaveformView, FeatureView, ClusterView,
    CorrelogramsView)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
# TODO: refactor this in proper mock data module in spikedetekt2
DIRPATH = tempfile.mkdtemp()

def rnd(*shape):
    return .25 * np.random.randn(*shape)
    
def rndint(*shape):
    return np.random.randint(size=shape, low=-32000, high=32000)

def setup():
    # Create files.
    prm = {'nfeatures': 3, 'waveforms_nsamples': 10, 'nchannels': 3,
           'nfeatures_per_channel': 1,
           'sampling_frequency': 20000.,
           'duration': 10.}
    prb = {'channel_groups': [
        {
            'channels': [4, 6, 8],
            'graph': [[4, 6], [8, 4]],
            'geometry': {4: [0.4, 0.6], 6: [0.6, 0.8], 8: [0.8, 0.0]},
        }
    ]}
    create_files('myexperiment', dir=DIRPATH, prm=prm, prb=prb)
    
    # Open the files.
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    
    # Add data.
    add_recording(files, 
                  sample_rate=20000.,
                  start_time=10., 
                  start_sample=200000.,
                  bit_depth=16,
                  band_high=100.,
                  band_low=500.,
                  nchannels=3,)
    add_event_type(files, 'myevents')
    add_cluster_group(files, channel_group_id='0', id='noise', name='Noise')
    add_cluster(files, channel_group_id='0', cluster_group=0)
    
    exp = Experiment(files=files)
    chgrp = exp.channel_groups[0]
    nspikes = 1000
    chgrp.spikes.time_samples.append(np.sort(rndint(nspikes)))
    chgrp.spikes.clusters.main.append(np.random.randint(size=nspikes, low=0, high=2))
    chgrp.spikes.features_masks.append(rnd(nspikes, 3, 2))
    chgrp.spikes.features_masks[..., 1] = chgrp.spikes.features_masks[..., 1] < .5
    chgrp.spikes.waveforms_raw.append(rndint(nspikes, 10, 3))
    chgrp.spikes.waveforms_filtered.append(rndint(nspikes, 10, 3))
    
    # Close the files
    close_files(files)

def teardown():
    files = get_filenames('myexperiment', dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_viewdata_waveform_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        data = get_waveformview_data(exp, clusters=[0])
        show_view(WaveformView, **data)
    
def test_viewdata_featureview_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        data = get_featureview_data(exp, clusters=[0])
        show_view(FeatureView, **data)
    
def test_viewdata_clusterview_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        data = get_clusterview_data(exp)
        show_view(ClusterView, **data)
    
def test_viewdata_correlogramsview_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        correlograms = create_correlograms([0], 50)
        data = get_correlogramsview_data(exp, clusters=[0], 
                                         correlograms=correlograms,
                                         )
        show_view(CorrelogramsView, **data)
    
    
    