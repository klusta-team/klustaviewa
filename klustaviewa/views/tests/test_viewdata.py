"""Unit tests for the viewdata module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import time

import numpy as np

from spikedetekt2.dataio import Experiment
from klustaviewa.views.viewdata import *
from klustaviewa.views.tests.utils import show_view
from klustaviewa.views import WaveformView, FeatureView
from spikedetekt2.dataio.tests.test_experiment import setup, teardown, DIRPATH


# -----------------------------------------------------------------------------
# Mock data
# -----------------------------------------------------------------------------
# TODO: refactor this in proper mock data module in spikedetekt2
def rnd(*shape):
    return np.random.rand(*shape)
    
def rndint(*shape):
    return np.random.randint(size=shape, low=-32000, high=32000)

def add_spikes(exp, nspikes=1000):
    chgrp = exp.channel_groups[0]
    chgrp.spikes.time_samples.append(np.sort(rndint(nspikes)))
    chgrp.spikes.clusters.main.append(np.zeros(nspikes, dtype=np.int32))
    chgrp.spikes.features_masks.append(rnd(nspikes, 3, 2))
    chgrp.spikes.features_masks[..., 1] = chgrp.spikes.features_masks[..., 1] < .5
    chgrp.spikes.waveforms_raw.append(rndint(nspikes, 10, 3))
    chgrp.spikes.waveforms_filtered.append(rndint(nspikes, 10, 3))
    
    
# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_viewdata_waveform_1():
    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        chgrp = exp.channel_groups[0]
        add_spikes(exp)
        data = get_waveformview_data(exp, clusters=[0])
        show_view(WaveformView, **data)
    