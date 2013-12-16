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
from klustaviewa.views import WaveformView


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_viewdata_1():
    filename = 'dat1s'
    with Experiment(filename, dir='data') as exp:
        data = get_waveformview_data(exp, clusters=[0])
        show_view(WaveformView, **data)
    