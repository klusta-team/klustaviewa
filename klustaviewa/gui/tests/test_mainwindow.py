"""Unit tests for the main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time

import numpy as np
import numpy.random as rnd
import pandas as pd
from galry import show_window

from klustaviewa.gui.mainwindow import MainWindow
# from klustaviewa.dataio.tests.mock_data import (setup, teardown, create_similarity_matrix,
        # nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from klustaviewa.dataio.loader import KlustersLoader
from klustaviewa.dataio.selection import select
from klustaviewa.dataio.tools import check_dtype, check_shape
from klustaviewa.gui.mainwindow import MainWindow


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_mainwindow():
    show_window(MainWindow)
    
if __name__ == '__main__':
    test_mainwindow()
    
    