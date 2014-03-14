"""Unit tests for the main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time

import numpy as np
import numpy.random as rnd
import pandas as pd
from qtools import show_window

from klustaviewa.gui.mainwindow import MainWindow
from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.selection import select
from kwiklib.dataio.tools import check_dtype, check_shape
from klustaviewa.gui.mainwindow import MainWindow


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def DISABLEDtest_mainwindow():
    show_window(MainWindow, dolog=False)
    
if __name__ == '__main__':
    test_mainwindow()
    
    