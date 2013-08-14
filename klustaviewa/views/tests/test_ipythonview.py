"""Unit tests for IPython view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time

import numpy as np
import numpy.random as rnd
import pandas as pd

import galry

from klustaviewa.views.ipythonview import IPythonView, IPYTHON
from klustaviewa import USERPREF
from klustaviewa.views.tests.utils import show_view, assert_fun


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_ipythonview():

    if not IPYTHON:
        return
    
    kwargs = {}
    
    kwargs['operators'] = [
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    window = show_view(IPythonView, **kwargs)
    
    
    