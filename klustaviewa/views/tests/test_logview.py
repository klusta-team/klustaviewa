"""Unit tests for log view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import time

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.views.logview import LogView
from klustaviewa import USERPREF
from klustaviewa.views.tests.utils import show_view, assert_fun


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_logview():

    kwargs = {}
    
    kwargs['operators'] = [
        lambda self: sys.stdout.write("Hello world!"),
        lambda self: assert_fun(self.view.get_text() == "Hello world!"),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    window = show_view(LogView, **kwargs)
    
    
    