"""Unit tests for correlation matrix view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.io.tests.mock_data import (setup, teardown, create_similarity_matrix,
        nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from klustaviewa.io.loader import KlustersLoader
from klustaviewa.io.selection import select
from klustaviewa.io.tools import check_dtype, check_shape
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.views import SimilarityMatrixView
from klustaviewa.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_similaritymatrixview():
    data = get_data()
    
    kwargs = {}
    kwargs['similarity_matrix'] = create_similarity_matrix(nclusters)
    kwargs['cluster_colors_full'] = data['cluster_colors_full']
    
    kwargs['operators'] = [
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(SimilarityMatrixView, **kwargs)
    
    