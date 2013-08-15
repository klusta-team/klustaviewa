"""Unit tests for correlation matrix view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from klustaviewa.views.tests.mock_data import (setup, teardown, create_similarity_matrix,
        nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.selection import select
from kwiklib.dataio.tools import check_dtype, check_shape
from klustaviewa import USERPREF
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
        lambda self: self.view.show_selection(5, 6),
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(SimilarityMatrixView, **kwargs)
    
    