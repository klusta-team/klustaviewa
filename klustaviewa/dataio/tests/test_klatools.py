"""Unit tests for klatools module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import pprint

import numpy as np
import numpy.random as rnd
import pandas as pd

from kwiklib.dataio import kla_to_json, load_kla_json

# -----------------------------------------------------------------------------
# KLA tests
# -----------------------------------------------------------------------------
def test_kla_1():
    kla = {1: {'cluster_colors': [1, 2, 5],
               'group_colors': [4],},
           2: {'cluster_colors': [6],
               'group_colors': [1, 3],}
               }
    kla_json = kla_to_json(kla)
    kla2 = load_kla_json(kla_json)
    
    for shank in (1, 2):
        for what in ('cluster_colors', 'cluster_colors'):
            assert kla2[shank][what] == kla[shank][what]
    
