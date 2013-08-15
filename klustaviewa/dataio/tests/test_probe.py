"""Unit tests for probe module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import pprint

import numpy as np
import numpy.random as rnd
import pandas as pd

from kwiklib.dataio import probe_to_json, load_probe_json

# -----------------------------------------------------------------------------
# Probe tests
# -----------------------------------------------------------------------------
def test_probe_1():
    graph = {1: [(0, 1), (0, 2), (1, 2), (1, 3)],
                           2: [(7, 4), (7, 5), (4, 5)]}
    probe_ns = {'probes': graph}
    probe_json = probe_to_json(probe_ns)
    probe = json.loads(probe_json)
    
    assert probe['nchannels'] == 7
    assert probe['dead_channels'] == []
    
    assert np.array_equal(probe['shanks'][0]['graph'], graph[1])
    assert np.array_equal(probe['shanks'][1]['graph'], graph[2])

def test_probe_2():
    graph = {1: [(0, 1), (0, 2), (1, 2), (1, 3)],
                           2: [(7, 4), (7, 5), (4, 5)]}
    probe_ns = {'probes': graph}
    probe_json = probe_to_json(probe_ns)
    probe = load_probe_json(probe_json)
    
    assert np.array_equal(probe[1]['graph'], graph[1])
    
