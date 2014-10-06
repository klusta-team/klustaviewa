"""Experiment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import tables as tb
from nose import with_setup

from klustaviewa.gui.recluster import run_klustakwik
from kwiklib.dataio.kwik import (add_recording, create_files, open_files,
    close_files, add_event_type, add_cluster_group, get_filenames,
    add_cluster, add_spikes)
from kwiklib.dataio.experiment import (Experiment, _resolve_hdf5_path,
    ArrayProxy, DictVectorizer)
from kwiklib.utils.six import itervalues
from kwiklib.utils.logger import info


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()

def _setup(_name, has_masks=True):
    # Create files.
    prm = {'nfeatures': 3*3, 
           'waveforms_nsamples': 10, 
           'nchannels': 3,
           'sample_rate': 20000.,
           'nfeatures_per_channel': 3,
           'has_masks': has_masks
           }
    prb = {0:
        {
            'channels': [0, 1, 2],
            'graph': [[0, 1], [0, 2]],
            'geometry': {0: [0.4, 0.6], 1: [0.6, 0.8], 2: [0.8, 0.0]},
        }
    }
    create_files(_name, dir=DIRPATH, prm=prm, prb=prb)
    
    # Open the files.
    files = open_files(_name, dir=DIRPATH, mode='a')
    
    # Add data.
    add_recording(files, 
                  sample_rate=20000.,
                  bit_depth=16,
                  band_high=100.,
                  band_low=500.,
                  nchannels=3,)
    add_event_type(files, 'myevents')
    add_cluster_group(files, channel_group_id='0', id='0', name='Noise')
    add_cluster_group(files, channel_group_id='0', id='1', name='MUA')
    add_cluster_group(files, channel_group_id='0', id='2', name='Good')
    add_cluster_group(files, channel_group_id='0', id='3', name='Unsorted')
    add_cluster(files, channel_group_id='0', cluster_group=0)

    add_spikes(files, channel_group_id='0',
        cluster=np.random.randint(5, 10, 1000),
        time_samples=np.cumsum(np.random.randint(0, 1000, 1000)).astype(np.int64),
        features=np.random.randint(-30000, 30000, (1000,9)).astype(np.int16),
        masks=np.random.randint(0, 2, (1000,9)).astype(np.int16),
        )
    
    # Close the files
    close_files(files)

def _teardown(_name):
    files = get_filenames(_name, dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]

def setup(): _setup('myexperiment')
def teardown(): _teardown('myexperiment')
    
def test_recluster():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        spikes, clu = run_klustakwik(exp, channel_group=0, clusters=[5, 8])
        assert 10 < len(clu) < 990

