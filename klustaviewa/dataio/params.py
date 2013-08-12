"""This module provides functions used to load params files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables
import time

import numpy as np
import matplotlib.pyplot as plt

from klustersloader import (find_filenames, find_index, read_xml,
    filename_to_triplet, triplet_to_filename, find_indices,
    find_hdf5_filenames,
    read_clusters, read_cluster_info, read_group_info,)
from tools import MemMappedText, MemMappedBinary


# -----------------------------------------------------------------------------
# Probe file functions
# -----------------------------------------------------------------------------
def params_to_json(metadata_xml):
    """Convert PARAMS from XML to JSON."""
    params = dict(
        SAMPLING_FREQUENCY=metadata_xml['freq'],
        FETDIM=metadata_xml['fetdim'],
        WAVEFORMS_NSAMPLES=metadata_xml['nsamples'],
    )
    return json.dumps(params)

def load_params_json(params_json):
    if not params_json:
        return None
    params_dict = json.loads(params_json)
    
    params = {}
    
    # Get the sampling frequency from the PARAMS file.
    params['freq'] = f = float(params_dict['SAMPLING_FREQUENCY'])
    params['nsamples'] = params_dict['WAVEFORMS_NSAMPLES']
    params['fetdim'] = params_dict['FETDIM']
    
    # if not 'nsamples' in params:
        # # Get the number of samples per waveform from the PARAMS file.
        # params['nsamples'] = int(f * float(params_dict['T_BEFORE']) + float(params_dict['T_AFTER']))
    
    return params
    

    

