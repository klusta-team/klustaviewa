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
    shanks = metadata_xml['shanks']
    params = dict(
        SAMPLING_FREQUENCY=metadata_xml['freq'],
        FETDIM={shank: metadata_xml[shank]['fetdim'] 
            for shank in shanks},
        WAVEFORMS_NSAMPLES={shank: metadata_xml[shank]['nsamples'] 
            for shank in shanks},
    )
    return json.dumps(params, indent=4)

def load_params_json(params_json):
    if not params_json:
        return None
    params_dict = json.loads(params_json)
    
    params = {}
    
    # Get the sampling frequency from the PARAMS file.
    params['freq'] = f = float(params_dict['SAMPLING_FREQUENCY'])
    
    # Number of samples per waveform.
    if isinstance(params_dict['WAVEFORMS_NSAMPLES'], dict):
        params['nsamples'] = {int(key): value 
            for key, value in params_dict['WAVEFORMS_NSAMPLES'].iteritems()}
    else:
        params['nsamples'] = int(params_dict['WAVEFORMS_NSAMPLES'])
        
    # Number of features.
    if isinstance(params_dict['FETDIM'], dict):
        params['fetdim'] = {int(key): value 
            for key, value in params_dict['FETDIM'].iteritems()}
    else:
        params['fetdim'] = int(params_dict['FETDIM'])
        
    # if not 'nsamples' in params:
        # # Get the number of samples per waveform from the PARAMS file.
        # params['nsamples'] = int(f * float(params_dict['T_BEFORE']) + float(params_dict['T_AFTER']))
    
    return params
    

    

