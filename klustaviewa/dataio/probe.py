"""This module provides functions used to load probe files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import pprint
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
def flatten(l):
    return sorted(set([item for sublist in l for item in sublist]))

def linear_probe(shanks):
    """shanks is a dict {shank: nchannels}."""
    # Linear graph.
    graph = {shank: [(i, (i + 1)) for i in xrange(nchannels - 1)] 
        for shank, nchannels in shanks.iteritems()}
    # Linear geometry.
    geometry = {
        shank:
            {channel: (0., float(channel)) for channel in xrange(nchannels)}
                for shank, nchannels in shanks.iteritems()}
    
    probe_python = "probes = {0:s}\ngeometry = {1:s}\n".format(
        pprint.pformat(graph, indent=4),
        pprint.pformat(geometry, indent=4),
    )
    return probe_python
    
def all_to_all_probe(shanks):
    """shanks is a dict {shank: nchannels}."""
    # All-to-all graph.
    graph = {shank: [(i, j) 
        for i in xrange(nchannels) 
            for j in xrange(i + 1, nchannels)] 
                for shank, nchannels in shanks.iteritems()}
    # Linear geometry.
    geometry = {
        shank:
            {channel: (0., float(channel)) for channel in xrange(nchannels)}
                for shank, nchannels in shanks.iteritems()}
    
    probe_python = "probes = {0:s}\ngeometry = {1:s}\n".format(
        pprint.pformat(graph, indent=4),
        pprint.pformat(geometry, indent=4),
    )
    return probe_python
    
def probe_to_json(probe_ns):
    """Convert from the old Python .probe file to the new JSON format."""
    graph = probe_ns['probes']
    shanks = sorted(graph.keys())
    if 'geometry' in probe_ns:
        geometry = probe_ns['geometry']
    else:
        geometry = None
    # Find the list of shanks.
    shank_channels = {shank: flatten(graph[shank]) for shank in shanks}
    # Find the list of channels.
    channels = flatten(shank_channels.values())
    nchannels = len(channels)
    # Create JSON dictionary.
    json_dict = {
        'nchannels': nchannels,
        'channel_names': {channel: 'ch{0:d}'.format(channel) 
            for channel in channels},
        'dead_channels': [],
        'shanks': [
                    {
                        'shank_index': shank,
                        'channels': shank_channels[shank],
                        'graph': graph[shank],
                    }
                    for shank in shanks
                  ]
            }
    # Add the geometry if it exists.
    if geometry:
        # Find out if there's one geometry per shank, or a common geometry
        # for all shanks.
        multiple = shank in geometry and isinstance(geometry[shank], dict)
        for shank_dict in json_dict['shanks']:
            shank = shank_dict['shank_index']
            if multiple:
                shank_dict['geometry'] = geometry[shank]
            else:
                shank_dict['geometry'] = geometry
    return json.dumps(json_dict, indent=4)
    
def load_probe_json(probe_json):
    if not probe_json:
        return None
    probe_dict = json.loads(probe_json)
    probe = {}
    probe['nchannels'] = int(probe_dict['nchannels'])
    probe['dead_channels'] = map(int, probe_dict['dead_channels'])
    # List of all channels.
    probe['channels'] = sorted(map(int, probe_dict['channel_names'].keys()))
    # List of alive channels.
    probe['channels_alive'] = sorted(map(int, set(probe['channels']) - 
        set(probe['dead_channels'])))
    probe['nchannels_alive'] = len(probe['channels_alive'])
    # Process all shanks.
    for shank_dict in probe_dict['shanks']:
        # Find alive channels.
        shank_dict['channels_alive'] = sorted(map(int, set(shank_dict['channels']) - 
            set(probe['dead_channels'])))
        # Convert the geometry dictionary into an array.
        if 'geometry' in shank_dict:
            # Convert the keys from strings to integers.
            shank_dict['geometry'] = {int(key): val 
                for key, val in shank_dict['geometry'].iteritems()}
            # Create the geometry array with alive channels only.
            shank_dict['geometry_array'] = np.array(
                [shank_dict['geometry'][key] 
                    for key in sorted(shank_dict['geometry'].keys())
                        if key not in probe['dead_channels']], 
                dtype=np.float32)
        probe[shank_dict['shank_index']] = shank_dict
    return probe
    

    

