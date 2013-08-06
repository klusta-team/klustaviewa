"""Functions that generate mock data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil

from klustaviewa.utils.colors import COLORS_COUNT
from klustaviewa.dataio import MemoryLoader
from klustaviewa.dataio.tools import normalize
from klustaviewa.stats.cache import IndexedMatrix


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# Mock parameters.
nspikes = 1000
nclusters = 20
nextrafet = 1
cluster_offset = 2
ngroups = 3
nsamples = 20
ncorrbins = 100
corrbin = .001
nchannels = 32
fetdim = 3
duration = 60.
freq = 20000.


# -----------------------------------------------------------------------------
# Data creation methods
# -----------------------------------------------------------------------------
def create_waveforms(nspikes, nsamples, nchannels):
    t = np.linspace(-np.pi, np.pi, nsamples)
    t = t.reshape((1, -1, 1))
    # Sinus shaped random waveforms.
    waveforms = (np.array(rnd.randint(size=(nspikes, nsamples, nchannels),
        low=-32768 // 2, high=32768 // 2), dtype=np.int16) -
            np.array(32768 // 2 * (.5 + .5 * rnd.rand()) * np.cos(t),
            dtype=np.int16))
    return waveforms
    
def create_features(nspikes, nchannels, fetdim, duration, freq):
    features = np.array(rnd.randint(size=(nspikes, nchannels * fetdim + 1),
        low=-1e5, high=1e5), dtype=np.float32)
    features[:, -1] = np.sort(np.random.randint(size=nspikes, low=0,
        high=duration * freq))
    return features
    
def create_clusters(nspikes, nclusters):
    clusters = rnd.randint(size=nspikes + 1, low=cluster_offset, 
        high=nclusters + cluster_offset)
    clusters[0] = nclusters
    return clusters
    
def create_cluster_info(nclusters, cluster_offset):
    cluster_info = np.zeros((nclusters, 3), dtype=np.int32)
    cluster_info[:, 0] = np.arange(2, nclusters + cluster_offset)
    cluster_info[:, 1] = np.mod(np.arange(nclusters, 
        dtype=np.int32), COLORS_COUNT) + 1
    # First column: color index, second column: group index (2 by
    # default)
    cluster_info[:, 2] = 2 * np.ones(nclusters)
    return cluster_info
    
def create_group_info(ngroups):
    group_info = np.zeros((ngroups, 3), dtype=object)
    group_info[:, 0] = np.arange(ngroups)
    group_info[:, 1] = np.mod(np.arange(ngroups), COLORS_COUNT) + 1
    group_info[:, 2] = np.array(['Group {0:d}'.format(i) 
        for i in xrange(ngroups)], dtype=object)
    return group_info
    
def create_masks(nspikes, nchannels, fetdim):
    return np.clip(rnd.rand(nspikes, nchannels * fetdim + 1) * 1.5, 0, 1)
    
def create_similarity_matrix(nclusters):
    return np.random.rand(nclusters, nclusters)
    
def create_correlograms(clusters, ncorrbins):
    n = len(np.unique(clusters))
    shape = (n, n, ncorrbins)
    data = np.random.rand(*shape)
    data[0, 0] /= 10
    data[1, 1] *= 10
    return IndexedMatrix(clusters, shape=shape,
        data=data)
    
def create_baselines(clusters):
    baselines = np.clip(np.random.rand(len(clusters), len(clusters)), .75, 1)
    baselines[0, 0] /= 10
    baselines[1, 1] *= 10
    return baselines
    
def create_xml(nchannels, nsamples, fetdim):
    channels = '\n'.join(["<channel>{0:d}</channel>".format(i) 
        for i in xrange(nchannels)])
    xml = """
    <parameters>
      <acquisitionSystem>
        <nBits>16</nBits>
        <nChannels>{0:d}</nChannels>
        <samplingRate>20000</samplingRate>
        <voltageRange>20</voltageRange>
        <amplification>1000</amplification>
        <offset>2048</offset>
      </acquisitionSystem>
      <anatomicalDescription>
        <channelGroups>
          <group>
            {2:s}
          </group>
        </channelGroups>
      </anatomicalDescription>
      <spikeDetection>
        <channelGroups>
          <group>
            <channels>
              {2:s}
            </channels>
            <nSamples>{1:d}</nSamples>
            <peakSampleIndex>10</peakSampleIndex>
            <nFeatures>{3:d}</nFeatures>
          </group>
        </channelGroups>
      </spikeDetection>
    </parameters>
    """.format(nchannels, nsamples, channels, fetdim)
    return xml

def create_probe(nchannels):
    probe = np.zeros((nchannels, 2), dtype=np.int32)
    probe[:, 0] = np.arange(nchannels)
    probe[::2, 0] *= -1
    probe[:, 1] = np.arange(nchannels)
    return probe


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
LOADER = None

def setup():
    waveforms = create_waveforms(nspikes, nsamples, nchannels)
    features = create_features(nspikes, nchannels, fetdim, duration, freq)
    clusters = create_clusters(nspikes, nclusters)
    masks = create_masks(nspikes, nchannels, fetdim)
    cluster_info = create_cluster_info(nclusters, cluster_offset)
    group_info = create_group_info(ngroups)
    similarity_matrix = create_similarity_matrix(nclusters)
    correlograms = create_correlograms(clusters, ncorrbins)
    baselines = create_baselines(clusters)
    probe = create_probe(nchannels)
    
    global LOADER
    LOADER = MemoryLoader(
        nsamples=nsamples,
        nchannels=nchannels,
        fetdim=fetdim,
        freq=freq,
        waveforms=waveforms,
        features=features,
        clusters=clusters,
        masks=masks,
        cluster_info=cluster_info,
        group_info=group_info,
        probe=probe,
    )
    
    return LOADER
    
def teardown():
    pass
    
    