"""Functions that generate mock data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil

from kwiklib.utils.colors import COLORS_COUNT
from kwiklib.dataio import (save_binary, save_text, check_dtype, 
    check_shape, save_cluster_info, save_group_info)
from klustaviewa.stats.cache import IndexedMatrix
import kwiklib.utils.logger as log


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# Mock parameters.
nspikes = 1000
nclusters = 20
nextrafet = 1
ngroups = 4
cluster_offset = 2
nsamples = 20
ncorrbins = 100
corrbin = .001
nchannels = 32
fetdim = 3
duration = 1.
freq = 20000.


# -----------------------------------------------------------------------------
# Data creation methods
# -----------------------------------------------------------------------------
def create_waveforms(nspikes, nsamples, nchannels):
    t = np.linspace(-np.pi, np.pi, nsamples)
    t = t.reshape((1, -1, 1))
    # Sinus shaped random waveforms.
    return (np.array(rnd.randint(size=(nspikes, nsamples, nchannels),
        low=-32768 // 2, high=32768 // 2), dtype=np.int16) -
            np.array(32768 // 2 * (.5 + .5 * rnd.rand()) * np.cos(t),
            dtype=np.int16))
    
def create_rawdata(nsamples, nchannels):
    noise = np.array(rnd.randint(size=(nsamples, nchannels),
        low=-1000, high=1000), dtype=np.int16)
    t = np.linspace(0., 100., nsamples)
    low = np.array(10000 * np.cos(t), dtype=np.int16)
    return noise + low[:, np.newaxis]
    
def create_features(nspikes, nchannels, fetdim, duration, freq):
    features = np.array(rnd.randint(size=(nspikes, nchannels * fetdim + 1),
        low=-1e5, high=1e5), dtype=np.float32)
    features[:, -1] = np.sort(np.random.randint(size=nspikes, low=0,
        high=duration * freq))
    return features
    
def create_clusters(nspikes, nclusters, cluster_offset=cluster_offset):
    # Add shift in cluster indices to test robustness.
    return rnd.randint(size=nspikes, low=cluster_offset, 
        high=nclusters + cluster_offset)
    
def create_cluster_colors(nclusters):
    return np.mod(np.arange(nclusters, dtype=np.int32), COLORS_COUNT) + 1
   
def create_group_colors(ngroups):
    return np.mod(np.arange(ngroups, dtype=np.int32), COLORS_COUNT) + 1
    
def create_group_names(ngroups):
    return ["Group {0:d}".format(group) for group in xrange(ngroups)]
    
def create_cluster_groups(nclusters):
    return np.array(np.random.randint(size=nclusters, low=0, high=4), 
        dtype=np.int32)
    
def create_masks(nspikes, nchannels, fetdim):
    return np.clip(rnd.rand(nspikes, nchannels * fetdim + 1) * 1.5, 0, 1)
    
def create_similarity_matrix(nclusters):
    return np.random.rand(nclusters, nclusters)
    
def create_correlograms(clusters, ncorrbins):
    n = len(clusters)
    shape = (n, n, ncorrbins)
    # data = np.clip(np.random.rand(*shape), .75, 1)
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
    # return np.random.randint(size=(nchannels, 2), low=0, high=10)
    geometry = np.zeros((nchannels, 2), dtype=np.int32)
    geometry[:, 0] = np.arange(nchannels)
    geometry[::2, 0] *= -1
    geometry[:, 1] = np.arange(nchannels)
    
    graph = [(i, (i + 1) % nchannels) for i in xrange(nchannels)]
    
    probe = {'probes': {1: graph}, 
             'geometry': {i: tuple(geometry[i, :]) for i in xrange(nchannels)}}
    
    probe_python = "probes = {0:s}\ngeometry = {{1: {1:s}}}\n".format(
        str(probe['probes']),
        str(probe['geometry']),
    )
    return probe_python

    
# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    # log.debug("Creating mock data for dataio subpackage.")
    
    # Create mock directory if needed.
    dir = TEST_FOLDER
    if not os.path.exists(dir):
        os.mkdir(dir)
    # else:
        # # No need to recreate the files.
        # return
        
    # Create mock data.
    waveforms = create_waveforms(nspikes, nsamples, nchannels)
    features = create_features(nspikes, nchannels, fetdim, duration, freq)
    clusters = create_clusters(nspikes, nclusters)
    
    cluster_colors = create_cluster_colors(nclusters)
    cluster_groups = create_cluster_groups(nclusters)
    cluster_info = pd.DataFrame(
            {'color': cluster_colors, 
             'group': cluster_groups}, 
         dtype=np.int32,
         index=np.unique(clusters))
         
    group_colors = create_group_colors(ngroups)
    group_names = create_group_names(ngroups)
    group_info = pd.DataFrame(
            {'color': group_colors, 
             'name': group_names}, 
         index=np.arange(ngroups))
         
    masks = create_masks(nspikes, nchannels, fetdim)
    xml = create_xml(nchannels, nsamples, fetdim)
    probe = create_probe(nchannels)
    
    
    # Create mock files.
    save_binary(os.path.join(dir, 'test.spk.1'), waveforms)
    save_text(os.path.join(dir, 'test.fet.1'), features,
        header=nchannels * fetdim + 1)
    save_text(os.path.join(dir, 'test.aclu.1'), clusters, header=nclusters)
    # save_cluster_info(os.path.join(dir, 'test.acluinfo.1'), cluster_info)
    # save_group_info(os.path.join(dir, 'test.groupinfo.1'), group_info)
    save_text(os.path.join(dir, 'test.clu.1'), clusters, header=nclusters)
    save_text(os.path.join(dir, 'test.fmask.1'), masks, header=nclusters,
        fmt='%.6f')
    save_text(os.path.join(dir, 'test.xml'), xml)
    save_text(os.path.join(dir, 'test.probe'), probe)
    
def teardown():
    # log.debug("Erasing mock data for dataio subpackage.")
    
    # Erase the temporary data directory.
    dir = TEST_FOLDER
    # if os.path.exists(dir):
        # shutil.rmtree(dir, ignore_errors=True)
    # Erase the contents instead, otherwise run into Access denied errors
    # when trying to re-create the directory right after it has been deleted.
    for the_file in os.listdir(dir):
        file_path = os.path.join(dir, the_file)
        try:
            os.unlink(file_path)
        except:
            pass
        
        
        
