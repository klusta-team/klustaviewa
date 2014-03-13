"""Get the keyword arguments for the views from the loader."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from spikedetekt2.dataio import *
import kwiklib.utils.logger as log
from kwiklib.dataio import (get_some_spikes_in_clusters, get_indices, 
    get_spikes_in_clusters, get_some_spikes, pandaize)
from kwiklib.utils.colors import random_color

from klustaviewa.stats.correlations import normalize
from klustaviewa.stats.correlograms import get_baselines, NCORRBINS_DEFAULT, CORRBIN_DEFAULT
from klustaviewa import USERPREF
from klustaviewa import SETTINGS
from klustaviewa.gui.threads import ThreadedTasks


# -----------------------------------------------------------------------------
# Get data from loader for views
# -----------------------------------------------------------------------------
def get_waveformview_data(exp, clusters=[], channel_group=0, clustering='main',
                          autozoom=None, wizard=None):
    clusters = np.array(clusters)
    fetdim = exp.application_data.spikedetekt.nfeatures_per_channel
    
    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    spikes_data = exp.channel_groups[channel_group].spikes
    channels_data = exp.channel_groups[channel_group].channels
    
    spike_clusters = getattr(spikes_data.clusters, clustering)[:]
    spikes_selected = get_some_spikes_in_clusters(clusters, spike_clusters)
    spike_clusters = spike_clusters[spikes_selected]
    cluster_colors = clusters_data.color[clusters]

    _, nsamples, nchannels = spikes_data.waveforms_filtered.shape
    if len(spikes_selected) > 0:
        waveforms = convert_dtype(
            spikes_data.waveforms_filtered[spikes_selected,...],
            np.float32)
        # Normalize waveforms.
        waveforms = waveforms * 1. / (waveforms.max())
        masks = spikes_data.masks[spikes_selected,::fetdim]
    else:
        waveforms = np.zeros((0, nsamples, nchannels), dtype=np.float32)
        masks = np.zeros((0, nchannels), dtype=np.float32)
    
    channel_positions = np.array([channels_data[ch].position or (0., ch) 
                                  for ch in channels_data.keys()])
    
    
    # Pandaize
    waveforms = pandaize(waveforms, spikes_selected)
    spike_clusters = pandaize(spike_clusters, spikes_selected)
    masks = pandaize(masks, spikes_selected)
    cluster_colors = pandaize(cluster_colors, clusters)
    
    data = dict(
        waveforms=waveforms,
        clusters=spike_clusters,
        cluster_colors=cluster_colors,
        clusters_selected=clusters,
        masks=masks,
        geometrical_positions=channel_positions,
        autozoom=autozoom,
        keep_order=wizard,
    )
    return data

def get_featureview_data(exp, clusters=[], channel_group=0, clustering='main',
                         nspikes_bg=None, autozoom=None,
                         alpha_selected=.75, alpha_background=.25,
                         normalization=None,
                         time_unit='second'):
    clusters = np.array(clusters)
    # TODO: add spikes=None and spikes_bg=None
    fetdim = exp.application_data.spikedetekt.nfeatures_per_channel
    nchannels = exp.application_data.spikedetekt.nchannels
    
    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    spikes_data = exp.channel_groups[channel_group].spikes
    channels_data = exp.channel_groups[channel_group].channels
    
    spike_clusters = getattr(spikes_data.clusters, clustering)[:]
    spikes_selected = get_spikes_in_clusters(clusters, spike_clusters)
    spikes_bg = get_some_spikes(spike_clusters, nspikes_max=nspikes_bg)
    cluster_colors = clusters_data.color[clusters]
    
    # HACK: work-around PyTables bug #310: expand the dimensions of the boolean 
    # indices
    # ind = np.tile(spikes_selected[:, np.newaxis, np.newaxis], 
                  # (1,) + spikes_data.features_masks.shape[1:])
    # fm = spikes_data.features_masks[ind].reshape((-1,) + spikes_data.features_masks.shape[1:])
    
    # HACK: need modification in PyTables as described here
    # https://github.com/PyTables/PyTables/pull/317#issuecomment-34210551
    spikes_selected = np.nonzero(spikes_selected)[0]
    _, nspikes, _ = spikes_data.features_masks.shape
    if len(spikes_selected) > 0:
        fm = spikes_data.features_masks[spikes_selected]
    else:
        fm = np.zeros((0, nspikes, 2), dtype=spikes_data.features_masks.dtype)
    
    features = fm[:, :, 0]
    masks = fm[:, ::fetdim, 1]
    
    nspikes = features.shape[0]
    spiketimes = spikes_data.time_samples[spikes_selected]
    spike_clusters = spike_clusters[spikes_selected]
    freq = exp.application_data.spikedetekt.sample_rate
    duration = spikes_data.time_samples[len(spikes_data.time_samples)-1]*1./freq
    
    # No need for hacky work-around here, since get_spikes returns a slice.
    features_bg = spikes_data.features_masks[spikes_bg, :, 0]
    
    # Normalize features.
    c = (normalization or (1. / (features.max()))) if nspikes > 0 else 1.
    features = features * c
    features_bg = features_bg * c
    
    # Pandaize
    features = pandaize(features, spikes_selected)
    features_bg = pandaize(features_bg, spikes_bg)
    masks = pandaize(masks, spikes_selected)
    spiketimes = pandaize(spiketimes, spikes_selected)
    spike_clusters = pandaize(spike_clusters, spikes_selected)
    cluster_colors = pandaize(cluster_colors, clusters)
    
    # TODO
    nextrafet = 0
    
    data = dict(
        features=features,
        features_background=features_bg,
        masks=masks,
        spiketimes=spiketimes,
        clusters=spike_clusters,
        clusters_selected=clusters,
        cluster_colors=cluster_colors,
        nchannels=nchannels,
        fetdim=fetdim,
        nextrafet=nextrafet,
        freq=freq,
        autozoom=autozoom,
        duration=duration,
        alpha_selected=alpha_selected,
        alpha_background=alpha_background,
        time_unit=time_unit,
    )
    return data

def get_clusterview_data(exp, statscache=None, channel_group=0, 
                         clustering='main',):
    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    cluster_groups_data = getattr(exp.channel_groups[channel_group].cluster_groups, clustering)
    
    # Get the list of all existing clusters.
    clusters = sorted(clusters_data.keys())
    groups = cluster_groups_data.keys()
    
    cluster_colors = pd.Series([clusters_data[cl].application_data.klustaviewa.color or 1
                           for cl in clusters], index=clusters)
    cluster_groups = pd.Series([clusters_data[cl].cluster_group or 0
                               for cl in clusters], index=clusters)
                                
    group_colors = pd.Series([cluster_groups_data[g].application_data.klustaviewa.color or 1
                             for g in groups], index=range(len(groups)))
    group_names = pd.Series([cluster_groups_data[g].name or 'Group'
                            for g in groups], index=range(len(groups)))
    
    # TODO: cache the cluster size instead of recomputing every time here
    # (in experiment class?)
    spike_clusters = getattr(exp.channel_groups[channel_group].spikes.clusters, 
                             clustering)[:]
    sizes = np.bincount(spike_clusters)
    cluster_sizes = sizes[clusters]
    
    data = dict(
        cluster_colors=cluster_colors,
        cluster_groups=cluster_groups,
        group_colors=group_colors,
        group_names=group_names,
        cluster_sizes=cluster_sizes,
    )
    if statscache is not None:
        data['cluster_quality'] = statscache.cluster_quality
    return data
    
def get_correlogramsview_data(exp, correlograms, clusters=[],
                              channel_group=0, clustering='main',
                              nclusters_max=10, ncorrbins=50, corrbin=.001):
    clusters = np.array(clusters, dtype=np.int32)
    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    cluster_groups_data = getattr(exp.channel_groups[channel_group].cluster_groups, clustering)
    
    cluster_colors = clusters_data.color[clusters]
    cluster_colors = pandaize(cluster_colors, clusters)
                            
    # TODO: cache and optimize this
    spike_clusters = getattr(exp.channel_groups[channel_group].spikes.clusters, 
                             clustering)[:]
    sizes = np.bincount(spike_clusters)
    cluster_sizes = sizes[clusters]
    
    
    clusters_selected0 = clusters
    
    # Subset of selected clusters if there are too many clusters.
    if len(clusters_selected0) < nclusters_max:
        clusters_selected = clusters_selected0
    else:
        clusters_selected = clusters_selected0[:nclusters_max]
    
    correlograms = correlograms.submatrix(clusters_selected)
        
    # Compute the baselines.
    # colors = select(loader.get_cluster_colors(), clusters_selected)
    # corrbin = SETTINGS.get('correlograms.corrbin', CORRBIN_DEFAULT)
    # ncorrbins = SETTINGS.get('correlograms.ncorrbins', NCORRBINS_DEFAULT)
    duration = corrbin * ncorrbins
    baselines = get_baselines(cluster_sizes, duration, corrbin)
    data = dict(
        correlograms=correlograms,
        baselines=baselines,
        clusters_selected=clusters_selected,
        cluster_colors=cluster_colors,
        ncorrbins=ncorrbins,
        corrbin=corrbin,
    )
    return data
    
def get_similaritymatrixview_data(exp, matrix=None,
        channel_group=0, clustering='main',):
    if matrix is None:
        return
    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    cluster_groups_data = getattr(exp.channel_groups[channel_group].cluster_groups, clustering)
    clusters = sorted(clusters_data.keys())
    cluster_colors = pd.Series([clusters_data[cl].application_data.klustaviewa.color or 1
                           for cl in clusters], index=clusters)
    cluster_groups = pd.Series([clusters_data[cl].cluster_group or 0
                               for cl in clusters], index=clusters)
                       
        
    # Clusters in groups 0 or 1 to hide.
    clusters_hidden = np.nonzero(np.in1d(cluster_groups, [0, 1]))[0]
    data = dict(
        # WARNING: copy the matrix here so that we don't modify the
        # original matrix while normalizing it.
        similarity_matrix=matrix,
        cluster_colors_full=cluster_colors,
        clusters_hidden=clusters_hidden,
    )
    return data
    
    
    
# TODO: loader ==> exp (supporting new file format)
def get_traceview_data(loader):
    return loader.get_trace()
    
def get_channelview_data(loader, channels=None):
    data = dict(
        channel_colors=loader.get_channel_colors('all',
            can_override=False),
        channel_groups=loader.get_channel_groups('all'),
        channel_names=loader.get_channel_names('all'),
        group_colors=loader.get_channel_group_colors('all'),
        group_names=loader.get_channel_group_names('all'),
    )
    return data
    