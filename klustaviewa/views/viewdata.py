"""Get the keyword arguments for the views from the loader."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from kwiklib.dataio import *
from kwiklib.utils import logger as log
from kwiklib.dataio import (get_some_spikes_in_clusters, get_indices,
    get_spikes_in_clusters, get_some_spikes, pandaize)
from kwiklib.utils.colors import random_color, next_color

from klustaviewa.stats.correlations import normalize
from klustaviewa.stats.correlograms import get_baselines, NCORRBINS_DEFAULT, CORRBIN_DEFAULT
from klustaviewa import USERPREF
from klustaviewa import SETTINGS
from klustaviewa.gui.threads import ThreadedTasks


# -----------------------------------------------------------------------------
# Get data from loader for views
# -----------------------------------------------------------------------------

def _get_color(clusters_data, cl):
    if cl not in clusters_data:
        return next_color(cl)
    cluster_data = clusters_data[cl]
    try:
        out = cluster_data.application_data.klustaviewa.color or 1
        return out
    except AttributeError as e:
        return next_color(cl)


def get_waveformview_data(exp, clusters=[], channel_group=0, clustering='main',
                          autozoom=None, wizard=None):
    clusters = np.array(clusters)
    fetdim = exp.application_data.spikedetekt.n_features_per_channel

    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    spikes_data = exp.channel_groups[channel_group].spikes
    channels_data = exp.channel_groups[channel_group].channels
    channels = exp.channel_groups[channel_group].channel_order

    spike_clusters = getattr(spikes_data.clusters, clustering)[:]
    # spikes_selected = get_some_spikes_in_clusters(clusters, spike_clusters)

    # cluster_colors = clusters_data.color[clusters]
    # get colors from application data:
    cluster_colors = pd.Series([_get_color(clusters_data, cl)
                                for cl in clusters], index=clusters)
    # cluster_colors = pd.Series([
    #     next_color(cl)
    #         if cl in clusters_data else 1
    #                        for cl in clusters], index=clusters)

    if spikes_data.waveforms_filtered is None:

        data = dict(
            waveforms=None,
            channels=channels,
            clusters=None,
            cluster_colors=None,
            clusters_selected=clusters,
            masks=None,
            geometrical_positions=None,
            autozoom=autozoom,
            keep_order=wizard,
        )

        return data

    _, nsamples, nchannels = spikes_data.waveforms_filtered.shape

    # Find spikes to display and load the waveforms.
    if len(clusters) > 0:
        spikes_selected, waveforms = spikes_data.load_waveforms(clusters=clusters,
            count=USERPREF['waveforms_nspikes_max_expected'])
    else:
        spikes_selected = []

    # Bake the waveform data.
    if len(spikes_selected) > 0:
        waveforms = convert_dtype(waveforms, np.float32)
        if spikes_data.masks is not None:
            masks = spikes_data.masks[spikes_selected, 0:fetdim*nchannels:fetdim]
        else:
            masks = None
    else:
        waveforms = np.zeros((0, nsamples, nchannels), dtype=np.float32)
        masks = np.ones((0, nchannels), dtype=np.float32)

    if masks is None:
        masks = np.ones((len(spikes_selected), nchannels), dtype=np.float32)

    spike_clusters = spike_clusters[spikes_selected]
    channel_positions = np.array([channels_data[ch].position
                                  if channels_data[ch].position is not None
                                  else (0., ch)
                                  for ch in channels],
                                 dtype=np.float32)

    # Pandaize
    waveforms = pandaize(waveforms, spikes_selected)
    spike_clusters = pandaize(spike_clusters, spikes_selected)
    masks = pandaize(masks, spikes_selected)
    cluster_colors = pandaize(cluster_colors, clusters)

    data = dict(
        waveforms=waveforms,
        channels=channels,
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
    fetdim = exp.application_data.spikedetekt.n_features_per_channel


    channels = exp.channel_groups[channel_group].channel_order

    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    spikes_data = exp.channel_groups[channel_group].spikes
    channels_data = exp.channel_groups[channel_group].channels
    nchannels = len(channels_data)

    spike_clusters = getattr(spikes_data.clusters, clustering)[:]
    # cluster_colors = clusters_data.color[clusters]
    # get colors from application data:
    cluster_colors = pd.Series([_get_color(clusters_data, cl)
                                for cl in clusters], index=clusters)
    # cluster_colors = pd.Series([
    #     next_color(cl)
    #         if cl in clusters_data else 1
    #                        for cl in clusters], index=clusters)

    if len(clusters) > 0:
        # TODO: put fraction in user parameters
        spikes_selected, fm = spikes_data.load_features_masks(clusters=clusters)
    else:
        spikes_selected = []
        fm = np.zeros((0, spikes_data.features_masks.shape[1], 2),
                      dtype=spikes_data.features_masks.dtype)

    fm = np.atleast_3d(fm)

    features = fm[:, :, 0]
    nextrafet = features.shape[1] - nchannels * fetdim

    if fm.shape[2] > 1:
        masks = fm[:, ::fetdim, 1]
    else:
        masks = None

    nspikes = features.shape[0]
    spiketimes_all = spikes_data.concatenated_time_samples[:]
    spiketimes = spiketimes_all[spikes_selected]
    spike_clusters = spike_clusters[spikes_selected]
    freq = exp.application_data.spikedetekt.sample_rate
    duration = spikes_data.concatenated_time_samples[len(spikes_data.concatenated_time_samples)-1]*1./freq

    spikes_bg, features_bg = spikes_data.load_features_masks_bg()

    features_bg = np.atleast_3d(features_bg)

    features_bg = features_bg[:,:,0].copy()
    spiketimes_bg = spiketimes_all[spikes_bg]

    # Add extra feature for time is necessary.
    if nextrafet == 0:
        features = np.hstack((features, np.ones((features.shape[0], 1))))
        features_bg = np.hstack((features_bg, np.ones((features_bg.shape[0], 1))))
        nextrafet = 1

    # Normalize features.
    def _find_max(x):
        if x.size == 0:
            return 1.
        return np.max(np.abs(x))

    c = (normalization or (1. / _find_max(features_bg[:,:-nextrafet]))) if nspikes > 0 else 1.
    features[:,:-nextrafet] *= c
    features_bg[:,:-nextrafet] *= c

    # Normalize extra features except time.
    for i in range(features_bg.shape[1]-nextrafet-1, features_bg.shape[1]-1):
        c = (1. / _find_max(features_bg[:,i])) if nspikes > 0 else 1.
        features[:,i] *= c
        features_bg[:,i] *= c

    # Normalize time.
    if features.size > 0:
        features[:,-1] = spiketimes
        features[:,-1] *= (1. / (duration * freq))
        features[:,-1] = 2 * features[:,-1] - 1

        features_bg[:,-1] = spiketimes_bg
        features_bg[:,-1] *= (1. / (duration * freq))
        features_bg[:,-1] = 2 * features_bg[:,-1] - 1

        # Pandaize
        features = pandaize(features, spikes_selected)
        features_bg = pandaize(features_bg, spikes_bg)
        if masks is not None:
            masks = pandaize(masks, spikes_selected)

    spiketimes = pandaize(spiketimes, spikes_selected)
    spike_clusters = pandaize(spike_clusters, spikes_selected)
    cluster_colors = pandaize(cluster_colors, clusters)

    # nextrafet = features.shape[1] - fetdim * nchannels


    data = dict(
        features=features,
        features_background=features_bg,
        masks=masks,
        spiketimes=spiketimes,
        clusters=spike_clusters,
        clusters_selected=clusters,
        cluster_colors=cluster_colors,
        nchannels=nchannels,
        channels=channels,
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
    # clusters = sorted(clusters_data.keys())

    spike_clusters = getattr(exp.channel_groups[channel_group].spikes.clusters,
                             clustering)[:]
    clusters = np.unique(spike_clusters)
    groups = cluster_groups_data.keys()

    # cluster_groups = pd.Series([clusters_data[cl].cluster_group or 0
    #                            for cl in clusters], index=clusters)
    # Make sure there's no crash if this is called before the clusters had a chance
    # to be added in the HDF5 file.

    # get colors from application data:
    cluster_colors = pd.Series([_get_color(clusters_data, cl)
                                for cl in clusters], index=clusters)

    # cluster_colors = pd.Series([
    #     next_color(cl)
    #         if cl in clusters_data else 1
    #                        for cl in clusters], index=clusters)

    cluster_groups = pd.Series([
        (clusters_data[cl].cluster_group
                                if clusters_data[cl].cluster_group is not None else 3)
            if cl in clusters_data else 3
                               for cl in clusters], index=clusters)

    group_colors = pd.Series([next_color(cl)
                             for g in groups], index=groups)
    group_names = pd.Series([cluster_groups_data[g].name or 'Group'
                            for g in groups], index=groups)
    # TODO: cache the cluster size instead of recomputing every time here
    # (in experiment class?)
    spike_clusters = getattr(exp.channel_groups[channel_group].spikes.clusters,
                             clustering)[:]
    sizes = np.bincount(spike_clusters)
    cluster_sizes = pd.Series(sizes[clusters], index=clusters)


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
                              channel_group=0, clustering='main', wizard=None,
                              nclusters_max=None, ncorrbins=50, corrbin=.001):

    clusters = np.array(clusters, dtype=np.int32)
    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    cluster_groups_data = getattr(exp.channel_groups[channel_group].cluster_groups, clustering)
    freq = exp.application_data.spikedetekt.sample_rate

    # cluster_colors = clusters_data.color[clusters]
    # cluster_colors = pandaize(cluster_colors, clusters)

    # get colors from application data:
    cluster_colors = pd.Series([_get_color(clusters_data, cl)
                                for cl in clusters], index=clusters)

    # cluster_colors = pd.Series([
    #     next_color(cl)
    #         if cl in clusters_data else 1
    #                        for cl in clusters], index=clusters)

    # TODO: cache and optimize this
    spike_clusters = getattr(exp.channel_groups[channel_group].spikes.clusters,
                             clustering)[:]
    sizes = np.bincount(spike_clusters)
    cluster_sizes = sizes[clusters]

    clusters_selected0 = clusters
    nclusters_max = nclusters_max or USERPREF['correlograms_max_nclusters']

    # Subset of selected clusters if there are too many clusters.
    if len(clusters_selected0) < nclusters_max:
        clusters_selected = clusters_selected0
    else:
        clusters_selected = clusters_selected0[:nclusters_max]

    correlograms = correlograms.submatrix(clusters_selected)
    cluster_colors = select(cluster_colors, clusters_selected)

    # Compute the baselines.
    # corrbin = SETTINGS.get('correlograms.corrbin', CORRBIN_DEFAULT)
    # ncorrbins = SETTINGS.get('correlograms.ncorrbins', NCORRBINS_DEFAULT)
    duration = exp.channel_groups[channel_group].spikes.concatenated_time_samples[:][-1] - exp.channel_groups[channel_group].spikes.concatenated_time_samples[:][0]
    duration /= freq
    if duration == 0:
        duration = 1.
    baselines = get_baselines(cluster_sizes, duration, corrbin)
    baselines = baselines[:nclusters_max,:nclusters_max]

    data = dict(
        correlograms=correlograms,
        baselines=baselines,
        clusters_selected=clusters_selected,
        cluster_colors=cluster_colors,
        ncorrbins=ncorrbins,
        corrbin=corrbin,
        keep_order=wizard,
    )

    return data

def get_similaritymatrixview_data(exp, matrix=None,
        channel_group=0, clustering='main',):
    if matrix is None:
        return {}
    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    cluster_groups_data = getattr(exp.channel_groups[channel_group].cluster_groups, clustering)
    clusters = sorted(clusters_data.keys())

    # get colors from application data:
    cluster_colors = pd.Series([_get_color(clusters_data, cl)
                                for cl in clusters], index=clusters)

    # cluster_colors = pd.Series([next_color(cl)
    #                        for cl in clusters], index=clusters)

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


def get_traceview_data(exp,
        channel_group=0, clustering='main'):

    if (len(exp.recordings) == 0) or exp.recordings[0].raw == None:
        data = dict(
            trace=None,
            )
        return data

    rawdata = exp.recordings[0].raw
    freq = exp.application_data.spikedetekt.sample_rate
    clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
    clusters = sorted(clusters_data.keys())
    spikes_data = exp.channel_groups[channel_group].spikes
    channels = exp.channel_groups[channel_group].channel_order
    spiketimes = spikes_data.time_samples
    spikeclusters = getattr(spikes_data.clusters, clustering)[:]

    _, nsamples, nchannels = spikes_data.waveforms_filtered.shape

    freq = exp.application_data.spikedetekt.sample_rate

    cluster_colors = pd.Series([next_color(cl)
                       for cl in clusters], index=clusters)
    fetdim = exp.application_data.spikedetekt.n_features_per_channel

    s_before = exp.application_data.spikedetekt.extract_s_before
    s_after = exp.application_data.spikedetekt.extract_s_after

    if spikes_data.masks is not None:
        spikemasks = np.zeros((spikes_data.masks.shape[0], rawdata.shape[1]))
        spikemasks[:,channels] = spikes_data.masks[:, 0:fetdim*nchannels:fetdim]

    cluster_colors = pandaize(cluster_colors, clusters)

    data = dict(
        freq=freq,
        trace=rawdata,
        spiketimes=spiketimes,
        spikemasks=spikemasks,
        spikeclusters=spikeclusters,
        cluster_colors = cluster_colors,
        s_before = s_before,
        s_after = s_after
    )
    return data

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
