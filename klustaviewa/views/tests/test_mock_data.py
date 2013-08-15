"""Unit tests for mock data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from mock_data import *
from kwiklib.dataio import MemoryLoader


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_mock_data():
        
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
    
    loader = MemoryLoader(
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
    )