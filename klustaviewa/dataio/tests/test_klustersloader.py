"""Unit tests for loader module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
from collections import Counter

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil
from nose.tools import with_setup

from kwiklib.dataio.tests.mock_data import (
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from kwiklib.dataio import (KlustersLoader, read_clusters, save_clusters,
    find_filename, find_indices, filename_to_triplet, triplet_to_filename,
    read_cluster_info, save_cluster_info, read_group_info, save_group_info,
    renumber_clusters, reorder, convert_to_clu, select, get_indices,
    check_dtype, check_shape, get_array, load_text)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_find_filename():
    dir = '/my/path/'
    extension_requested = 'spk'
    files = [
        'blabla.aclu.1',
        'blabla_test.aclu.1',
        'blabla_test2.aclu.1',
        'blabla_test3.aclu.3',
        'blabla.spk.1',
        'blabla_test.spk.1',
        'blabla_test.spk.1',
        ]
    spkfile = find_filename('/my/path/blabla.clu.1', extension_requested,
        files=files, dir=dir)
    assert spkfile == dir + 'blabla.spk.1'
        
    spkfile = find_filename('/my/path/blabla_test.clu.1', extension_requested,
        files=files, dir=dir)
    assert spkfile == dir + 'blabla_test.spk.1'
        
    spkfile = find_filename('/my/path/blabla_test2.clu.1', extension_requested,
        files=files, dir=dir)
    assert spkfile == dir + 'blabla_test.spk.1'
        
    spkfile = find_filename('/my/path/blabla_test3.clu.1', extension_requested,
        files=files, dir=dir)
    assert spkfile == dir + 'blabla_test.spk.1'
        
    spkfile = find_filename('/my/path/blabla_test3.clu.3', extension_requested,
        files=files, dir=dir)
    assert spkfile == None
    
def test_find_filename2():
    dir = '/my/path/'
    extension_requested = 'spk'
    files = [
        'blabla.aclu.2',
        'blabla_test.aclu.2',
        'blabla_test2.aclu.2',
        'blabla_test3.aclu.3',
        'blabla.spk.2',
        'blabla_test.spk.2',
        ]    
    spkfile = find_filename('/my/path/blabla_test.xml', extension_requested,
        files=files, dir=dir)
    
    assert spkfile == dir + 'blabla_test.spk.2'

def test_find_indices():
    dir = '/my/path/'
    files = [
        'blabla.aclu.2',
        'blabla_test.aclu.2',
        'blabla_test.spk.4',
        'blabla_test2.aclu.2',
        'blabla.aclu.9',
        'blabla_test3.aclu.3',
        'blabla.spk.2',
        'blabla_test.spk.2',
        ]    
    indices = find_indices('/my/path/blabla_test.xml', 
        files=files, dir=dir)
    
    assert indices == [2, 4]

def test_triplets():
    filename = 'my/path/blabla.aclu.2'
    triplet = filename_to_triplet(filename)
    filename2 = triplet_to_filename(triplet)
    
    assert filename == filename2
    assert triplet_to_filename(triplet[:2] + ('34',)) == \
        'my/path/blabla.aclu.34'
    
def test_clusters():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    clufile = os.path.join(dir, 'test.aclu.1')
    clufile2 = os.path.join(dir, 'test.aclu.1.saved')
    clusters = read_clusters(clufile)
    
    assert clusters.dtype == np.int32
    assert clusters.shape == (1000,)
    
    # Save.
    save_clusters(clufile2, clusters)
    
    # Open again.
    clusters2 = read_clusters(clufile2)
    
    assert np.array_equal(clusters, clusters2)
    
    # Check the headers.
    clusters_with_header = load_text(clufile, np.int32, skiprows=0)
    clusters2_with_header = load_text(clufile2, np.int32, skiprows=0)
    
    assert np.array_equal(clusters_with_header, clusters2_with_header)
    
def test_reorder():
    # Generate clusters and permutations.
    clusters = np.random.randint(size=1000, low=10, high=100)
    clusters_unique = np.unique(clusters)
    permutation = clusters_unique[np.random.permutation(len(clusters_unique))]
    
    # Reorder.
    clusters_reordered = reorder(clusters, permutation)
    
    # Check.
    i = len(clusters_unique) // 2
    c = clusters_unique[i]
    i_new = np.nonzero(permutation == c)[0][0]
    
    my_clusters = clusters == c
    
    assert np.all(clusters_reordered[my_clusters] == i_new)
    
def test_renumber_clusters():
    # Create clusters.
    clusters = np.random.randint(size=20, low=10, high=100)
    clusters_unique = np.unique(clusters)
    n = len(clusters_unique)
    
    # Create cluster info.
    cluster_info = np.zeros((n, 3), dtype=np.int32)
    cluster_info[:, 0] = clusters_unique
    cluster_info[:, 1] = np.mod(np.arange(n, dtype=np.int32), 35) + 1
    
    # Set groups.
    k = n // 3
    cluster_info[:k, 2] = 1
    cluster_info[k:2 * n // 3, 2] = 0
    cluster_info[2 * k:, 2] = 2
    cluster_info[n // 2, 2] = 1
    
    
    cluster_info = pd.DataFrame({
        'color': cluster_info[:, 1],
        'group': cluster_info[:, 2]},
        dtype=np.int32, index=cluster_info[:, 0])
    
    # Renumber
    clusters_renumbered, cluster_info_renumbered = renumber_clusters(clusters,
        cluster_info)
        
    # Test.
    c0 = clusters_unique[k]  # group 0
    c1 = clusters_unique[0]  # group 1
    c2 = clusters_unique[2 * k]  # group 2
    cm = clusters_unique[n // 2]  # group 1
    
    c0next = clusters_unique[k + 1]
    c1next = clusters_unique[0 + 1]
    c2next = clusters_unique[2 * k + 1]
    
    # New order:
    # c0 ... cm-1, cm+1, ..., c2-1, c1, ..., c0-1, cm, c2, ...
    
    assert np.array_equal(clusters == c0, clusters_renumbered == 0 + 2)
    assert np.array_equal(clusters == c0next, 
        clusters_renumbered == 1 + 2)
    assert np.array_equal(clusters == c1, clusters_renumbered == k - 1 + 2)
    assert np.array_equal(clusters == c1next, 
        clusters_renumbered == k + 2)
    assert np.array_equal(clusters == c2, clusters_renumbered == 2 * k + 2)
    assert np.array_equal(clusters == c2next, 
        clusters_renumbered == 2 * k + 1 + 2)
    
    assert np.array_equal(get_indices(cluster_info_renumbered),
        np.arange(n) + 2)
    
    # Increasing groups with the new numbering.
    assert np.all(np.diff(get_array(cluster_info_renumbered)[:,1]) >= 0)
    
    assert np.all(select(cluster_info_renumbered, 0 + 2) == 
        select(cluster_info, c0))
    assert np.all(select(cluster_info_renumbered, 1 + 2) == 
        select(cluster_info, c0next))
    assert np.all(select(cluster_info_renumbered, k - 1 + 2) == 
        select(cluster_info, c1))
    assert np.all(select(cluster_info_renumbered, k + 2) == 
        select(cluster_info, c1next))
    assert np.all(select(cluster_info_renumbered, 2 * k + 2) == 
        select(cluster_info, c2))
    assert np.all(select(cluster_info_renumbered, 2 * k + 1 + 2) == 
        select(cluster_info, c2next))
    
def test_convert_to_clu():
    clusters = np.random.randint(size=1000, low=10, high=100)
    clusters0 = clusters == 10
    clusters1 = clusters == 20
    clusters[clusters0] = 2
    clusters[clusters1] = 3
    clusters_unique = np.unique(clusters)
    n = len(clusters_unique)
    
    cluster_groups = np.random.randint(size=n, low=0, high=4)
    noise = np.in1d(clusters, clusters_unique[np.nonzero(cluster_groups == 0)[0]])
    mua = np.in1d(clusters, clusters_unique[np.nonzero(cluster_groups == 1)[0]])
    cluster_info = pd.DataFrame({'group': cluster_groups,
        'color': np.zeros(n, dtype=np.int32)}, 
        index=clusters_unique,
        dtype=np.int32)
    
    clusters_new = convert_to_clu(clusters, cluster_info['group'])
    
    assert np.array_equal(clusters_new == 0, noise)
    assert np.array_equal(clusters_new == 1, mua)
    
def test_cluster_info():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    clufile = os.path.join(dir, 'test.aclu.1')
    cluinfofile = os.path.join(dir, 'test.acluinfo.1')

    clusters = read_clusters(clufile)
    
    indices = np.unique(clusters)
    colors = np.random.randint(low=0, high=10, size=len(indices))
    groups = np.random.randint(low=0, high=2, size=len(indices))
    cluster_info = pd.DataFrame({'color': pd.Series(colors, index=indices),
        'group': pd.Series(groups, index=indices)})
    
    save_cluster_info(cluinfofile, cluster_info)
    cluster_info2 = read_cluster_info(cluinfofile)
    
    assert np.array_equal(cluster_info.values, cluster_info2.values)
    
def test_group_info():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    groupinfofile = os.path.join(dir, 'test.groups.1')

    group_info = np.zeros((4, 2), dtype=object)
    group_info[:,0] = (np.arange(4) + 1)
    group_info[:,1] = np.array(['Noise', 'MUA', 'Good', 'Unsorted'],
        dtype=object)
    group_info = pd.DataFrame(group_info)
    
    save_group_info(groupinfofile, group_info)
    group_info2 = read_group_info(groupinfofile)
    
    assert np.array_equal(group_info.values, group_info2.values)
    
def test_klusters_loader_1():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(filename=xmlfile)
    
    # Get full data sets.
    features = l.get_features()
    # features_some = l.get_some_features()
    masks = l.get_masks()
    waveforms = l.get_waveforms()
    clusters = l.get_clusters()
    spiketimes = l.get_spiketimes()
    nclusters = len(Counter(clusters))
    
    probe = l.get_probe()
    cluster_colors = l.get_cluster_colors()
    cluster_groups = l.get_cluster_groups()
    group_colors = l.get_group_colors()
    group_names = l.get_group_names()
    cluster_sizes = l.get_cluster_sizes()
    
    # Check the shape of the data sets.
    # ---------------------------------
    assert check_shape(features, (nspikes, nchannels * fetdim + 1))
    # assert features_some.shape[1] == nchannels * fetdim + 1
    assert check_shape(masks, (nspikes, nchannels))
    assert check_shape(waveforms, (nspikes, nsamples, nchannels))
    assert check_shape(clusters, (nspikes,))
    assert check_shape(spiketimes, (nspikes,))
    
    assert check_shape(probe, (nchannels, 2))
    assert check_shape(cluster_colors, (nclusters,))
    assert check_shape(cluster_groups, (nclusters,))
    assert check_shape(group_colors, (4,))
    assert check_shape(group_names, (4,))
    assert check_shape(cluster_sizes, (nclusters,))
    
    
    # Check the data type of the data sets.
    # -------------------------------------
    assert check_dtype(features, np.float32)
    assert check_dtype(masks, np.float32)
    # HACK: Panel has no dtype(s) attribute
    # assert check_dtype(waveforms, np.float32)
    assert check_dtype(clusters, np.int32)
    assert check_dtype(spiketimes, np.float32)
    
    assert check_dtype(probe, np.float32)
    assert check_dtype(cluster_colors, np.int32)
    assert check_dtype(cluster_groups, np.int32)
    assert check_dtype(group_colors, np.int32)
    assert check_dtype(group_names, object)
    assert check_dtype(cluster_sizes, np.int32)
    
    l.close()
    
def test_klusters_loader_2():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(filename=xmlfile)
    
    # Get full data sets.
    features = l.get_features()
    masks = l.get_masks()
    waveforms = l.get_waveforms()
    clusters = l.get_clusters()
    spiketimes = l.get_spiketimes()
    nclusters = len(Counter(clusters))
    
    probe = l.get_probe()
    cluster_colors = l.get_cluster_colors()
    cluster_groups = l.get_cluster_groups()
    group_colors = l.get_group_colors()
    group_names = l.get_group_names()
    cluster_sizes = l.get_cluster_sizes()
    
    
    # Check selection.
    # ----------------
    index = nspikes / 2
    waveform = select(waveforms, index)
    cluster = clusters[index]
    spikes_in_cluster = np.nonzero(clusters == cluster)[0]
    nspikes_in_cluster = len(spikes_in_cluster)
    l.select(clusters=[cluster])
    
    
    # Check the size of the selected data.
    # ------------------------------------
    assert check_shape(l.get_features(), (nspikes_in_cluster, 
                                          nchannels * fetdim + 1))
    assert check_shape(l.get_masks(full=True), (nspikes_in_cluster, 
                                                nchannels * fetdim + 1))
    assert check_shape(l.get_waveforms(), 
                       (nspikes_in_cluster, nsamples, nchannels))
    assert check_shape(l.get_clusters(), (nspikes_in_cluster,))
    assert check_shape(l.get_spiketimes(), (nspikes_in_cluster,))
    
    
    # Check waveform sub selection.
    # -----------------------------
    waveforms_selected = l.get_waveforms()
    assert np.array_equal(get_array(select(waveforms_selected, index)), 
        get_array(waveform))
        
    l.close()
        
def test_klusters_loader_control():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(filename=xmlfile)
    
    # Take all spikes in cluster 3.
    spikes = get_indices(l.get_clusters(clusters=3))
    
    # Put them in cluster 4.
    l.set_cluster(spikes, 4)
    spikes_new = get_indices(l.get_clusters(clusters=4))
    
    # Ensure all spikes in old cluster 3 are now in cluster 4.
    assert np.all(np.in1d(spikes, spikes_new))
    
    # Change cluster groups.
    clusters = [2, 3, 4]
    group = 0
    l.set_cluster_groups(clusters, group)
    groups = l.get_cluster_groups(clusters)
    assert np.all(groups == group)
    
    # Change cluster colors.
    clusters = [2, 3, 4]
    color = 12
    l.set_cluster_colors(clusters, color)
    colors = l.get_cluster_colors(clusters)
    assert np.all(colors == color)
    
    # Change group name.
    group = 0
    name = l.get_group_names(group)
    name_new = 'Noise new'
    assert name == 'Noise'
    l.set_group_names(group, name_new)
    assert l.get_group_names(group) == name_new
    
    # Change group color.
    groups = [1, 2]
    colors = l.get_group_colors(groups)
    color_new = 10
    l.set_group_colors(groups, color_new)
    assert np.all(l.get_group_colors(groups) == color_new)
    
    # Add cluster and group.
    spikes = get_indices(l.get_clusters(clusters=3))[:10]
    # Create new group 100.
    l.add_group(100, 'New group', 10)
    # Create new cluster 10000 and put it in group 100.
    l.add_cluster(10000, 100, 10)
    # Put some spikes in the new cluster.
    l.set_cluster(spikes, 10000)
    clusters = l.get_clusters(spikes=spikes)
    assert np.all(clusters == 10000)
    groups = l.get_cluster_groups(10000)
    assert groups == 100
    l.set_cluster(spikes, 2)
    
    # Remove the new cluster and group.
    l.remove_cluster(10000)
    l.remove_group(100)
    assert np.all(~np.in1d(10000, l.get_clusters()))
    assert np.all(~np.in1d(100, l.get_cluster_groups()))
    
    l.close()
    
def test_klusters_save():
    """WARNING: this test should occur at the end of the module since it
    changes the mock data sets."""
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(filename=xmlfile)
    
    clusters = l.get_clusters()
    cluster_colors = l.get_cluster_colors()
    cluster_groups = l.get_cluster_groups()
    group_colors = l.get_group_colors()
    group_names = l.get_group_names()
    
    # Set clusters.
    indices = get_indices(clusters)
    l.set_cluster(indices[::2], 2)
    l.set_cluster(indices[1::2], 3)
    
    # Set cluster info.
    cluster_indices = l.get_clusters_unique()
    l.set_cluster_colors(cluster_indices[::2], 10)
    l.set_cluster_colors(cluster_indices[1::2], 20)
    l.set_cluster_groups(cluster_indices[::2], 1)
    l.set_cluster_groups(cluster_indices[1::2], 0)
    
    # Save.
    l.remove_empty_clusters()
    l.save()
    
    clusters = read_clusters(l.filename_aclu)
    cluster_info = read_cluster_info(l.filename_acluinfo)
    
    assert np.all(clusters[::2] == 2)
    assert np.all(clusters[1::2] == 3)
    
    assert np.array_equal(cluster_info.index, cluster_indices)
    assert np.all(cluster_info.values[::2, 0] == 10)
    assert np.all(cluster_info.values[1::2, 0] == 20)
    assert np.all(cluster_info.values[::2, 1] == 1)
    assert np.all(cluster_info.values[1::2, 1] == 0)

    l.close()
    
    
    