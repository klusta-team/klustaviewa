"""This module provides utility classes and functions to load spike sorting
data sets."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import os.path
import re
from collections import Counter

import numpy as np
import pandas as pd
from galry import QtGui, QtCore

from loader import (Loader, default_group_info, reorder, renumber_clusters,
    default_cluster_info)
from tools import (load_text, load_xml, normalize,
    load_binary, load_pickle, save_text, get_array,
    first_row, load_binary_memmap)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters, get_some_spikes, get_indices)
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.utils.settings import SETTINGS
from klustaviewa.utils.logger import (debug, info, warn, exception, FileLogger,
    register, unregister)
from klustaviewa.utils.colors import COLORS_COUNT, generate_colors


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def find_index(filename):
    """Search the file index of the filename, if any, or return None."""
    r = re.search(r"([^\n]+)\.([^\.]+)\.([0-9]+)$", filename)
    if r:
        return int(r.group(3))
    # If the filename has no index in it, and if the file does not actually
    # exist, return the index of an existing filename.
    # if not os.path.exists(filename):
    return find_index(find_filename(filename, 'fet'))

def find_indices(filename, dir='', files=[]):
    """Return the list of all indices for the given filename, present
    in the filename's directory."""
    # get the extension-free filename, extension, and file index
    # template: FILENAME.xxx.0  => FILENAME (can contain points), 0 (index)
    # try different patterns
    patterns = [r"([^\n]+)\.([^\.]+)\.([0-9]+)$",
                r"([^\n]+)\.([^\.]+)$"]
    for pattern in patterns:
        r = re.search(pattern, filename)
        if r:
            filename = r.group(1)
            # extension = r.group(2)
            break
    
    # get the full path
    if not dir:
        dir = os.path.dirname(filename)
    filename = os.path.basename(filename)
    # try obtaining the list of all files in the directory
    if not files:
        try:
            files = os.listdir(dir)
        except (WindowsError, OSError, IOError):
            raise IOError("Error when accessing '{0:s}'.".format(dir))
    
    # If the requested filename does not have a file index, then get the 
    # smallest available fileindex in the files list.
    fileindex_set = set()
    for file in files:
        r = re.search(r"([^\n]+)\.([^\.]+)\.([0-9]+)$", file)
        if r:
            if r.group(1) == filename:
                fileindex_set.add(int(r.group(3)))
        
    return sorted(fileindex_set)
          
def find_filename(filename, extension_requested, dir='', files=[]):
    """Search the most plausible existing filename corresponding to the
    requested approximate filename, which has the required file index and
    extension.
    
    Arguments:
    
      * filename: the full filename of an existing file in a given dataset
      * extension_requested: the extension of the file that is requested
    
    """
    
    # get the extension-free filename, extension, and file index
    # template: FILENAME.xxx.0  => FILENAME (can contain points), 0 (index)
    # try different patterns
    patterns = [r"([^\n]+)\.([^\.]+)\.([0-9]+)$",
                r"([^\n]+)\.([^\.]+)$"]
    fileindex = None
    for pattern in patterns:
        r = re.search(pattern, filename)
        if r:
            filename = r.group(1)
            extension = r.group(2)
            if len(r.groups()) >= 3:
                fileindex = int(r.group(3))
            # else:
                # fileindex = None
            break
    
    # get the full path
    if not dir:
        dir = os.path.dirname(filename)
    filename = os.path.basename(filename)
    # try obtaining the list of all files in the directory
    if not files:
        try:
            files = os.listdir(dir)
        except (WindowsError, OSError, IOError):
            raise IOError("Error when accessing '{0:s}'.".format(dir))
    
    # If the requested filename does not have a file index, then get the 
    # smallest available fileindex in the files list.
    if fileindex is None:
        fileindex_set = set()
        for file in files:
            r = re.search(r"([^\n]+)\.([^\.]+)\.([0-9]+)$", file)
            if r:
                fileindex_set.add(int(r.group(3)))
        if fileindex_set:
            fileindex = sorted(fileindex_set)[0]
    
    # try different suffixes
    if fileindex is not None:
        suffixes = [
                    '.{0:s}.{1:d}'.format(extension_requested, fileindex),
                    '.{0:s}'.format(extension_requested),
                    ]
    else:
        suffixes = [
                    # '.{0:s}.{1:d}'.format(extension_requested, fileindex),
                    '.{0:s}'.format(extension_requested),
                    ]
    
    # find the real filename with the longest path that fits the requested
    # filename
    for suffix in suffixes:
        filtered = []
        prefix = filename
        while prefix and not filtered:
            filtered = filter(lambda file: (file.startswith(prefix) and 
                file.endswith(suffix)), files)
            prefix = prefix[:-1]
        # order by increasing length and return the shortest
        filtered = sorted(filtered, cmp=lambda k, v: len(k) - len(v))
        if filtered:
            return os.path.join(dir, filtered[0])
    
    return None

def find_any_filename(filename, extension_requested, dir='', files=[]):
    # get the full path
    if not dir:
        dir = os.path.dirname(filename)
    
    # try obtaining the list of all files in the directory
    if not files:
        try:
            files = os.listdir(dir)
        except (WindowsError, OSError, IOError):
            raise IOError("Error when accessing '{0:s}'.".format(dir))
    
    filtered = filter(lambda f: f.endswith('.' + extension_requested), files)
    if filtered:
        return os.path.join(dir, filtered[0])
    
def find_filename_or_new(filename, extension_requested,
        have_file_index=True, dir='', files=[]):
    """Find an existing filename with a requested extension, or create
    a new filename based on an existing file."""
    # Find the filename with the requested extension.
    filename_found = find_filename(filename, extension_requested, dir=dir, files=files)
    # If it does not exist, find a file that exists, and replace the extension 
    # with the requested one.
    if not filename_found:
        if have_file_index:
            filename_existing = find_filename(filename, 'fet', dir=dir, files=files)
            filename_new = filename_existing.replace('.fet.', '.{0:s}.'.format(
                extension_requested))
        else:
            filename_existing = find_filename(filename, 'xml', dir=dir, files=files)
            filename_new = filename_existing.replace('.xml', 
                '.' + extension_requested)
        return filename_new
    else:
        return filename_found
    
def find_filenames(filename):
    """Find the filenames of the different files for the current
    dataset."""
    filenames = {}
    for ext in ['xml', 'fet', 'spk', 'uspk', 'res', 'dat',]:
        filenames[ext] = find_filename(filename, ext) or ''
    for ext in ['clu', 'aclu', 'acluinfo', 'groupinfo',]:
        filenames[ext] = find_filename_or_new(filename, ext)
    filenames['probe'] = (find_filename(filename, 'probe') or
                          find_any_filename(filename, 'probe'))
    filenames['mask'] = (find_filename(filename, 'fmask') or
                         find_filename(filename, 'mask'))
    # HDF5 file format
    filenames.update(find_hdf5_filenames(filename))
    return filenames

def filename_to_triplet(filename):
    patterns = [r"([^\n]+)\.([^\.]+)\.([0-9]+)$",
                r"([^\n]+)\.([^\.]+)$"]
    fileindex = None
    for pattern in patterns:
        r = re.search(pattern, filename)
        if r:
            filename = r.group(1)
            extension = r.group(2)
            if len(r.groups()) >= 3:
                fileindex = int(r.group(3))
            return (filename, extension, fileindex)
    return (filename, )
    
def triplet_to_filename(triplet):
    return '.'.join(map(str, triplet))
    
def find_hdf5_filenames(filename):
    filenames = {}
    for key in ['main', 'wave', 'raw', 'low', 'high']:
        filenames['hdf5_' + key] = os.path.abspath(
            find_filename_or_new(filename, key + '.h5', have_file_index=False))
    return filenames

# -----------------------------------------------------------------------------
# File reading functions
# -----------------------------------------------------------------------------
def read_xml(filename_xml, fileindex):
    """Read the XML file associated to the current dataset,
    and return a metadata dictionary."""
    
    params = load_xml(filename_xml, fileindex=fileindex)
    
    # klusters tests
    metadata = dict(
        nchannels=params['nchannels'],
        nsamples=params['nsamples'],
        fetdim=params['fetdim'],
        freq=params['rate'])
    
    return metadata

# Features.
def process_features(features, fetdim, nchannels, freq, nfet=None):
    features = np.array(features, dtype=np.float32)
    nspikes, ncol = features.shape
    if nfet is not None:
        nextrafet = nfet - fetdim * nchannels
    else:
        nextrafet = ncol - fetdim * nchannels
            
    # get the spiketimes
    spiketimes = features[:,-1].copy()
    spiketimes *= (1. / freq)
    
    # normalize normal features while keeping symmetry
    features_normal = normalize(features[:,:fetdim * nchannels],
                                        symmetric=True)
    features_time = spiketimes.reshape((-1, 1)) * 1. / spiketimes[-1] * 2 - 1
    # normalize extra features without keeping symmetry
    if nextrafet > 1:
        features_extra = normalize(features[:,-nextrafet:-1],
                                            symmetric=False)
        features = np.hstack((features_normal, features_extra, features_time))
    else:
        features = np.hstack((features_normal, features_time))
    return features, spiketimes
    
def read_features(filename_fet, nchannels, fetdim, freq, do_process=True):
    """Read a .fet file and return the normalize features array,
    as well as the spiketimes."""
    try:
        features = load_text(filename_fet, np.int64, skiprows=1, delimiter=' ')
    except ValueError:
        features = load_text(filename_fet, np.float32, skiprows=1, delimiter='\t')
    if do_process:
        return process_features(features, fetdim, nchannels, freq, 
            nfet=first_row(filename_fet))
    else:
        return features
    
# Clusters.
def process_clusters(clusters):
    return clusters[1:]

def read_clusters(filename_clu):
    clusters = load_text(filename_clu, np.int32)
    return process_clusters(clusters)

# RES file.
def process_res(spiketimes, freq=None):
    if freq is None:
        return spiketimes
    else:
        return spiketimes * 1. / freq

def read_res(filename_res, freq=None):
    res = load_text(filename_res, np.int32)
    return process_res(res, freq)

# Cluster info.
def process_cluster_info(cluster_info):
    cluster_info = pd.DataFrame({'color': cluster_info[:, 1], 
        'group': cluster_info[:, 2]}, dtype=np.int32, index=cluster_info[:, 0])
    return cluster_info
    
def read_cluster_info(filename_acluinfo):
    # For each cluster (absolute indexing): cluster index, color index, 
    # and group index
    cluster_info = load_text(filename_acluinfo, np.int32)
    return process_cluster_info(cluster_info)
    
# Group info.
def process_group_info(group_info):
    group_info = pd.DataFrame(
        {'color': group_info[:, 1].astype(np.int32),
         'name': group_info[:, 2]}, index=group_info[:, 0].astype(np.int32))
    return group_info

def read_group_info(filename_groupinfo):
    # For each group (absolute indexing): color index, and name
    group_info = load_text(filename_groupinfo, str, delimiter='\t')
    return process_group_info(group_info)
    
# Masks.
def process_masks(masks_full, fetdim):
    masks = masks_full[:,:-1:fetdim]
    return masks, masks_full

def read_masks(filename_mask, fetdim):
    masks_full = load_text(filename_mask, np.float32, skiprows=1)
    return process_masks(masks_full, fetdim)
    
# Waveforms.
def process_waveforms(waveforms, nsamples, nchannels):
    waveforms = np.array(waveforms, dtype=np.float32)
    waveforms = normalize(waveforms, symmetric=True)
    waveforms = waveforms.reshape((-1, nsamples, nchannels))
    return waveforms

def read_waveforms(filename_spk, nsamples, nchannels):
    waveforms = np.array(load_binary(filename_spk), dtype=np.float32)
    n = waveforms.size
    if n % nsamples != 0 or n % nchannels != 0:
        waveforms = load_text(filename_spk, np.float32)
    return process_waveforms(waveforms, nsamples, nchannels)
    
# DAT.
def read_dat(filename_dat, nchannels):
    nsamples = os.path.getsize(filename_dat) // nchannels
    return load_binary_memmap(filename_dat, dtype=np.int16,
                             shape=(nsamples, nchannels))

# Probe.
def process_probe(probe):
    return normalize(probe)

def read_probe(filename_probe):
    if not filename_probe:
        return
    if os.path.exists(filename_probe):
        # Try the text-flavored probe file.
        try:
            probe = load_text(filename_probe, np.float32)
        except:
            # Or try the Python-flavored probe file (SpikeDetekt, with an
            # extra field 'geometry').
            try:
                ns = {}
                execfile(filename_probe, ns)
                probe = ns['geometry']
                probe = np.array([probe[i] for i in sorted(probe.keys())],
                                    dtype=np.float32)
            except:
                return None
        return process_probe(probe)


# -----------------------------------------------------------------------------
# File saving functions
# -----------------------------------------------------------------------------
def save_cluster_info(filename_cluinfo, cluster_info):
    cluster_info_array = np.hstack((cluster_info.index.reshape((-1, 1)), 
        cluster_info.values))
    save_text(filename_cluinfo, cluster_info_array)
    
def save_group_info(filename_groupinfo, group_info):
    group_info_array = np.hstack((group_info.index.reshape((-1, 1)), 
        group_info.values))
    save_text(filename_groupinfo, group_info_array, fmt='%s', delimiter='\t')
    
def save_clusters(filename_clu, clusters):
    save_text(filename_clu, clusters, header=len(np.unique(clusters)))

def convert_to_clu(clusters, cluster_info):
    cluster_groups = cluster_info['group']
    clusters_new = np.array(clusters, dtype=np.int32)
    for i in (0, 1):
        clusters_new[cluster_groups.ix[clusters] == i] = i
    # clusters_unique = np.unique(set(clusters_new).union(set([0, 1])))
    # clusters_renumbered = reorder(clusters_new, clusters_unique)
    # return clusters_renumbered
    return clusters_new

    
# -----------------------------------------------------------------------------
# Klusters Loader
# -----------------------------------------------------------------------------
class KlustersLoader(Loader):
    def open(self, filename):
        """Open a file."""
        self.filename = filename
        # Find the file index associated to the filename, or 1 by default.
        self.fileindex = find_index(filename) or 1
        self.find_filenames()
        self.save_original_clufile()
        self.read()
        
    def find_filenames(self):
        # """Find the filenames of the different files for the current
        # dataset."""
        for ext, filename in find_filenames(self.filename).iteritems():
            setattr(self, 'filename_' + ext, filename)
        
    def save_original_clufile(self):
        filename_clu_original = find_filename(self.filename, 'clu_original')
        if filename_clu_original is None:
            if os.path.exists(self.filename_clu):
                # Save the original clu file if it does not exist yet.
                with open(self.filename_clu, 'r') as f:
                    clu = f.read()
                with open(self.filename_clu.replace('.clu.', 
                    '.clu_original.'), 'w') as f:
                    f.write(clu)
            if os.path.exists(self.filename_aclu):
                # Save the original clu file if it does not exist yet.
                with open(self.filename_aclu, 'r') as f:
                    clu = f.read()
                with open(self.filename_aclu.replace('.aclu.', 
                    '.aclu_original.'), 'w') as f:
                    f.write(clu)
    
    
    # Internal read methods.
    # ----------------------
    def read_metadata(self):
        try:
            self.metadata = read_xml(self.filename_xml, self.fileindex)
        except:
            # Die if no XML file is available for this dataset, as it contains
            # critical metadata.
            raise IOError("The XML file is missing.")
            
        self.nsamples = self.metadata.get('nsamples')
        self.nchannels = self.metadata.get('nchannels')
        self.fetdim = self.metadata.get('fetdim')
        self.freq = self.metadata.get('freq')
        
    def read_probe(self):
        if self.filename_probe is None:
            info("No probe file has been found.")
            self.probe = None
        else:
            try:
                self.probe = read_probe(self.filename_probe)
                info("Successfully loaded {0:s}".format(self.filename_probe))
            except Exception as e:
                info(("There was an error while loading the probe file "
                          "'{0:s}' : {1:s}").format(self.filename_probe,
                            e.message))
                self.probe = None
    
    def read_features(self):
        try:
            self.features, self.spiketimes = read_features(self.filename_fet,
                self.nchannels, self.fetdim, self.freq)
            info("Successfully loaded {0:s}".format(self.filename_fet))
        except IOError:
            raise IOError("The FET file is missing.")
        # Convert to Pandas.
        self.features = pd.DataFrame(self.features, dtype=np.float32)
        self.duration = self.spiketimes[-1]
        self.spiketimes = pd.Series(self.spiketimes, dtype=np.float32)
        
        # Count the number of spikes and save it in the metadata.
        self.nspikes = self.features.shape[0]
        self.metadata['nspikes'] = self.nspikes
        self.nextrafet = self.features.shape[1] - self.nchannels * self.fetdim
    
    def read_res(self):
        try:
            self.spiketimes_res = read_res(self.filename_res, self.freq)
            self.spiketimes_res = pd.Series(self.spiketimes_res, dtype=np.float32)
        except IOError:
            warn("The RES file is missing.")
    
    def read_clusters(self):
        try:
            # Try reading the ACLU file, or fallback on the CLU file.
            if os.path.exists(self.filename_aclu):
                self.clusters = read_clusters(self.filename_aclu)
                info("Successfully loaded {0:s}".format(self.filename_aclu))
            else:
                self.clusters = read_clusters(self.filename_clu)
                info("Successfully loaded {0:s}".format(self.filename_clu))
        except IOError:
            warn("The CLU file is missing.")
            # Default clusters if the CLU file is not available.
            self.clusters = np.zeros(self.nspikes, dtype=np.int32)
        # Convert to Pandas.
        self.clusters = pd.Series(self.clusters, dtype=np.int32)
        
        # Count clusters.
        self._update_data()
    
    def read_cluster_info(self):
        try:
            self.cluster_info = read_cluster_info(self.filename_acluinfo)
            info("Successfully loaded {0:s}".format(self.filename_acluinfo))
        except IOError:
            info("The CLUINFO file is missing, generating a default one.")
            self.cluster_info = default_cluster_info(self.clusters_unique)
                
        if not np.array_equal(self.cluster_info.index, self.clusters_unique):
            info("The CLUINFO file does not correspond to the loaded CLU file.")
            self.cluster_info = default_cluster_info(self.clusters_unique)
            
        self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)
        
    def read_group_info(self):
        try:
            self.group_info = read_group_info(self.filename_groupinfo)
            info("Successfully loaded {0:s}".format(self.filename_groupinfo))
        except IOError:
            info("The GROUPINFO file is missing, generating a default one.")
            self.group_info = default_group_info()
        
        # Convert to Pandas.
        self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']
        
    def read_masks(self):
        try:
            self.masks, self.masks_full = read_masks(self.filename_mask,
                                                     self.fetdim)
            info("Successfully loaded {0:s}".format(self.filename_mask))
        except IOError:
            warn("The MASKS/FMASKS file is missing.")
            # Default masks if the MASK/FMASK file is not available.
            self.masks = np.ones((self.nspikes, self.nchannels))
            self.masks_full = np.ones(self.features.shape)
        self.masks = pd.DataFrame(self.masks)
        self.masks_full = pd.DataFrame(self.masks_full)
    
    def read_waveforms(self):
        try:
            self.waveforms = read_waveforms(self.filename_spk, self.nsamples,
                                            self.nchannels)
            info("Successfully loaded {0:s}".format(self.filename_spk))
        except IOError:
            warn("The SPK file is missing.")
            self.waveforms = np.zeros((self.nspikes, self.nsamples, 
                self.nchannels))
        # Convert to Pandas.
        self.waveforms = pd.Panel(self.waveforms, dtype=np.float32)
    
    def read_dat(self):
        try:
            self.dat = read_dat(self.filename_dat, self.nchannels)
        except IOError:
            warn("The DAT file is missing.")
    
    def read_fil(self):
        try:
            self.fil = read_dat(self.filename_fil, self.nchannels)
        except IOError:
            warn("The FIL file is missing.")
    
    def read_stats(self):
        self.ncorrbins = SETTINGS.get('correlograms.ncorrbins', 100)
        self.corrbin = SETTINGS.get('correlograms.corrbin', .001)

        
    # Log file.
    # ---------
    def initialize_logfile(self):
        filename = self.filename_fet.replace('.fet.', '.kvwlg.')
        self.logfile = FileLogger(filename, name='datafile', 
            level=USERPREF['loglevel_file'])
        # Register log file.
        register(self.logfile)
        
    
    # Public methods.
    # ---------------
    def read(self):
        self.initialize_logfile()
        # Load the similarity measure chosen by the user in the preferences
        # file: 'gaussian' or 'kl'.
        # Refresh the preferences file when a new file is opened.
        USERPREF.refresh()
        self.similarity_measure = USERPREF['similarity_measure'] or 'gaussian'
        info("Similarity measure: {0:s}.".format(self.similarity_measure))
        info("Opening {0:s}.".format(self.filename))
        self.report_progress(0, 5)
        self.read_metadata()
        self.read_probe()
        self.report_progress(1, 5)
        self.read_features()
        self.report_progress(2, 5)
        self.read_res()
        self.read_clusters()
        self.report_progress(3, 5)
        self.read_cluster_info()
        self.read_group_info()
        self.read_masks()
        self.report_progress(4, 5)
        self.read_waveforms()
        self.report_progress(5, 5)
        self.read_stats()
    
    def save(self, renumber=False):
        self.update_cluster_info()
        self.update_group_info()
        
        if renumber:
            self.renumber()
            clusters = get_array(self.clusters_renumbered)
            cluster_info = self.cluster_info_renumbered
        else:
            clusters = get_array(self.clusters)
            cluster_info = self.cluster_info
        
        # Save both ACLU and CLU files.
        save_clusters(self.filename_aclu, clusters)
        save_clusters(self.filename_clu, 
            convert_to_clu(clusters, cluster_info))
        
        # Save CLUINFO and GROUPINFO files.
        save_cluster_info(self.filename_acluinfo, cluster_info)
        save_group_info(self.filename_groupinfo, self.group_info)
    
    def close(self):
        if hasattr(self, 'logfile'):
            unregister(self.logfile)
            
    def __del__(self):
        self.close()
        
    
# -----------------------------------------------------------------------------
# Memory Loader
# -----------------------------------------------------------------------------
class MemoryLoader(Loader):
    def __init__(self, parent=None, **kwargs):
        super(MemoryLoader, self).__init__(parent)
        self.read(**kwargs)
    
    
    # Internal read methods.
    # ----------------------
    def read_metadata(self, nsamples=None, nchannels=None, fetdim=None,
        freq=None):
        self.nsamples = nsamples
        self.nchannels = nchannels
        self.fetdim = fetdim
        self.freq = freq
        
    def read_probe(self, probe):
        try:
            self.probe = process_probe(probe)
        except Exception as e:
            info(("There was an error while loading the probe: "
                      "'{0:s}'").format(e.message))
            self.probe = None
    
    def read_features(self, features):
        self.features, self.spiketimes = process_features(features,
            self.nchannels, self.fetdim, self.freq)
        # Convert to Pandas.
        self.features = pd.DataFrame(self.features, dtype=np.float32)
        self.duration = self.spiketimes[-1]
        self.spiketimes = pd.Series(self.spiketimes, dtype=np.float32)
        
        # Count the number of spikes and save it in the metadata.
        self.nspikes = self.features.shape[0]
        self.nextrafet = self.features.shape[1] - self.nchannels * self.fetdim
    
    def read_clusters(self, clusters):
        self.clusters = process_clusters(clusters)
        # Convert to Pandas.
        self.clusters = pd.Series(self.clusters, dtype=np.int32)
        # Count clusters.
        self._update_data()
    
    def read_cluster_info(self, cluster_info):
        self.cluster_info = process_cluster_info(cluster_info)
                
        assert np.array_equal(self.cluster_info.index, self.clusters_unique), \
            "The CLUINFO file does not correspond to the loaded CLU file."
            
        self.cluster_colors = self.cluster_info['color'].astype(np.int32)
        self.cluster_groups = self.cluster_info['group'].astype(np.int32)
        
    def read_group_info(self, group_info):
        self.group_info = process_group_info(group_info)
        # Convert to Pandas.
        self.group_colors = self.group_info['color'].astype(np.int32)
        self.group_names = self.group_info['name']
        
    def read_masks(self, masks):
        self.masks, self.masks_full = process_masks(masks, self.fetdim)
        self.masks = pd.DataFrame(self.masks)
        self.masks_full = pd.DataFrame(self.masks_full)
    
    def read_waveforms(self, waveforms):
        self.waveforms = process_waveforms(waveforms, self.nsamples,
                                        self.nchannels)
        # Convert to Pandas.
        self.waveforms = pd.Panel(self.waveforms, dtype=np.float32)
    
    def read_stats(self):
        self.ncorrbins = SETTINGS.get('correlograms.ncorrbins', 100)
        self.corrbin = SETTINGS.get('correlograms.corrbin', .001)
    
    
    # Public methods.
    # ---------------
    def read(self, nsamples=None, nchannels=None, fetdim=None,
            freq=None, probe=None, features=None, clusters=None,
            cluster_info=None, group_info=None, masks=None,
            waveforms=None):
        self.read_metadata(nsamples=nsamples, nchannels=nchannels,
            fetdim=fetdim, freq=freq)
        self.read_probe(probe)
        self.read_features(features)
        self.read_clusters(clusters)
        self.read_cluster_info(cluster_info)
        self.read_group_info(group_info)
        self.read_masks(masks)
        self.read_waveforms(waveforms)
        self.read_stats()
    
    
    