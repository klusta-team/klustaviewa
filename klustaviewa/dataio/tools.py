"""Utility functions for loading/saving files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os.path
import re
import cPickle
import xml.etree.ElementTree as ET

import numpy as np
# Try importing Pandas.
try:
    import pandas as pd
    # Make sure that read_csv is available.
    assert hasattr(pd, 'read_csv')
    HAS_PANDAS = True
except (ImportError, AssertionError):
    log_warn("You should install Pandas v>=0.8.")
    HAS_PANDAS = False


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def find_index(filename):
    """Search the file index of the filename, if any, or return None."""
    r = re.search(r"([^\n]+)\.([^\.]+)\.([0-9]+)$", filename)
    if r:
        return int(r.group(3))
    else:
        return None

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
    for pattern in patterns:
        r = re.search(pattern, filename)
        if r:
            filename = r.group(1)
            extension = r.group(2)
            if len(r.groups()) >= 3:
                fileindex = int(r.group(3))
            else:
                fileindex = 1
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
    
    # try different suffixes
    suffixes = ['.{0:s}'.format(extension_requested),
                '.{0:s}.{1:d}'.format(extension_requested, fileindex)]
    
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
    filename = os.path.basename(filename)
    
    # try obtaining the list of all files in the directory
    if not files:
        try:
            files = os.listdir(dir)
        except (WindowsError, OSError, IOError):
            raise IOError("Error when accessing '{0:s}'.".format(dir))
    
    filtered = filter(lambda f: f.endswith('.' + extension_requested), files)
    if filtered:
        return os.path.join(dir, filtered[0])
    
    
# -----------------------------------------------------------------------------
# Utility data functions
# -----------------------------------------------------------------------------
def check_dtype(data, dtype):
    if hasattr(data, 'dtype'):
        return data.dtype == dtype
    elif hasattr(data, 'dtypes'):
        return (data.dtypes == dtype).all()
        
def check_shape(data, shape):
    return tuple(data.shape) == shape

def get_array(data, copy=False):
    """Get a NumPy array from a NumPy array or a Pandas data object (Series,
    DataFrame or Panel)."""
    if type(data) == np.ndarray:
        if data.dtype == np.int64:
            return data.astype(np.int32)
        elif data.dtype == np.float64:
            return data.astype(np.float32)
        else:
            if copy:
                return data.copy()
            else:
                return data
    elif isinstance(data, (pd.DataFrame, pd.Panel)):
        return np.array(data.sort_index().values)
    elif isinstance(data, (pd.Int64Index, pd.Index)):
        return np.sort(data.values)
    else:
        return np.array(data.sort_index().values)
    

# -----------------------------------------------------------------------------
# Text files related functions
# -----------------------------------------------------------------------------
def load_text(filepath, dtype, skiprows=0):
    if not filepath:
        raise IOError("The filepath is empty.")
    return np.loadtxt(filepath, dtype=dtype, skiprows=skiprows)

def first_row(filepath):
    with open(filepath, 'r') as f:
        n = f.readline().strip().split('\t')[0]
    return int(n)

# Faster load_text version if Pandas is installed.
if HAS_PANDAS:
    def load_text(filepath, dtype, skiprows=0, delimiter=' '):
        if not filepath:
            raise IOError("The filepath is empty.")
        with open(filepath, 'r') as f:
            for _ in xrange(skiprows):
                f.readline()
            x = pd.read_csv(f, header=None, 
                sep=delimiter).values.astype(dtype).squeeze()
        return x
    
def save_text(filepath, data, header=None, fmt='%d', delimiter=' '):
    if isinstance(data, basestring):
        with open(filepath, 'w') as f:
            f.write(data)
    else:
        np.savetxt(filepath, data, fmt=fmt, newline='\n', delimiter=delimiter)
        # Write a header.
        if header is not None:
            with open(filepath, 'r') as f:
                contents = f.read()
            contents_updated = str(header) + '\n' + contents
            with open(filepath, 'w') as f:
                f.write(contents_updated)
        


# -----------------------------------------------------------------------------
# XML functions
# -----------------------------------------------------------------------------
def load_xml(filepath, fileindex=1):
    """Load a XML Klusters file."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    d = {}

    ac = root.find('acquisitionSystem')
    if ac is not None:
        nc = ac.find('nChannels')
        if nc is not None:
            d['total_channels'] = int(nc.text)
        sr = ac.find('samplingRate')
        if sr is not None:
            d['rate'] = float(sr.text)

    sd = root.find('spikeDetection')
    if sd is not None:
        cg = sd.find('channelGroups')
        if cg is not None:
            # find the group corresponding to the fileindex
            g = cg.findall('group')[fileindex-1]
            if g is not None:
                ns = g.find('nSamples')
                if ns is not None:
                    d['nsamples'] = int(ns.text)
                nf = g.find('nFeatures')
                if nf is not None:
                    d['fetdim'] = int(nf.text)
                c = g.find('channels')
                if c is not None:
                    d['nchannels'] = len(c.findall('channel'))
    
    if 'nchannels' not in d:
        d['nchannels'] = d['total_channels']
    
    return d
    

# -----------------------------------------------------------------------------
# Binary files functions
# -----------------------------------------------------------------------------
def load_binary(file, dtype=None, count=None):
    if dtype is None:
        dtype = np.dtype(np.int16)
    if count is None:
        X = np.fromfile(file, dtype=dtype)
    else:
        X = np.fromfile(file, dtype=dtype, count=count)
    return X

def save_binary(file, data):
    data.tofile(file)
    
def save_pickle(file, obj):
    with open(file, 'wb') as f:
        cPickle.dump(obj, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        obj = cPickle.load(f)
    return obj
    

# -----------------------------------------------------------------------------
# Preprocessing functions
# -----------------------------------------------------------------------------
def normalize(data, range=(-1., 1.), symmetric=False):
    """Normalize an array so that all values fit in a given range.
    
    Symmetrical normalization means that after normalization, values equal to
    0 stay equal to 0.
    
    """
    m = data.min()
    M = data.max()
    
    if symmetric:
        vx = max(np.abs(m), np.abs(M))
        m, M = -vx, vx
        
    data = range[0] + (range[1] - range[0]) * (data - m) * (1. / (M - m))
    
    return data

    
    