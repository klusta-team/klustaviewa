"""Utility functions for loading/saving files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os.path
import re
import cPickle

import numpy as np
# Try importing Pandas.
# try:
import pandas as pd
    # # Make sure that read_csv is available.
    # assert hasattr(pd, 'read_csv')
    # HAS_PANDAS = True
# except (ImportError, AssertionError):
    # log_warn("You should install Pandas v>=0.8.")
    # HAS_PANDAS = False

 
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

def get_array(data, copy=False, dosort=False):
    """Get a NumPy array from a NumPy array or a Pandas data object (Series,
    DataFrame or Panel)."""
    if data is None:
        return None
    if type(data) == np.ndarray:
        if copy:
            return data.copy()
        else:
            return data
    elif isinstance(data, (pd.DataFrame, pd.Panel)):
        if dosort:
            return np.array(data.sort_index().values)
        else:
            return data.values
    elif isinstance(data, (pd.Int64Index, pd.Index)):
        if dosort:
            return np.sort(data.values)
        else:
            return data.values
    else:
        if dosort:
            return np.array(data.sort_index().values)
        else:
            return data.values
    

# -----------------------------------------------------------------------------
# Text files related functions
# -----------------------------------------------------------------------------
# def load_text(filepath, dtype, skiprows=0):
    # if not filepath:
        # raise IOError("The filepath is empty.")
    # return np.loadtxt(filepath, dtype=dtype, skiprows=skiprows)

def first_row(filepath):
    with open(filepath, 'r') as f:
        n = f.readline().strip()
    return int(n)

# Faster load_text version if Pandas is installed.
# if HAS_PANDAS:
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
# Memory mapping
# -----------------------------------------------------------------------------
def load_binary_memmap(file, dtype=None, shape=None):
    return np.memmap(file, dtype=dtype, shape=shape)

def get_chunk(f, dtype, start, stop):
    itemsize = np.dtype(dtype).itemsize
    count = (stop - start)
    f.seek(itemsize * start, os.SEEK_SET)
    return np.fromfile(f, dtype=dtype, count=count)

def get_chunk_line(f, dtype):
    return np.fromstring(f.readline(), dtype=dtype, sep=' ')

class MemMappedArray(object):
    def __init__(self, filename, dtype):
        self.filename = filename
        self.dtype = dtype
        self.itemsize = np.dtype(self.dtype).itemsize
        self.f = open(filename, 'rb')
        
    def __getitem__(self, key):
        if isinstance(key, (int, long)):
            return get_chunk(self.f, self.dtype, key, key + 1)[0]
        elif isinstance(key, slice):
            return get_chunk(self.f, self.dtype, key.start, key.stop)
        
    def __del__(self):
        self.f.close()
        
class MemMappedBinary(object):
    def __init__(self, filename, dtype, rowsize=None):
        self.filename = filename
        self.dtype = dtype
        
        # Number of bytes of each item.
        self.itemsize = np.dtype(self.dtype).itemsize
        # Number of items in each row.
        self.rowsize = rowsize
        # Number of bytes in each row.
        self.rowsize_bytes = self.rowsize * self.itemsize
        # Current row.
        self.row = 0
        
        # Open the file in binary mode, even for text files.
        self.f = open(filename, 'rb')
        
    def next(self):
        """Return the values in the next row."""
        self.f.seek(self.rowsize_bytes * self.row, os.SEEK_SET)
        values = np.fromfile(self.f, dtype=self.dtype, count=self.rowsize)
        self.row += 1
        return values
        
    def close(self):
        self.f.close()
        
    def __del__(self):
        self.close()
    
class MemMappedText(object):
    BUFFER_SIZE = 10000
    
    def __init__(self, filename, dtype, skiprows=0):
        self.filename = filename
        self.dtype = dtype
        
        # Open the file in binary mode, even for text files.
        self.f = open(filename, 'rb')
        # Skip rows in non-binary mode.
        for _ in xrange(skiprows):
            self.f.readline()
            
        self._buffer_size = self.BUFFER_SIZE
        self._next_lines()
        
    def _next_lines(self):
        """Read several lines at once as it's faster than f.readline()."""
        self._lines = self.f.readlines(self._buffer_size)
        self._nlines = len(self._lines)
        self._index = 0
        
    def _next_line(self):
        if self._index >= self._nlines:
            self._next_lines()
        if self._index < self._nlines:
            line = self._lines[self._index]
            self._index += 1
        else:
            line = ''
        return line
        
    def next(self):
        """Return the values in the next row."""
        # HACK: remove the double spaces.
        l = self._next_line()
        if not l:
            return None
        l = l.replace('  ', ' ')
        values = np.fromstring(l, dtype=self.dtype, sep=' ')
        return values
        
    def close(self):
        self.f.close()
        
    def __del__(self):
        self.close()
    

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

    
    