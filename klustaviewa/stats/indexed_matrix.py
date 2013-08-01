"""This module implements an indexed matrix for storing pairwise statistics."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import namedtuple
from itertools import product

import numpy as np


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def is_default_slice(item):
    return (isinstance(item, slice) and item.start is None and item.stop is None
        and item.step is None)

def is_indices(item):
    return (isinstance(item, list) or isinstance(item, tuple) or 
        isinstance(item, np.ndarray) or isinstance(item, (int, long, np.integer)))
        

# -----------------------------------------------------------------------------
# Indexed matrix
# -----------------------------------------------------------------------------
class IndexedMatrix(object):
    def __init__(self, indices=[], dtype=None, shape=None, data=None):
        self.indices = np.sort(np.unique(indices))
        self.dtype = dtype
        self.n = len(self.indices)
        if data is not None:
            shape = data.shape
        if shape is None:
            self.shape = (self.n,) * 2
        else:
            assert shape[:2] == (self.n, self.n)
            self.shape = shape
        if data is None:
            self._array = np.zeros(self.shape, dtype=self.dtype)
        else:
            self._array = data
        self.ndim = self._array.ndim
    
    
    # Indices
    # -------
    def add_indices(self, indices):
        """Add new indices only if they don't already exist."""
        if isinstance(indices, (int, long, np.integer)):
            indices = [indices]
        if len(indices) == 0:
            return
        # Keep only those indices which do not exist already.
        indices = self.not_in_indices(indices)
        # Raise an error if at least one requested index is already in the
        # current array indices.
        if np.any(np.in1d(indices, self.indices)):
            index = indices[np.nonzero(np.in1d(indices, self.indices))[0][0]]
            raise IndexError("Index {0:d} is already an index of the array".
                format(index))
        # Get the new indices.
        indices_old = self.indices
        indices_new = np.array(sorted(set(indices_old).union(indices)))
        # Create the new array.
        self.indices = indices_new
        self.n = len(indices_new)
        self.shape = (self.n, self.n) + tuple(self.shape[2:])
        array_new = np.zeros(self.shape, dtype=self.dtype)
        notblank_new = np.zeros(self.shape[:2], dtype=np.bool)
        # Fill the new array with the existing values, except if the previous
        # array was empty.
        if len(indices_old) > 0:
            indices_relative = self.to_relative(indices_old)
            for j in xrange(len(indices_relative)):
                array_new[indices_relative, indices_relative[j], ...] = \
                    self._array[:, j, ...]
        self._array = array_new
    
    def remove_indices(self, indices):
        if isinstance(indices, (int, long, np.integer)):
            indices = [indices]
        if len(indices) == 0:
            return
        # Raise an error if at least one requested index is already in the
        # current array indices.
        if np.any(~np.in1d(indices, self.indices)):
            index = indices[np.nonzero(~np.in1d(indices, self.indices))[0][0]]
            raise IndexError("Index {0:d} is not an index of the array".
                format(index))
        indices_kept = np.array(sorted(set(self.indices) - set(indices)))
        indices_relative = self.to_relative(indices_kept)
        self._array = self._array[indices_relative, :, ...] \
            [:, indices_relative, ...]
        self.indices = indices_kept
        self.n = len(self.indices)
        self.shape = (self.n, self.n) + tuple(self.shape[2:])
    
    def to_array(self, copy=False):
        if copy:
            return self._array.copy()
        else:
            return self._array
        
    def to_absolute(self, indices_relative, conserve_single_indices=True):
        if isinstance(indices_relative, (int, long, np.integer)):
            indices_relative = [indices_relative]
            single_index = True
        else:
            single_index = False
        if len(indices_relative) == 0:
            return []
        indices_absolute = self.indices[indices_relative]
        if single_index and conserve_single_indices:
            indices_absolute = indices_absolute[0]
        return indices_absolute
        
    def to_relative(self, indices_absolute, conserve_single_indices=True):
        if isinstance(indices_absolute, (int, long, np.integer)):
            indices_absolute = [indices_absolute]
            single_index = True
        else:
            single_index = False
        if len(indices_absolute) == 0:
            return []
        # Ensure all requested absolute indices are valid.
        if not np.all(np.in1d(indices_absolute, self.indices)):
            index = indices_absolute[
                np.nonzero(~np.in1d(indices_absolute, self.indices))[0][0]]
            raise IndexError("The index {0:d} is not valid.".format(index))
        indices_relative = np.digitize(indices_absolute, self.indices) - 1
        if single_index and conserve_single_indices:
            indices_relative = indices_relative[0]
        return indices_relative
    
    def not_in_indices(self, indices=None):
        if indices is None:
            indices = self.indices
        return sorted(set(indices) - set(self.indices))
    
    @property
    def size(self):
        return np.prod(self.shape)
    
    
    # Access
    # ------
    def __getitem__(self, item):
        """Access [:,indices] or [indices,:]."""
        # If item is (item0, item1).
        if isinstance(item, tuple) and len(item) == 2:
            # item0 is default slice, and item1 contains indices.
            if is_default_slice(item[0]) and is_indices(item[1]):
                return self._array[:, self.to_relative(item[1]), ...]
            # item0 contains indices, and item1 is default slice.
            elif is_default_slice(item[1]) and is_indices(item[0]):
                return self._array[self.to_relative(item[0]), :, ...]
            # item0 and item1 are indices.
            elif is_indices(item[0]) and is_indices(item[1]):
                value = (self._array[self.to_relative(item[0], False), :, ...]
                    [:, self.to_relative(item[1], False), ...])
                # Squeeze the appropriate dimensions if the requested indices
                # are scalars and not enumerables.
                if (isinstance(item[0], (int, long, np.integer)) and 
                    isinstance(item[1], (int, long, np.integer))):
                    value = value[0, 0, ...]
                elif isinstance(item[0], (int, long, np.integer)):
                    value = value[0, ...]
                elif isinstance(item[1], (int, long, np.integer)):
                    value = value[:, 0, ...]
                return value
        raise IndexError(("Indexed matrices can only be accessed with [x,y] "
        "with x and y indices or default slice ':'."))
        
    def __setitem__(self, item, value):
        # If item is (item0, item1).
        if isinstance(item, tuple) and len(item) == 2:
            # item0 is default slice, and item1 contains indices.
            if is_default_slice(item[0]) and is_indices(item[1]):
                self._array[:, self.to_relative(item[1]), ...] = value
                return
            # item0 contains indices, and item1 is default slice.
            elif is_default_slice(item[1]) and is_indices(item[0]):
                self._array[self.to_relative(item[0]), :, ...] = value
                return
            # item0 and item1 are indices.
            elif is_indices(item[0]) and is_indices(item[1]):
                # Case where both items are enumerables.
                if (not isinstance(item[0], (int, long, np.integer)) and 
                    not isinstance(item[1], (int, long, np.integer))):
                    # TODO: this is inefficient. Rather inspire from update.
                    # Assign value slice after slice.
                    for j in xrange(len(item[1])):
                        try:
                            self._array[self.to_relative(item[0]), 
                                        self.to_relative(item[1][j]), ...] \
                                        = value[:, j, ...]
                        except TypeError:
                            # Case where value is a scalar and cannot be
                            # sliced.
                            self._array[self.to_relative(item[0]), 
                                        self.to_relative(item[1][j]), ...] \
                                        = value
                else:
                    self._array[self.to_relative(item[0]),
                                self.to_relative(item[1]), 
                                ...] = value
                return
        raise IndexError(("Indexed matrices can only be accessed with [x,y] "
        "with x and y indices or default slice ':'."))
         
    def __len__(self):
        return self.n
    
    def submatrix(self, indices):
        if len(indices) == 0:
            return IndexedMatrix(shape=(0, 0) + self.shape[2:])
        if not np.all(np.in1d(indices, self.indices)):
            raise IndexError("Some indices are not valid.")
        indices = sorted(indices)
        submatrix = IndexedMatrix(indices=indices,
            data=self[indices, indices].copy())
        return submatrix
        
    def __repr__(self):
        return self._array.__repr__()
    

class CacheMatrix(IndexedMatrix):
    """Implement a cache using an underlying IndexedMatrix.
    
    Requirements:
      
      * Initialize with empty indices.
      * The matrix should be symmetric at all times.
      * add_indices and remove_indices should never be called directly.
      * When updating, always update using all (i, *) and (*, i) pairs,
        using update_from_dict.
      * Indices are transparently added when updating the cache.
      * One can call invalidate to remove indices.
    
    """
    def __init__(self, dtype=None, shape=None, data=None):
        super(CacheMatrix, self).__init__(dtype=dtype, shape=shape, data=data)
        # List of key indices.
        self.key_indices = []
    
    def invalidate(self, indices):
        """Remove indices from the cache."""
        if isinstance(indices, (int, long, np.integer)):
            indices = [indices]
        # Only remove the indices that are present in the existing indices.
        indices = sorted(set(indices).intersection(set(self.indices)))
        self.remove_indices(indices)
        for index in indices:
            if index in self.key_indices:
                self.key_indices.remove(index)
    
    def not_in_key_indices(self, indices):
        """Return those indices which are not key indices and thus need to
        be updated."""
        if isinstance(indices, (int, long, np.integer)):
            indices = [indices]
        return sorted(set(indices) - set(self.key_indices))
    
    def update(self, key_indices, dic):
        """Update the cache using a dictionary indexed by pairs of absolute
        indices. New indices are silently added. The key indices must also
        be provided."""
        
        items0, items1 = zip(*dic.keys())
        # Add non-existing indices.
        indices_new0 = self.not_in_indices(items0)
        indices_new1 = self.not_in_indices(items1)
        indices_new = sorted(set(indices_new0).union(set(indices_new1)))
        if len(indices_new) > 0:
            self.add_indices(indices_new)
        # Update key indices.
        if isinstance(key_indices, (int, long, np.integer)):
            key_indices = [key_indices]
        self.key_indices = sorted(set(self.key_indices).union(
            set(key_indices)))
        # Update the matrix with the new values.
        items0_relative = self.to_relative(items0)
        items1_relative = self.to_relative(items1)
        for (item0, item1, item0_relative, item1_relative) in zip(
                items0, items1, items0_relative, items1_relative):
            self._array[item0_relative, item1_relative, ...] = dic[(
                item0, item1)]
       
