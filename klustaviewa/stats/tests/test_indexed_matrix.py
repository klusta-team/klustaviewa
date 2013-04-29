"""Unit tests for stats.cache module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from nose.tools import raises
import numpy as np

from klustaviewa.stats.indexed_matrix import IndexedMatrix, CacheMatrix


# -----------------------------------------------------------------------------
# Indexed matrix tests
# -----------------------------------------------------------------------------
def test_indexed_matrix_0():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices)
    
    assert np.array_equal(matrix.to_absolute([0, 3, 1]), [2, 7, 3])
    assert np.array_equal(matrix.to_absolute(2), 5)
    
    assert np.array_equal(matrix.to_relative([2, 7, 3]), [0, 3, 1])
    assert np.array_equal(matrix.to_relative(5), 2)
    
@raises(IndexError)
def test_indexed_matrix_1():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices)
    # This should raise an IndexError.
    matrix[0, 0]
    
def test_indexed_matrix_2():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices)
    
    assert matrix[2, 2] == 0.
    
    assert np.array_equal(matrix[:, 2], np.zeros(4))
    assert np.array_equal(matrix[7, :], np.zeros(4))
    assert np.array_equal(matrix[[2, 5], :], np.zeros((2, 4)))
    
    assert np.array_equal(matrix[[2, 5, 3], [2]], np.zeros((3, 1)))
    assert np.array_equal(matrix[[2, 5, 3], 2], np.zeros(3))
    assert np.array_equal(matrix[[2], [2, 5, 3]], np.zeros((1, 3)))
    assert np.array_equal(matrix[2, [2, 5, 3]], np.zeros(3))
    assert np.array_equal(matrix[[5, 7], [3, 2]], np.zeros((2, 2)))
    
def test_indexed_matrix_3():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices)
    
    matrix.add_indices(4)
    
    assert matrix.shape == (5, 5)
    assert np.array_equal(matrix.indices, [2, 3, 4, 5, 7])
    
def test_indexed_matrix_4():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices)
    matrix.add_indices(7)
    assert np.array_equal(matrix.indices, indices)
    
def test_indexed_matrix_5():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices)
    
    matrix.add_indices([6, 10])
    
    assert matrix.shape == (6, 6)
    assert np.array_equal(matrix.indices, [2, 3, 5, 6, 7, 10])
    
    matrix.remove_indices(7)
    
    assert matrix.shape == (5, 5)
    assert np.array_equal(matrix.indices, [2, 3, 5, 6, 10])
    
@raises(IndexError)
def test_indexed_matrix_6():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices)
    
    matrix.add_indices([6, 10])
    
    # One of the indices does not exist, so this raises an Exception.
    matrix.remove_indices([5, 6, 9])
    
def test_indexed_matrix_7():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices)
    
    matrix[2, 3] = 10
    assert np.all(matrix[2, 3] == 10)
    
    matrix[5, :] = 20
    assert np.all(matrix[5, :] == 20)
    
    matrix[:, 7] = 30
    assert np.all(matrix[:, 7] == 30)
    
    matrix[[2, 3], 5] = 40
    assert np.all(matrix[[2, 3], 5] == 40)
    
    matrix[[2, 3], [5, 7]] = 50
    assert np.all(matrix[[2, 3], [5, 7]] == 50)
    
def test_indexed_matrix_8():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices, shape=(4, 4, 10))
    
    x = np.random.rand(10)
    matrix[7, 7] = x
    assert np.array_equal(matrix[7, 7], x)
    
    assert np.array_equal(matrix[7, :][-1, :], x)
    assert np.array_equal(matrix[[2, 7], 7][-1, :], x)
    
    assert np.array_equal(matrix[[2, 5, 3], [2]], np.zeros((3, 1, 10)))
    assert np.array_equal(matrix[[2, 5, 3], 2], np.zeros((3, 10)))
    assert np.array_equal(matrix[[2], [2, 5, 3]], np.zeros((1, 3, 10)))
    assert np.array_equal(matrix[2, [2, 5, 3]], np.zeros((3, 10)))
    assert np.array_equal(matrix[[5, 7], [3, 2]], np.zeros((2, 2, 10)))

    matrix.remove_indices(5)
    
    assert matrix.to_array().shape == (3, 3, 10)
    assert np.array_equal(matrix[7, 7], x)
    
def test_indexed_matrix_9():
    matrix = IndexedMatrix()
    indices = [10, 20]
    matrix.add_indices(10)
    assert np.array_equal(matrix.not_in_indices(indices), [20])
    
    matrix[10, 10] = 1
    assert np.array_equal(matrix.not_in_indices(indices), [20])
    
    matrix.add_indices(20)
    assert np.array_equal(matrix.not_in_indices(indices), [])
    
    matrix[20, :] = 0
    matrix[:, 20] = 0
    
    assert np.array_equal(matrix.not_in_indices(indices), [])
    
def test_indexed_matrix_10():
    indices = [2, 3, 5, 7]
    matrix = IndexedMatrix(indices=indices, shape=(4, 4, 10))
    matrix[3, 7] = np.ones(10)
    matrix[2, 5] = 2 * np.ones(10)
    
    submatrix = matrix.submatrix([3,7])
    assert submatrix.shape == (2, 2, 10)
    assert np.array_equal(submatrix.to_array()[0, 1, ...], np.ones(10))
    
    submatrix = matrix.submatrix([2,5])
    assert submatrix.shape == (2, 2, 10)
    assert np.array_equal(submatrix.to_array()[0, 1, ...], 2 * np.ones(10))
    
    
# -----------------------------------------------------------------------------
# Cache matrix tests
# -----------------------------------------------------------------------------
def test_cache_matrix_1():
    indices = [2, 3, 5, 7]
    matrix = CacheMatrix(shape=(0, 0, 10))
    
    assert np.array_equal(matrix.not_in_indices(indices), indices)
    
    d = {(i, j): i + j for i in indices for j in indices}
    matrix.update(indices, d)
    
    matrix_actual = (np.array(indices).reshape((-1, 1)) + 
        np.array(indices).reshape((1, -1)))
    assert np.array_equal(matrix.to_array()[:, :, 0], matrix_actual)
    
    assert np.array_equal(matrix.not_in_indices(indices), [])
    
def test_cache_matrix_2():
    indices = [2, 3, 5, 7]
    matrix = CacheMatrix(shape=(0, 0, 10))
    
    d = {(i, j): i + j for i in indices for j in indices}
    matrix.update(indices, d)
    
    assert np.array_equal(matrix.not_in_indices(indices), [])
    
    matrix.invalidate([2, 5])
    assert np.array_equal(matrix.not_in_indices(indices), [2, 5])
    
def test_cache_matrix_2():
    indices = [2, 3, 5, 7]
    matrix = CacheMatrix()
    
    assert np.array_equal(matrix.not_in_key_indices(indices), indices)
    
    matrix.update(2, {(2, 2): 0, (2, 3): 0, (3, 2): 0})
    assert np.array_equal(matrix.not_in_key_indices(indices), [3, 5, 7])
    
    matrix.update([2, 3], {(2, 2): 0, (2, 3): 0, (3, 2): 0, (3, 3): 0})
    assert np.array_equal(matrix.not_in_key_indices(indices), [5, 7])
    
    matrix.invalidate([2, 5])
    assert np.array_equal(matrix.not_in_key_indices(indices), [2, 5, 7])
    
    d = {(i, j): i + j for i in indices for j in indices}
    matrix.update(indices, d)
    assert np.array_equal(matrix.not_in_key_indices(indices), [])
    
    
    