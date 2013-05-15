"""Unit tests for wizard module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import OrderedDict

from nose.tools import raises
import numpy as np

from klustaviewa.wizard.pair_navigator import PairNavigator


# -----------------------------------------------------------------------------
# Navigation tests
# -----------------------------------------------------------------------------
def test_navigation():
    nitems = 10
    pairs = OrderedDict()
    items0 = np.random.permutation(nitems)
    for item0 in items0:
        pairs[item0] = np.random.permutation(nitems)
    
    # Initialize the navigator.
    n = PairNavigator(pairs)
    
    
    # Test impossible previous.
    assert n.previous0() is None
    assert n.previous1() is None
    
    
    # Check that the first propositions contain the best item.
    for _ in xrange(nitems // 2):
        assert items0[0] in n.next1()
    
    
    # Check next/previous.
    pair0 = n.next1()
    pair1 = n.next1()
    assert n.previous1() == pair0
    assert n.next1() == pair1
    
    
    # Check next item.
    assert items0[1] in n.next0()
    for _ in xrange(nitems // 2):
        assert items0[1] in n.next1()
    

    # Check previous item.
    pair = n.previous0()
    assert items0[0] in pair
    
    
    # Next again.
    assert items0[0] in n.next1()
    assert n.previous0() is None
    
    
    # Go to the end.
    [n.next1() for _ in xrange(nitems)]
    assert n.next1() is None
    
    n.next0()
    assert n.next1() is not None
    [n.next1() for _ in xrange(nitems)]
    assert n.next1() is None
    
    
def test_navigation_empty():
    nitems = 10
    pairs = OrderedDict()
    items0 = np.random.permutation(nitems)
    for item0 in items0:
        pairs[item0] = np.random.permutation(nitems)
    
    # Initialize the navigator.
    n = PairNavigator()
    
    
    # Test impossible previous.
    assert n.previous0() is None
    assert n.previous1() is None
    
    n.update(pairs)
    
    
    # Test impossible previous.
    assert n.previous0() is None
    assert n.previous1() is None
    
    
def test_navigation_update():
    nitems = 10
    pairs = OrderedDict()
    items0 = np.random.permutation(nitems)
    for item0 in items0:
        pairs[item0] = np.random.permutation(nitems)
    
    # Initialize the navigator.
    n = PairNavigator(pairs)
    
    item0, item1 = n.next1()
    if item0 == item1:
        item0, item1 = n.next1()
    
    # Next item0.
    item0next = pairs.keys()[1]
    
    # Simulate a merge.
    item0_new = nitems
    pairs_new = OrderedDict()
    renaming = {item0: item0_new, item1: item0_new}
    
    # The first item0 has changed.
    pairs_new[item0next] = np.random.permutation(nitems)
    
    # The previous best item0 is not second best.
    pairs_new[item0_new] = np.random.permutation(nitems)
    
    for key in pairs.keys():
        if key not in (item0next, item0_new, item0, item1):
            pairs_new[key] = list(set(np.random.permutation(nitems)) - set((item0, item1)))

    # Update the pairs
    n.update(pairs_new, renaming=renaming)
    
    # Go to the next item0.
    pair = n.next0()
    # The new best cluster should be there.
    [n.next1() for _ in xrange(nitems // 2)]
    assert item0next in n.next1()
    
    # Go to the next item0.
    pair = n.next0()
    # The old (renamed) best cluster should not be there as we visited it.
    assert item0_new not in pair
    
    
    