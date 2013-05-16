"""A navigator within pairs of items."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter, OrderedDict

import numpy as np


# -----------------------------------------------------------------------------
# Pair Navigator
# -----------------------------------------------------------------------------
class PairNavigator(object):
    """Navigator within pairs of items."""
    def __init__(self, pairs=OrderedDict()):
        """
        pairs is an OrderedDict where each key is an item0, and each value
        is a list of item1.
        """
        self.pairs = pairs
        self.reset()
        
    def reset(self):
        self.history = []  # list of visited item0
        self.index = (0, -1)  # best item index, item index
        self.renaming = {}
        self.hidden = set()  # items that are currently hidden
        
        
    # Navigation get methods.
    # -----------------------
    def item0(self):
        if not self.pairs:
            return
        index0, index1 = self.index
        # Constrain index0.
        index0 = np.clip(index0, 0, len(self.pairs) - 1)
        # Access the item.
        best_item = self.pairs.keys()[index0]
        return best_item
        
    def item1(self):
        if not self.pairs:
            return
        index0, index1 = self.index
        best_item = self.item0()
        # Constrain index1.
        index1 = np.clip(index1, 0, len(self.pairs[best_item]) - 1)
        # Access the item.
        return self.pairs[best_item][index1]
    
    def position(self):
        if not self.pairs:
            return
        index0, index1 = self.index
        # Constrain index.
        index0 = np.clip(index0, 0, len(self.pairs) - 1)
        best_item = self.pairs.keys()[index0]
        index1 = np.clip(index1, 0, len(self.pairs[best_item]) - 1)
        # Access the item pair.
        best_item = self.pairs.keys()[index0]
        item0 = best_item
        if len(self.pairs[best_item]) == 0:
            return
        item1 = self.pairs[best_item][index1]
        return item0, item1
        
    def constrain_index(self):
        if not self.pairs:
            return
        index0, index1 = self.index
        index0 = np.clip(index0, 0, len(self.pairs) - 1)
        best_item = self.pairs.keys()[index0]
        index1 = np.clip(index1, 0, len(self.pairs[best_item]) - 1)
        self.index = (index0, index1)
        
        
    # Navigation set methods.
    # -----------------------
    def visit(self):
        """Mark current item0 as visited."""
        if not self.pairs:
            return
        item0 = self.item0()
        if item0 not in self.history:
            self.history.append(item0)
    
    def next0(self):
        if not self.pairs:
            return
        i0, i1 = self.index
        if i0 >= len(self.pairs) - 1:
            return
        else:
            # Mark the previous item as done, except if the item0 is 0.
            if i0 >= 0:
                self.visit()
            # Go to the next item0 that is not marked.
            for k in xrange(1, len(self.pairs) - i0):
                self.index = (i0 + k, 0)
                if (self.item0() not in self.history):
                    break
            while self.is_hidden(self.renamed(self.position())) is True:
                i0, i1 = self.index
                if i0 >= len(self.pairs) - 1:
                    return
                else:
                    self.index = (i0 + 1, i1)
            return self.renamed(self.position())
        
    def next1(self):
        if not self.pairs:
            return
        i0, i1 = self.index
        item0 = self.item0()
        items1 = self.pairs[item0]
        if i1 >= len(items1) - 1 or i0 < 0:
            return
        else:
            self.index = (i0, i1 + 1)
            pair = self.position()
        while self.is_hidden(self.renamed(self.position())) is True:
            i0, i1 = self.index
            if i1 >= len(items1) - 1 or i0 < 0:
                return
            else:
                self.index = (i0, i1 + 1)
        return self.renamed(self.position())
            
    def previous0(self):
        if not self.pairs:
            return
        i0, i1 = self.index
        if i0 <= 0:
            return
        else:
            self.index = (i0 - 1, 0)
            while self.is_hidden(self.renamed(self.position())) is True:
                i0, i1 = self.index
                if i0 <= 0:
                    return
                else:
                    self.index = (i0 - 1, 0)
            return self.renamed(self.position())
            
    def previous1(self):
        if not self.pairs:
            return
        i0, i1 = self.index
        if i1 > 0:
            self.index = (i0, i1 - 1)
            # return self.renamed(self.position())
            while self.is_hidden(self.renamed(self.position())) is True:
                i0, i1 = self.index
                if i1 > 0:
                    self.index = (i0, i1 - 1)
                else:
                    return
            return self.renamed(self.position())
        else:
            return
    
    def current(self):
        while self.is_hidden(self.renamed(self.position())) is True:
            if self.next1() is None:
                if self.next0() is None:
                    break
        return self.renamed(self.position())
    
    def is_hidden(self, pair):
        if pair is None:
            return None
        item0, item1 = pair
        is_hidden = item0 in self.hidden or item1 in self.hidden
        return is_hidden
            
    
    
    # Update methods.
    # ---------------
    def rename(self, renaming):
        """Rename items. Can be undone."""
        for i in xrange(len(self.history)):
            item = self.history[i]
            if item in renaming:
                self.history[i] = renaming[item]
        # Items that are deleted are those who are renamed.
        deleted = set(renaming.keys()) - set(renaming.values())
        self.history = list(set(self.history) - deleted)
        self.renaming.update(renaming)
    
    def undo_rename(self, keys):
        """Undo a renaming operation."""
        for item in keys:
            self.renaming.pop(item, None)
        
    def renamed(self, pair):
        """Rename filter called right before returning items, so that renaming
        can be reversed."""
        if pair is None:
            return
        item0, item1 = pair
        while item0 in self.renaming:
            item0 = self.renaming[item0]
        while item1 in self.renaming:
            item1 = self.renaming[item1]
        return item0, item1
        
    def hide(self, items):
        if not hasattr(items, '__len__'):
            items = [items]
        [self.hidden.add(item) for item in items]
    
    def unhide(self, items):
        if not hasattr(items, '__len__'):
            items = [items]
        [self.hidden.remove(item) 
            for item in items if item in self.hidden]
        
    def update(self, pairs, renaming={}):
        """Happens when going to the next item0, if a modification happened."""
        # Mark the current item0 as visited.
        self.visit()
        # Update the pairs.
        self.pairs = pairs
        # # Handle renaming in the history.
        if renaming:
            self.rename(renaming)
        # Reset the indices. The next call to next0() will make the index
        # jump to the next non-visited item.
        self.renaming = {}
        # self.hidden = set()
        self.index = (-1, -1)
    
    