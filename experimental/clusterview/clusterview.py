"""Cluster View: show all clusters and groups."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import inspect
import bisect
from collections import OrderedDict

import pandas as pd
import numpy as np
import numpy.random as rnd
from qtools import QtGui, QtCore, show_window

from kwiklib.utils.colors import COLORMAP, random_color
# from klustaviewa.views.tests.utils import show_view

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class Cluster(object):
    def __init__(self, idx, color=1, group=None, size=0, quality=0.):
        self.idx = idx
        self.color = color
        self.group = group
        self.size = size
        self.quality = quality
     
    def __repr__(self):
        return "<Cluster {idx}, group {grp}>".format(idx=self.idx,
                                                     grp=self.group)
     
class Group(object):
    def __init__(self, idx, color=1, name=None, clusters=None):
        self.idx = idx
        self.color = color
        if name is None:
            name = 'Group {0:d}'.format(idx)
        self.name = name
        if clusters is None:
            clusters = []
        for cluster in clusters:
            cluster.group = idx
        self.clusters = sorted(clusters)

    def add_cluster(self, cluster):
        cluster.group = self.idx
        self.clusters[cluster.idx] = cluster

    def add_clusters(self, clusters):
        for cluster in clusters:
            self.add_cluster(cluster)
        
    def remove_cluster(self, cluster):
        if not isinstance(cluster, Cluster):
            cluster = self.clusters[cluster]
        cluster.group = None
        del self.clusters[cluster.idx]

    def remove_clusters(self, clusters):
        for cluster in clusters:
            self.remove_cluster(cluster)

    def __repr__(self):
        return "<Group {idx}, clusters {clu}>".format(idx=self.idx,
            clu=str([clu.idx for clu in self.clusters]))
            
            
# -----------------------------------------------------------------------------
# Mock data
# -----------------------------------------------------------------------------
def random_cluster(idx):
    return Cluster(idx, color=rnd.randint(low=1, high=30), 
                        size=rnd.randint(low=10, high=1000),
                        quality=rnd.rand(),
                        )
            
def random_clusters(indices):
    return [random_cluster(_) for _ in indices]
        
def random_group(idx, cluster_indices):
    return Group(idx, color=rnd.randint(low=1, high=30), 
                      clusters=random_clusters(cluster_indices))

def random_groups(ngroups, nclusters=None):
    groups = []
    cluster_groups = np.random.randint(low=0, high=ngroups, size=nclusters)
    for _ in range(ngroups):
        cluster_indices = np.nonzero(cluster_groups == _)[0]
        groups.append(random_group(_, cluster_indices))
    return groups
    

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def show_view(view_class, **kwargs):
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
            self.setMouseTracking(True)
            self.setWindowTitle("View")
            self.view = view_class(self, **kwargs)
            self.setCentralWidget(self.view)
            self.move(100, 100)
            self.show()
    window = show_window(TestWindow)
    return window
    
    
# -----------------------------------------------------------------------------
# Qt Model
# -----------------------------------------------------------------------------
class ClusterModel(QtCore.QAbstractItemModel):
    def __init__(self, group_sizes={}, cluster_groups=None):
        """
        
        Arguments:
        * group_sizes: a dictionary {groupidx: nclusters} that gives the number
          of clusters in each group.
        * cluster_groups: an object such that cluster_groups[clusteridx]
          returns the groupidx that contains that cluster.
        
        """
        QtCore.QAbstractItemModel.__init__(self)
        self.group_sizes = group_sizes
        self.cluster_groups = cluster_groups
        self.beginInsertRows(QtCore.QModelIndex(), 0, len(self.group_sizes))
        self.endInsertRows()
        
        
    def index(self, row, column, parent=None):
        if parent is None:
            parent = QtCore.QModelIndex()
        data = parent.internalPointer()
        if data is None:
            return self.createIndex(row, column, ('group', row))
        else:
            tp, idx = data
            assert tp == 'cluster'
            return self.createIndex(row, column, ('cluster', idx))
        
    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()
        tp, idx = index.internalPointer()
        if tp == 'group':
            return QtCore.QModelIndex()
        elif tp == 'cluster':
            # Get groupidx from clusteridx (=idx).
            groupidx = self.cluster_groups[idx]
            return self.createIndex(groupidx, 0, ('group', groupidx))
        return QtCore.QModelIndex()

    def rowCount(self, parent):
        # if parent is None:
            # return len(self.group_sizes)
        # else:
        data = parent.internalPointer()
        if data is None:
            # Total number of groups.
            return len(self.group_sizes)
        tp, idx = data
        if tp == 'group':
            # Number of clusters in group idx.
            return self.group_sizes[idx]
        else:
            # Clusters do not have children.
            return 0
        
    def columnCount(self, parent):
        return 4

    def data(self, index, role):
        if role != QtCore.Qt.DisplayRole:
            return None
        item = index.internalPointer()
        if item is None:
            return
        tp, idx = item
        print tp, idx
        col = index.column()
        
        if tp == 'group':
            return 'Group {0:d}'.format(idx)
        elif tp == 'cluster':
            return 'Cluster {0:d}'.format(idx)
        
        
    def setData(self, index, data, role):
        return False
        
    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
               QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
            

class ClusterView(QtGui.QTreeView):
    class ClusterDelegate(QtGui.QStyledItemDelegate):
        def paint(self, painter, option, index):
            """Disable the color column so that the color remains the same even
            when it is selected."""
            # deactivate all columns except the first one, so that selection
            # is only possible in the first column
            if index.column() >= 1:
                if option.state and QtGui.QStyle.State_Selected:
                    option.state = option.state and QtGui.QStyle.State_Off
            super(ClusterView.ClusterDelegate, self).paint(painter, option, index)
    
    def __init__(self, parent, getfocus=None, **kwargs):
        super(ClusterView, self).__init__(parent)
        
        # Focus policy.
        if getfocus:
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
        else:
            self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.setAllColumnsShowFocus(True)
        # select full rows
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.setMaximumWidth(300)
        
        # self.setRootIsDecorated(False)
        self.setItemDelegate(self.ClusterDelegate())
        
        self.model = ClusterModel(**kwargs)
        
        self.setModel(self.model)
        self.expandAll()
        
        # set spkcount column size
        self.header().resizeSection(1, 60)
        self.header().resizeSection(2, 60)
        # set color column size
        self.header().resizeSection(3, 40)

if __name__ == '__main__':
    ngroups = 3
    nclusters = 100

    cluster_groups = np.random.randint(low=0, high=ngroups, size=nclusters)    
    group_sizes = np.bincount(cluster_groups)

    show_view(ClusterView,
              group_sizes=group_sizes, 
              cluster_groups=cluster_groups)
    
