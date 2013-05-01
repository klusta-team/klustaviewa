"""Cluster View: show all clusters and groups."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import inspect
from collections import OrderedDict

import pandas as pd
import numpy as np
import numpy.random as rnd
from galry import QtGui, QtCore

from klustaviewa.io.selection import get_indices, select
from klustaviewa.gui.icons import get_icon
from klustaviewa.utils.colors import COLORMAP, next_color
import klustaviewa.utils.logger as log
from klustaviewa.utils.settings import SETTINGS
from klustaviewa.utils.persistence import encode_bytearray, decode_bytearray


# Generic classes
# ---------------
class TreeItem(object):
    def __init__(self, parent=None, data=None):
        """data is an OrderedDict"""
        self.parent_item = parent
        self.index = QtCore.QModelIndex()
        self.children = []
        # by default: root
        if data is None:
            data = OrderedDict(name='root')
        self.item_data = data
    
    def appendChild(self, child):
        self.children.append(child)
    
    def removeChild(self, child):
        self.children.remove(child)

    def removeChildAt(self, row):
        self.children.pop(row)
        
    def insertChild(self, child, index):
        self.children.insert(index, child)
        
    def child(self, row):
        return self.children[row]
        
    def rowCount(self):
        return len(self.children)
        
    def columnCount(self):
        return len(self.item_data)

    def data(self, column):
        if column >= self.columnCount():
            return None
        return self.item_data.get(self.item_data.keys()[column], None)
        
    def row(self):
        if self.parent_item is None:
            return 0
        return self.parent_item.children.index(self)
        
    def parent(self):
        return self.parent_item
       
        
class TreeModel(QtCore.QAbstractItemModel):
    def __init__(self, headers):
        QtCore.QAbstractItemModel.__init__(self)
        self.root_item = TreeItem()
        self.headers = headers
        
    def add_node(self, item_class=None, item=None, parent=None, **kwargs):
        """Add a node in the tree.
        
        
        """
        if parent is None:
            parent = self.root_item
        if item is None:
            if item_class is None:
                item_class = TreeItem
            item = item_class(parent=parent, **kwargs)
        
        row = parent.rowCount()
        item.index = self.createIndex(row, 0, item)
        
        self.beginInsertRows(parent.index, row-1, row-1)
        parent.appendChild(item)
        self.endInsertRows()
        
        return item
        
    def remove_node(self, child, parent=None):
        if parent is None:
            parent = self.root_item
            
        row = child.row()
        self.beginRemoveRows(parent.index, row, row)
        parent.removeChild(child)
        self.endRemoveRows()
        
    # def move_node(self, child, parent_target, child_target=None):
        # row = child.row()
        # parent_source = child.parent()
        # if child_target is not None:
            # child_target_row = child_target.row()
        # else:
            # child_target_row = parent_target.rowCount()
        # canmove = self.beginMoveRows(parent_source.index, row, row,
            # parent_target.index, child_target_row)
        # if canmove:
            
            
            # if parent is None:
                # parent = self.root_item
            # if item is None:
                # if item_class is None:
                    # item_class = TreeItem
            # item = child._(parent=parent, **kwargs)
            
            # row = parent.rowCount()
            # item.index = self.createIndex(row, 0, item)
            
            
            # parent_target.insertChild(child_new, child_target_row)
            # if parent_target == parent_source:
                # if child_target_row < row:
                    # row += 1
                # parent_source.removeChildAt(row)
            # else:
                # parent_source.removeChild(child)
            # # child.parent_item = parent_target
            
            # self.endMoveRows()
    
    def get_descendants(self, parents):
        if type(parents) != list:
            parents = [parents]
        nodes = []
        for parent in parents:
            nodes.append(parent)
            if parent.children:
                nodes.extend(self.get_descendants(parent.children))
        return nodes
        
    def all_nodes(self):
        return self.get_descendants(self.root_item)
        
    def index(self, row, column, parent=None):
        if parent is None:
            parent = self.root_item.index
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        child_item = parent_item.child(row)
        if child_item:
            index = self.createIndex(row, column, child_item)
            child_item.index = index
            return index
        else:
            return QtCore.QModelIndex()

    def parent(self, item):
        if not item.isValid():
            return QtCore.QModelIndex()
        item = item.internalPointer()
        parent_item = item.parent()
        if (parent_item == self.root_item):
            return QtCore.QModelIndex()
        index = self.createIndex(parent_item.row(), 0, parent_item)
        parent_item.index = index
        return index

    def rowCount(self, parent=None):
        if parent is None:
            parent = QtCore.QModelIndex()
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        return parent_item.rowCount()
        
    def columnCount(self, parent=None):
        if parent is None:
            parent = QtCore.QModelIndex()
        if not parent.isValid():
            return len(self.headers)
        return parent.internalPointer().columnCount()

    def data(self, index, role):
        if role != QtCore.Qt.DisplayRole:
            return None
        item = index.internalPointer()
        return item.data(index.column())
        
    def setData(self, index, data, role):
        return False
        
    def supportedDropActions(self): 
        return QtCore.Qt.MoveAction         

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
               QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
            
    def mimeTypes(self):
        return ['text/xml']

    def mimeData(self, indexes):
        data = ",".join(set([str(index.internalPointer()) for index in indexes]))
        mimedata = QtCore.QMimeData()
        mimedata.setData('text/xml', data)
        return mimedata

    def dropMimeData(self, data, action, row, column, parent):
        parent_item = parent.internalPointer()
        target = parent_item
        sources = data.data('text/xml').split(',')
        self.drag(target, sources)
        return True

    def drag(self, target, sources):
        """
        
        To be overriden.
        
        """
        print "drag", target, sources


# Specific item classes
# ---------------------
class ClusterItem(TreeItem):
    def __init__(self, parent=None, clusteridx=None, color=None, 
            spkcount=None):
        if color is None:
            color = 0 #(1., 1., 1.)
        data = OrderedDict()
        # different columns fields
        data['spkcount'] = spkcount
        data['color'] = color
        # the index is the last column
        data['clusteridx'] = clusteridx
        super(ClusterItem, self).__init__(parent=parent, data=data)

    def spkcount(self):
        return self.item_data['spkcount']

    def color(self):
        return self.item_data['color']
                
    def clusteridx(self):
        return self.item_data['clusteridx']


class GroupItem(TreeItem):
    def __init__(self, parent=None, name=None, groupidx=None, color=None, spkcount=None):
        data = OrderedDict()
        # different columns fields
        data['name'] = name
        data['spkcount'] = spkcount
        data['color'] = color
        # the index is the last column
        data['groupidx'] = groupidx
        super(GroupItem, self).__init__(parent=parent, data=data)

    def name(self):
        return self.item_data['name']

    def color(self):
        return self.item_data['color']

    def spkcount(self):
        return self.item_data['spkcount']
        
    def groupidx(self):
        return self.item_data['groupidx']
        
    def __repr__(self):
        return "<group {0:d} '{1:s}'>".format(self.groupidx(), self.name())
        

# Custom model
# ------------
class ClusterGroupManager(TreeModel):
    headers = ['Cluster', 'Spikes', 'Color']
    clustersMoved = QtCore.pyqtSignal(np.ndarray, int)
    
    def __init__(self, cluster_colors=None, cluster_groups=None,
        group_colors=None, group_names=None, cluster_sizes=None):
        """Initialize the tree model.
        
        Arguments:
          * clusters: a Nspikes long array with the cluster index for each
            spike.
          * clusters_info: an Info object with fields names, colors, spkcounts,
            groups_info.
        
        """
        super(ClusterGroupManager, self).__init__(self.headers)
        self.load(cluster_colors=cluster_colors,
                  cluster_groups=cluster_groups,
                  group_colors=group_colors,
                  group_names=group_names,
                  cluster_sizes=cluster_sizes)
        
    
    # I/O methods
    # -----------
    def load(self, cluster_colors=None, cluster_groups=None,
        group_colors=None, group_names=None, cluster_sizes=None):

        # Create the tree.
        # go through all groups
        for groupidx, groupname in group_names.iteritems():
            # add group
            spkcount = np.sum(cluster_sizes[cluster_groups == groupidx])
            groupitem = self.add_group_node(groupidx=groupidx, name=groupname,
                # color=group_colors[groupidx], spkcount=spkcount)
                color=select(group_colors, groupidx), spkcount=spkcount)
        
        # go through all clusters
        for clusteridx, color in cluster_colors.iteritems():
            # add cluster
            clusteritem = self.add_cluster(
                clusteridx=clusteridx,
                # name=info.names[clusteridx],
                color=color,
                # spkcount=cluster_sizes[clusteridx],
                spkcount=select(cluster_sizes, clusteridx),
                # assign the group as a parent of this cluster
                # parent=self.get_group(cluster_groups[clusteridx]))
                parent=self.get_group(select(cluster_groups, clusteridx)))
    
    def save(self):
        groups = self.get_groups()
        allclusters = self.get_clusters()
        
        ngroups = len(groups)
        nclusters = len(allclusters)
        
        # Initialize objects.
        cluster_colors = pd.Series(np.zeros(nclusters, dtype=np.int32))
        cluster_groups = pd.Series(np.zeros(nclusters, dtype=np.int32))
        group_colors = pd.Series(np.zeros(ngroups, dtype=np.int32))
        group_names = pd.Series(np.zeros(ngroups, dtype=np.str_))
        
        # Loop through all groups.
        for group in groups:
            groupidx = group.groupidx()
            clusters = self.get_clusters_in_group(groupidx)
            # set the group info object
            group_colors[groupidx] = group.color()
            group_names[groupidx] = group.name()
            # Loop through clusters in the current group.
            for cluster in clusters:
                clusteridx = cluster.clusteridx()
            cluster_colors[clusteridx] = cluster.color()
            cluster_groups[clusteridx] = groupidx
        
        return dict(cluster_colors=cluster_colors,
                    cluster_groups=cluster_groups,
                    group_colors=group_colors,
                    group_names=group_names)
    
    
    # Data methods
    # ------------
    def headerData(self, section, orientation, role):
        if (orientation == QtCore.Qt.Horizontal) and (role == QtCore.Qt.DisplayRole):
            return self.headers[section]
        
    def data(self, index, role):
        """Return custom background color for the last column of cluster
        items."""
        item = index.internalPointer()
        
        col = index.column()
        # group item
        if type(item) == GroupItem:
            if col == 0:
                if role == QtCore.Qt.DisplayRole:
                    return str(item.name())
            # spkcount
            elif col == 1:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                if role == QtCore.Qt.DisplayRole:
                    return "%d" % item.spkcount()
            # color
            elif col == self.columnCount() - 1:
                if role == QtCore.Qt.BackgroundRole:
                    if item.color() >= 0:
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(*color)
                elif role == QtCore.Qt.DisplayRole:
                    return ""
                
        # cluster item
        if type(item) == ClusterItem:
            # clusteridx
            if col == 0:
                if role == QtCore.Qt.DisplayRole:
                    return str(item.clusteridx())
            # spkcount
            elif col == 1:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                if role == QtCore.Qt.DisplayRole:
                    return "%d" % item.spkcount()
            # color
            elif col == self.columnCount() - 1:
                if role == QtCore.Qt.BackgroundRole:
                    color = np.array(COLORMAP[item.color()]) * 255
                    return QtGui.QColor(*color)
                    
        # default
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return item.data(col)
          
    def setData(self, index, data, role=None):
        if role is None:
            role = QtCore.Qt.EditRole
        if index.isValid() and role == QtCore.Qt.EditRole:
            item = index.internalPointer()
            if index.column() == 0:
                item.item_data['name'] = data
            elif index.column() == 1:
                item.item_data['spkcount'] = data
            elif index.column() == 2:
                item.item_data['color'] = data
            self.dataChanged.emit(index, index)
            return True
    
    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
               QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
    
    def update_group_sizes(self):
        for group in self.get_groups():
            spkcount = np.sum([cluster.spkcount() 
                for cluster in self.get_clusters_in_group(group.groupidx())])
            self.setData(self.index(group.row(), 1), spkcount)
    
    
    # Action methods
    # --------------
    def add_group(self, name, color):
        """Add a group."""
        groupidx = max([group.groupidx() for group in self.get_groups()]) + 1
        # Add the group in the tree.
        groupitem = self.add_group_node(groupidx=groupidx,
            name=name, spkcount=0, color=color)
        return groupitem
        
    def add_group_node(self, **kwargs):
        return self.add_node(item_class=GroupItem, **kwargs)
        
    def remove_group(self, group):
        """Remove an empty group. Raise an error if the group is not empty."""
        groupidx = group.groupidx()
        # check that the group is empty
        if self.get_clusters_in_group(groupidx):
            raise ValueError("group %d is not empty, unable to delete it" % \
                    groupidx)
        groups = [g for g in self.get_groups() if g.groupidx() == groupidx]
        if groups:
            group = groups[0]
            self.remove_node(group)
        else:
            log.warn("Group %d does not exist0" % groupidx)
        
    def add_cluster(self, parent=None, **kwargs):
        cluster = self.add_node(item_class=ClusterItem, parent=parent, 
                            **kwargs)
        return cluster
    
    def move_clusters(self, sources, target):
        # Get the groupidx if the target is a group,
        if type(target) == GroupItem:
            groupidx = target.groupidx()
            target = None
        # else, if it is a cluster, take the corresponding group.
        elif type(target) == ClusterItem:
            groupidx = self.get_groupidx(target.clusteridx())
        else:
            return None
            
        # Move clusters.
        target_group = self.get_group(groupidx)
        for node in sources:
            self._move_cluster(node, target_group, target)
        
        self.update_group_sizes()
    
    def _move_cluster(self, cluster, parent_target, child_target=None):
        row = cluster.row()
        parent_source = cluster.parent()
        # Find the row where the cluster needs to be inserted.
        if child_target is not None:
            child_target_row = child_target.row()
        else:
            child_target_row = parent_target.rowCount()
        # Begin moving the row.
        canmove = self.beginMoveRows(parent_source.index, row, row,
            parent_target.index, child_target_row)
        if canmove:
            # Create a new cluster, clone of the old one.
            cluster_new = ClusterItem(parent=parent_target,
                clusteridx=cluster.clusteridx(),
                spkcount=cluster.spkcount(),
                color=cluster.color())
            # Create the index.
            cluster_new.index = self.createIndex(child_target_row, 
                0, cluster_new)
            # Insert the new cluster.
            parent_target.insertChild(cluster_new, child_target_row)
            # Remove the old cluster.
            if parent_target == parent_source:
                if child_target_row < row:
                    row += 1
                parent_source.removeChildAt(row)
            else:
                parent_source.removeChild(cluster)
            self.endMoveRows()
        
    
    # Drag and drop for moving clusters
    # ---------------------------------
    def drag(self, target, sources):
        # Get source ClusterItem nodes.
        source_items = [node for node in self.all_nodes() 
            if (str(node) in sources and type(node) == ClusterItem )]
        # Find the target group.
        if type(target) == GroupItem:
            groupidx = target.groupidx()
        # else, if it is a cluster, take the corresponding group.
        elif type(target) == ClusterItem:
            groupidx = self.get_groupidx(target.clusteridx())
        # Emit internal signal to let TreeView emit a public signal, and to
        # effectively move the clusters.
        self.clustersMoved.emit(np.array([cluster.clusteridx() 
            for cluster in source_items]), groupidx)
        # Move clusters.
        # self.move_clusters(source_items, target)
    
    def rename_group(self, group, name):
        self.setData(self.index(group.row(), 0), name)
        
    def change_group_color(self, group, color):
        self.setData(self.index(group.row(), 2), color)
        
    def change_cluster_color(self, cluster, color):
        groupidx = self.get_groupidx(cluster.clusteridx())
        group = self.get_group(groupidx)
        self.setData(self.index(cluster.row(), 2, parent=group.index), color)
        
        
    # Getter methods
    # --------------
    def get_groups(self):
        return [group for group in self.get_descendants(self.root_item) \
            if (type(group) == GroupItem)]
        
    def get_group(self, groupidx):
        return [group for group in self.get_descendants(self.root_item) \
            if (type(group) == GroupItem) and \
                (group.groupidx() == groupidx)][0]
        
    def get_clusters(self):
        return [cluster for cluster in self.get_descendants(self.root_item) \
          if (type(cluster) == ClusterItem)]
            
    def get_cluster(self, clusteridx):
        l = [cluster for cluster in self.get_descendants(self.root_item) \
                  if (type(cluster) == ClusterItem) and \
                        (cluster.clusteridx() == clusteridx)]
        if l:
            return l[0]
                
    def get_clusters_in_group(self, groupidx):
        group = self.get_group(groupidx)
        return [cluster for cluster in self.get_descendants(group) \
            if (type(cluster) == ClusterItem)]
        
    def get_groupidx(self, clusteridx):
        """Return the group index currently assigned to the specifyed cluster
        index."""
        for group in self.get_groups():
            clusterindices = [cluster.clusteridx() \
                            for cluster in self.get_clusters_in_group(group.groupidx())]
            if clusteridx in clusterindices:
                return group.groupidx()
        return None
          
        
# Top-level widget
# ----------------
class ClusterView(QtGui.QTreeView):
    # Signals
    # -------
    # Selection.
    clustersSelected = QtCore.pyqtSignal(np.ndarray)
    # groupsSelected = QtCore.pyqtSignal(np.ndarray)
    
    # Cluster and group info.
    clusterColorChanged = QtCore.pyqtSignal(int, int)
    groupColorChanged = QtCore.pyqtSignal(int, int)
    groupRenamed = QtCore.pyqtSignal(int, object)

    clustersMoved = QtCore.pyqtSignal(np.ndarray, int)
    groupRemoved = QtCore.pyqtSignal(int)
    groupAdded = QtCore.pyqtSignal(int, str, int)
    
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
    
    def __init__(self, parent, getfocus=None):
        super(ClusterView, self).__init__(parent)
        # Current item.
        self.current_item = None
        
        # Focus policy.
        if getfocus:
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
        else:
            self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.setAllColumnsShowFocus(True)
        # self.setFirstColumnSpanned(0, QtCore.QModelIndex(), True)
        # select full rows
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.setMaximumWidth(300)
        
        # self.setRootIsDecorated(False)
        self.setItemDelegate(self.ClusterDelegate())
        
        # Create menu.
        self.create_actions()
        self.create_context_menu()
        
        self.restore_geometry()
    
    # Data methods
    # ------------
    def set_data(self, 
        cluster_colors=None,
        cluster_groups=None,
        group_colors=None,
        group_names=None,
        cluster_sizes=None,):
        
        if cluster_colors is None:
            return
        
        self.model = ClusterGroupManager(
                          cluster_colors=cluster_colors,
                          cluster_groups=cluster_groups,
                          group_colors=group_colors,
                          group_names=group_names,
                          cluster_sizes=cluster_sizes)
        
        self.setModel(self.model)
        self.expandAll()
        
        # set spkcount column size
        self.header().resizeSection(1, 100)
        # set color column size
        self.header().resizeSection(2, 40)
        
        # HACK: drag is triggered in the model, so connect it to move_clusters
        # in this function
        self.model.clustersMoved.connect(self.move_clusters)
        
    
    # Public methods
    # --------------
    def select(self, clusters):
        """Select multiple clusters from their indices."""
        if isinstance(clusters, (int, long)):
            clusters = [clusters]
        selection_model = self.selectionModel()
        selection = QtGui.QItemSelection()
        for clusteridx in clusters:
            cluster = self.model.get_cluster(clusteridx)
            if cluster is not None:
                selection.select(cluster.index, cluster.index)
        selection_model.select(selection, 
                selection_model.Clear |
                selection_model.Current |
                selection_model.Select | 
                selection_model.Rows 
                )
        if len(clusters) > 0:
            cluster = self.model.get_cluster(clusters[-1])
            if cluster is not None:
                selection_model.setCurrentIndex(
                    cluster.index,
                    QtGui.QItemSelectionModel.NoUpdate)
    
    def unselect(self):
        self.selectionModel().clear()
    
    def move_clusters(self, clusters, groupidx):
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        if len(clusters) == 0:
            return
        # self.model.move_clusters([self.model.get_cluster(clusteridx)
            # for clusteridx in clusters], self.model.get_group(groupidx))
        # Signal.
        log.debug("Moving clusters {0:s} to group {1:d}.".format(
            str(clusters), groupidx))
        self.clustersMoved.emit(np.array(clusters), groupidx)
    
    def add_group(self, name, clusters=[]):
        color = next_color(max([group.color()
            for group in self.model.get_groups()]))
        group = self.model.add_group(name, color)
        groupidx = group.groupidx()
        # Signal.
        log.debug("Adding group {0:s}.".format(name))
        self.groupAdded.emit(groupidx, name, color)
        # Move the selected clusters to the new group.
        if clusters:
            self.move_clusters(clusters, groupidx)
        self.expandAll()
        return groupidx
    
    def rename_group(self, groupidx, name):
        self.model.rename_group(self.model.get_group(groupidx), name)
        # Signal.
        log.debug("Rename group {0:d} to {1:s}.".format(
            groupidx, name))
        self.groupRenamed.emit(groupidx, name)
        
    def remove_group(self, groupidx):
        self.model.remove_group(self.model.get_group(groupidx))
        # Signal.
        log.debug("Removed group {0:d}.".format(groupidx))
        self.groupRemoved.emit(groupidx)
        
    def change_cluster_color(self, clusteridx, color):
        self.model.change_cluster_color(self.model.get_cluster(clusteridx), 
            color)
        # Signal.
        log.debug("Changed color of cluster {0:d} to {1:d}.".format(
            clusteridx, color))
        self.clusterColorChanged.emit(clusteridx, color)
        
    def change_group_color(self, groupidx, color):
        self.model.change_group_color(self.model.get_group(groupidx), color)
        # Signal.
        log.debug("Changed color of group {0:d} to {1:d}.".format(
            groupidx, color))
        self.groupColorChanged.emit(groupidx, color)
    
    def move_to_noise(self, clusters):
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        self.move_clusters(clusters, 0)
    
    def move_to_mua(self, clusters):
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        self.move_clusters(clusters, 1)
    
    
    # Menu methods
    # ------------
    def create_color_dialog(self):
        self.color_dialog = QtGui.QColorDialog(self)
        self.color_dialog.setOptions(QtGui.QColorDialog.DontUseNativeDialog)
        for i in xrange(48):
            if i < len(COLORMAP):
                rgb = COLORMAP[i] * 255
            else:
                rgb = (255, 255, 255)
                # rgb = (1., 1., 1.)
            k = 6 * (np.mod(i, 8)) + i // 8
            self.color_dialog.setStandardColor(k, QtGui.qRgb(*rgb))
        
    def create_actions(self):
        
        self.change_color_action = QtGui.QAction("Change &color", self)
        self.change_color_action.triggered.connect(self.change_color_callback)
        
        self.add_group_action = QtGui.QAction("&Add group", self)
        self.add_group_action.triggered.connect(self.add_group_callback)
        
        self.rename_group_action = QtGui.QAction("Re&name group", self)
        self.rename_group_action.setShortcut(QtCore.Qt.Key_F2)
        self.rename_group_action.triggered.connect(self.rename_group_callback)
        
        self.remove_group_action = QtGui.QAction("&Remove group", self)
        self.remove_group_action.triggered.connect(self.remove_group_callback)
        
        self.move_to_mua_action = QtGui.QAction("Move to &MUA", self)
        self.move_to_mua_action.setShortcut("Delete")
        self.move_to_mua_action.setIcon(get_icon('multiunit'))
        self.move_to_mua_action.triggered.connect(self.move_to_mua_callback)
        
        self.move_to_noise_action = QtGui.QAction("Move to &noise", self)
        self.move_to_noise_action.setShortcut('Shift+Delete')
        self.move_to_noise_action.setIcon(get_icon('noise'))
        self.move_to_noise_action.triggered.connect(self.move_to_noise_callback)
        
        # Add actions to the widget.
        self.addAction(self.change_color_action)
        self.addAction(self.add_group_action)
        self.addAction(self.rename_group_action)
        self.addAction(self.remove_group_action)
        self.addAction(self.move_to_noise_action)
        self.addAction(self.move_to_mua_action)
        
    def create_context_menu(self):
        self.create_color_dialog()
        
        self.context_menu = QtGui.QMenu(self)
        self.context_menu.addAction(self.change_color_action)
        self.context_menu.addSeparator()
        self.context_menu.addAction(self.move_to_noise_action)
        self.context_menu.addAction(self.move_to_mua_action)
        self.context_menu.addSeparator()
        self.context_menu.addAction(self.add_group_action)
        self.context_menu.addAction(self.rename_group_action)
        self.context_menu.addAction(self.remove_group_action)
        
    def contextMenuEvent(self, event):
        clusters = self.selected_clusters()
        groups = self.selected_groups()
        
        if len(groups) > 0:
            self.rename_group_action.setEnabled(True)
            # First two groups are not removable (noise and MUA).
            if 0 not in groups and 1 not in groups:
                self.remove_group_action.setEnabled(True)
            else:
                self.remove_group_action.setEnabled(False)
        else:
            self.rename_group_action.setEnabled(False)
            self.remove_group_action.setEnabled(False)
            
        if len(clusters) > 0 or len(groups) > 0:
            self.change_color_action.setEnabled(True)
        else:
            self.change_color_action.setEnabled(False)
            
        if len(clusters) > 0:
            self.move_to_noise_action.setEnabled(True)
            self.move_to_mua_action.setEnabled(True)
        else:
            self.move_to_noise_action.setEnabled(False)
            self.move_to_mua_action.setEnabled(False)
            
        action = self.context_menu.exec_(self.mapToGlobal(event.pos()))
    
    def currentChanged(self, index, previous):
        self.current_item = index.internalPointer()
    
    
    # Callback
    # --------
    def change_color_callback(self, checked):
        item = self.current_item
        initial_color = item.color()
        if initial_color >= 0:
            initial_color = 255 * COLORMAP[initial_color]
            initial_color = QtGui.QColor(*initial_color)
            color = QtGui.QColorDialog.getColor(initial_color)
        else:
            color = QtGui.QColorDialog.getColor()
        # return if the user canceled
        if not color.isValid():
            return
        # get the RGB values of the chosen color
        rgb = np.array(color.getRgbF()[:3]).reshape((1, -1))
        # take the closest color in the palette
        color = np.argmin(np.abs(COLORMAP - rgb).sum(axis=1))
        # Change the color and emit the signal.
        if isinstance(item, ClusterItem):
            self.change_cluster_color(item.clusteridx(), color)
        elif isinstance(item, GroupItem):
            self.change_group_color(item.groupidx(), color)
            
    def add_group_callback(self, checked):
        text, ok = QtGui.QInputDialog.getText(self, 
            "Group name", "Name group:",
            QtGui.QLineEdit.Normal, "New group")
        if ok:
            self.add_group(text, self.selected_clusters())
        
    def remove_group_callback(self, checked):
        item = self.current_item
        if isinstance(item, GroupItem):
            self.remove_group(item.groupidx())
            
    def rename_group_callback(self, checked):
        group = self.current_item
        if isinstance(group, GroupItem):
            groupidx = group.groupidx()
            name = group.name()
            text, ok = QtGui.QInputDialog.getText(self, 
                "Group name", "Rename group:",
                QtGui.QLineEdit.Normal, name)
            if ok:
                # Rename the group.
                self.rename_group(groupidx, text)
    
    def move_to_noise_callback(self, checked):
        clusters = self.selected_clusters()
        self.move_to_noise(clusters)
        
    def move_to_mua_callback(self, checked):
        clusters = self.selected_clusters()
        self.move_to_mua(clusters)
    
    
    # Get methods
    # -----------
    def get_cluster_indices(self):
        return [cluster.clusteridx() for cluster in self.model.get_clusters()]
    
    def get_group_indices(self):
        return [group.groupidx() for group in self.model.get_groups()]
    
    def get_cluster_indices_in_group(self, groupidx):
        return [cluster.clusteridx() 
            for cluster in self.model.get_clusters_in_group(groupidx)]
    
    
    # Selection methods
    # -----------------
    def selectionChanged(self, selected, deselected):
        super(ClusterView, self).selectionChanged(selected, deselected)
        selected_clusters = self.selected_clusters()
        selected_groups = self.selected_groups()
        # All clusters in selected groups minus selected clusters.
        clusters = [cluster 
            for group in selected_groups
                for cluster in self.get_cluster_indices_in_group(group)
                    if cluster not in selected_clusters]
        # Add selected clusters not in selected groups.
        clusters.extend([cluster
            for cluster in selected_clusters
                if (cluster not in clusters and
                    self.model.get_groupidx(cluster) not in selected_groups)
            ])
        
        # Selected groups.
        group_indices = [self.model.get_group(groupidx)
            for groupidx in selected_groups]
        
        # log.debug("Selected {0:d} clusters.".format(len(clusters)))
        # log.debug("Selected clusters {0:s}.".format(str(clusters)))
        self.clustersSelected.emit(np.array(clusters, dtype=np.int32))
        
        if group_indices:
            self.scrollTo(group_indices[-1].index)
        elif clusters:
            self.scrollTo(self.model.get_cluster(clusters[-1]).index)
    
    
    # Selected items
    # --------------
    def selected_items(self):
        """Return the list of selected cluster indices."""
        return [(v.internalPointer()) \
                    for v in self.selectedIndexes() \
                        if v.column() == 0]
                            
    def selected_clusters(self):
        """Return the list of selected cluster indices."""
        return [(v.internalPointer().clusteridx()) \
                    for v in self.selectedIndexes() \
                        if v.column() == 0 and \
                           type(v.internalPointer()) == ClusterItem]
              
    def selected_groups(self):
        """Return the list of selected groups."""
        return [(v.internalPointer().groupidx()) \
                    for v in self.selectedIndexes() \
                        if v.column() == 0 and \
                           type(v.internalPointer()) == GroupItem]
                

    # Event methods
    # -------------
    def keyPressEvent(self, e):
        key = e.key()
        modif = e.modifiers()
        ctrl = modif & QtCore.Qt.ControlModifier
        shift = modif & QtCore.Qt.ShiftModifier
        alt = modif & QtCore.Qt.AltModifier
        if (ctrl and key == QtCore.Qt.Key_A):
            self.select(self.get_cluster_indices())
        else:
            return super(ClusterView, self).keyPressEvent(e)
        
    def sizeHint(self):
        return QtCore.QSize(300, 600)
        
    
    # Save and restore geometry
    # -------------------------
    def save_geometry(self):
        SETTINGS['cluster_widget.geometry'] = encode_bytearray(
            self.saveGeometry())
        SETTINGS['cluster_widget.header'] = encode_bytearray(
            self.header().saveState())
        
    def restore_geometry(self):
        g = SETTINGS['cluster_widget.geometry']
        h = SETTINGS['cluster_widget.header']
        if g:
            self.restoreGeometry(decode_bytearray(g))
        if h:
            self.header().restoreState(decode_bytearray(h))
    
        
    def closeEvent(self, e):
        # Save the window geometry when closing the software.
        self.save_geometry()
        return super(ClusterView, self).closeEvent(e)
        
        