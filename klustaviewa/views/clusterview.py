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
from qtools import QtGui, QtCore

from kwiklib.dataio import get_array
from kwiklib.dataio.selection import get_indices, select
from klustaviewa.gui.icons import get_icon
from kwiklib.utils.colors import COLORMAP, random_color
from kwiklib.utils import logger as log
from klustaviewa import SETTINGS
from kwiklib.utils.persistence import encode_bytearray, decode_bytearray
from klustaviewa.views.treemodel import TreeModel, TreeItem

# Specific item classes
# ---------------------
class ClusterItem(TreeItem):
    def __init__(self, parent=None, clusteridx=None, color=None, bgcolor=None,
            spkcount=None, quality=None):
        if color is None:
            color = 0
        if quality is None:
            quality = 0.
        data = OrderedDict()
        # different columns fields
        data['quality'] = quality
        data['spkcount'] = spkcount
        data['color'] = color
        self.bgcolor = bgcolor
        # data['bgcolor'] = bgcolor
        # the index is the last column
        data['clusteridx'] = clusteridx
        super(ClusterItem, self).__init__(parent=parent, data=data)

    def spkcount(self):
        return self.item_data['spkcount']

    def quality(self):
        return self.item_data['quality']

    def color(self):
        return self.item_data['color']
                
    def clusteridx(self):
        return self.item_data['clusteridx']


class GroupItem(TreeItem):
    def __init__(self, parent=None, name=None, groupidx=None, color=None, spkcount=None):
        data = OrderedDict()
        # different columns fields
        data['name'] = name
        data['quality'] = 0.
        data['spkcount'] = spkcount
        data['color'] = color
        # the index is the last column
        data['groupidx'] = groupidx
        super(GroupItem, self).__init__(parent=parent, data=data)

    def name(self):
        return self.item_data['name']

    def color(self):
        return self.item_data['color']

    def quality(self):
        return self.item_data['quality']
        
    def spkcount(self):
        return self.item_data['spkcount']
        
    def groupidx(self):
        return self.item_data['groupidx']
        
    def __repr__(self):
        return "<group {0:d} '{1:s}'>".format(self.groupidx(), self.name())
        

# Custom model
# ------------
class ClusterViewModel(TreeModel):
    headers = ['Cluster', 'Quality', 'Spikes', 'Color']
    clustersMoved = QtCore.pyqtSignal(np.ndarray, int)
    
    def __init__(self, **kwargs):
        """Initialize the tree model.
        
        Arguments:
          * clusters: a Nspikes long array with the cluster index for each
            spike.
          * clusters_info: an Info object with fields names, colors, spkcounts,
            groups_info.
        
        """
        super(ClusterViewModel, self).__init__(self.headers)
        self.background = {}
        self.load(**kwargs)
        
    
    # I/O methods
    # -----------
    def load(self, cluster_colors=None, cluster_groups=None,
        group_colors=None, group_names=None, cluster_sizes=None,
        cluster_quality=None, background={}):
        
        if group_names is None or cluster_colors is None:
            return
        
        # Create the tree.
        # go through all groups
        for groupidx, groupname in group_names.iteritems():
            spkcount = np.sum(cluster_sizes[cluster_groups == groupidx])
            groupitem = self.add_group_node(groupidx=groupidx, name=groupname,
                # color=group_colors[groupidx], spkcount=spkcount)
                color=select(group_colors, groupidx), spkcount=spkcount)
        
        # go through all clusters
        for clusteridx, color in cluster_colors.iteritems():
            if cluster_quality is not None:
                try:
                    quality = get_array(select(cluster_quality, clusteridx))[0]
                except IndexError:
                    quality = 0.
            else:
                quality = 0.
            # add cluster
            bgcolor = background.get(clusteridx, None)
            clusteritem = self.add_cluster(
                clusteridx=clusteridx,
                # name=info.names[clusteridx],
                color=color,
                bgcolor=bgcolor,
                quality=quality,
                # spkcount=cluster_sizes[clusteridx],
                spkcount=select(cluster_sizes, clusteridx),
                # assign the group as a parent of this cluster
                parent=self.get_group(select(cluster_groups, clusteridx)))
    
    
    # Data methods
    # ------------
    def headerData(self, section, orientation, role):
        if (orientation == QtCore.Qt.Horizontal) and (role == QtCore.Qt.DisplayRole):
            return self.headers[section]
        
    def data(self, index, role):
        item = index.internalPointer()
        
        col = index.column()
        # group item
        if type(item) == GroupItem:
            if col == 0:
                if role == QtCore.Qt.DisplayRole:
                    return str(item.name())
            # quality
            elif col == 1:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                if role == QtCore.Qt.DisplayRole:
                    return #"%." % item.quality()
            # spkcount
            elif col == 2:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                if role == QtCore.Qt.DisplayRole:
                    return str(item.spkcount())
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
                elif role == QtCore.Qt.BackgroundRole:
                    if item.bgcolor is None:
                        return
                    elif item.bgcolor == 'candidate':
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(color[0], color[1], color[2], 90)
                    elif item.bgcolor == 'target':
                        # return QtGui.QColor(177, 177, 177, 255)
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(color[0], color[1], color[2], 255)
                elif role == QtCore.Qt.ForegroundRole:
                    if item.bgcolor is None:
                        return QtGui.QColor(177, 177, 177, 255)
                    elif item.bgcolor == 'target':
                        return QtGui.QColor(0, 0, 0, 255)
            # quality
            elif col == 1:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                elif role == QtCore.Qt.DisplayRole:
                    return "%.3f" % item.quality()
                elif role == QtCore.Qt.BackgroundRole:
                    if item.bgcolor is None:
                        return
                    elif item.bgcolor == 'candidate':
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(color[0], color[1], color[2], 90)
                    elif item.bgcolor == 'target':
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(color[0], color[1], color[2], 255)
                        # return QtGui.QColor(177, 177, 177, 255)
                elif role == QtCore.Qt.ForegroundRole:
                    if item.bgcolor is None:
                        return QtGui.QColor(177, 177, 177, 255)
                    elif item.bgcolor == 'target':
                        return QtGui.QColor(0, 0, 0, 255)
            # spkcount
            elif col == 2:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                if role == QtCore.Qt.DisplayRole:
                    return "%d" % item.spkcount()
                elif role == QtCore.Qt.BackgroundRole:
                    if item.bgcolor is None:
                        return
                    elif item.bgcolor == 'candidate':
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(color[0], color[1], color[2], 90)
                    elif item.bgcolor == 'target':
                        # return QtGui.QColor(177, 177, 177, 255)
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(color[0], color[1], color[2], 255)
                elif role == QtCore.Qt.ForegroundRole:
                    if item.bgcolor is None:
                        return QtGui.QColor(177, 177, 177, 255)
                    elif item.bgcolor == 'target':
                        return QtGui.QColor(0, 0, 0, 255)
                
            # color
            elif col == self.columnCount() - 1:
                if role == QtCore.Qt.BackgroundRole:
                    color = np.array(COLORMAP[item.color()]) * 255
                    return QtGui.QColor(*color)
                    
        # default
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return item.data(col)
            
        if role == QtCore.Qt.ForegroundRole:
            return QtGui.QColor(177, 177, 177, 255)
          
    def setData(self, index, data, role=None):
        if role is None:
            role = QtCore.Qt.EditRole
        if index.isValid() and role == QtCore.Qt.EditRole:
            item = index.internalPointer()
            if index.column() == 0:
                item.item_data['name'] = data
            elif index.column() == 1:
                item.item_data['quality'] = data
            elif index.column() == 2:
                item.item_data['spkcount'] = data
            elif index.column() == 3:
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
            self.setData(self.index(group.row(), 2), spkcount)
    
    def set_quality(self, quality):
        """quality is a Series with cluster index and quality value."""
        for clusteridx, value in quality.iteritems():
            groupidx = self.get_groupidx(clusteridx)
            # If the cluster does not exist yet in the view, just discard it.
            if groupidx is None:
                continue
            group = self.get_group(groupidx)
            cluster = self.get_cluster(clusteridx)
            self.setData(self.index(cluster.row(), 1, parent=group.index), value)
    
    def set_background(self, background=None):
        """Set the background of some clusters. The argument is a dictionary
        clusteridx ==> color index."""
        if background is not None:
            # Record the changed clusters.
            self.background.update(background)
            # Get all clusters to update.
            keys = self.background.keys()
            # Reset the keys
            if not background:
                self.background = {}
        # If background is None, use the previous one.
        else:
            background = self.background
            keys = self.background.keys()
        for clusteridx in keys:
            bgcolor = background.get(clusteridx, None)
            groupidx = self.get_groupidx(clusteridx)
            # If the cluster does not exist yet in the view, just discard it.
            if groupidx is None:
                continue
            group = self.get_group(groupidx)
            cluster = self.get_cluster(clusteridx)
            index = self.index(cluster.row(), 0, parent=group.index)
            index1 = self.index(cluster.row(), 1, parent=group.index)
            if index.isValid():
                item = index.internalPointer()
                # bgcolor = True means using the same color
                # if bgcolor is True:
                    # bgcolor = item.color()
                item.bgcolor = bgcolor
                self.dataChanged.emit(index, index1)
    
    
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
    # The boolean indicates whether the selection has been initiated externally
    # or not (internally by clicking on items in the view).
    clustersSelected = QtCore.pyqtSignal(np.ndarray, bool)
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
        self.wizard = False
        self.clusters_selected_previous = []
        
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
    def set_data(self, **kwargs):
        
        # if cluster_colors is None:
        if kwargs.get('cluster_colors', None) is None:
            return
        
        self.model = ClusterViewModel(**kwargs)
        
        self.setModel(self.model)
        self.expandAll()
        
        # set spkcount column size
        self.header().resizeSection(1, 60)
        self.header().resizeSection(2, 60)
        # set color column size
        self.header().resizeSection(3, 40)
        
        # HACK: drag is triggered in the model, so connect it to move_clusters
        # in this function
        self.model.clustersMoved.connect(self.move_clusters)
        
    def clear(self):
        self.setModel(ClusterViewModel())
        
        
    
    # Public methods
    # --------------
    def select(self, clusters, groups=None, wizard=False):
        """Select multiple clusters from their indices."""
        self.wizard = wizard
        if clusters is None:
            clusters = []
        if groups is None:
            groups = []
        if isinstance(clusters, (int, long, np.integer)):
            clusters = [clusters]
        if isinstance(groups, (int, long, np.integer)):
            groups = [groups]
        if len(clusters) == len(groups) == 0:
            return
        # Convert to lists.
        clusters = list(clusters)
        groups = list(groups)
        selection_model = self.selectionModel()
        selection = QtGui.QItemSelection()
        # Select groups.
        for groupidx in groups:
            group = self.model.get_group(groupidx)
            if group is not None:
                selection.select(group.index, group.index)
        # Select clusters.
        for clusteridx in clusters:
            cluster = self.model.get_cluster(clusteridx)
            if cluster is not None:
                selection.select(cluster.index, cluster.index)
        # Process selection.
        selection_model.select(selection, 
                selection_model.Clear |
                selection_model.Current |
                selection_model.Select | 
                selection_model.Rows 
                )
        if len(clusters) > 0:
            cluster = self.model.get_cluster(clusters[-1])
            if cluster is not None:
                # Set current index in the selection.
                selection_model.setCurrentIndex(
                    cluster.index,
                    QtGui.QItemSelectionModel.NoUpdate)
                # Scroll to that cluster.
                self.scrollTo(cluster.index)
                    
    def unselect(self):
        self.selectionModel().clear()
    
    def move_clusters(self, clusters, groupidx):
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        if len(clusters) == 0:
            return
        # Signal.
        log.debug("Moving clusters {0:s} to group {1:d}.".format(
            str(clusters), groupidx))
        self.clustersMoved.emit(np.array(clusters), groupidx)
    
    def add_group(self, name, clusters=[]):
        color = random_color()
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
    
    def move_to_good(self, clusters):
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        self.move_clusters(clusters, 2)
    
    def move_to_mua(self, clusters):
        if not hasattr(clusters, '__len__'):
            clusters = [clusters]
        self.move_clusters(clusters, 1)
    
    def set_quality(self, quality):
        self.model.set_quality(quality)
    
    def set_background(self, background=None):
        self.model.set_background(background)
    
    
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
        self.rename_group_action.setShortcut("F2")
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
        
        self.move_to_good_action = QtGui.QAction("Move to &good", self)
        # self.move_to_good_action.setIcon(get_icon('noise'))
        self.move_to_good_action.triggered.connect(self.move_to_good_callback)
        
        # Add actions to the widget.
        self.addAction(self.change_color_action)
        self.addAction(self.add_group_action)
        self.addAction(self.rename_group_action)
        self.addAction(self.remove_group_action)
        self.addAction(self.move_to_noise_action)
        self.addAction(self.move_to_mua_action)
        self.addAction(self.move_to_good_action)
        
    def create_context_menu(self):
        self.create_color_dialog()
        
        self.context_menu = QtGui.QMenu(self)
        self.context_menu.addAction(self.change_color_action)
        self.context_menu.addSeparator()
        self.context_menu.addAction(self.move_to_noise_action)
        self.context_menu.addAction(self.move_to_mua_action)
        self.context_menu.addAction(self.move_to_good_action)
        self.context_menu.addSeparator()
        self.context_menu.addAction(self.add_group_action)
        self.context_menu.addAction(self.rename_group_action)
        self.context_menu.addAction(self.remove_group_action)
        
    def update_actions(self):
        clusters = self.selected_clusters()
        groups = self.selected_groups()
        
        if len(groups) > 0:
            self.rename_group_action.setEnabled(True)
            # First three groups are not removable (noise and MUA and good).
            if 0 not in groups and 1 not in groups and 2 not in groups:
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
            self.move_to_good_action.setEnabled(True)
        else:
            self.move_to_noise_action.setEnabled(False)
            self.move_to_mua_action.setEnabled(False)
            self.move_to_good_action.setEnabled(False)
        
    def contextMenuEvent(self, event):
        action = self.context_menu.exec_(self.mapToGlobal(event.pos()))
    
    def currentChanged(self, index, previous):
        self.current_item = index.internalPointer()
    
    
    # Callback
    # --------
    def change_color_callback(self, checked=None):
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
        color = np.argmin(np.abs(COLORMAP[1:,:] - rgb).sum(axis=1)) + 1 
        # Change the color and emit the signal.
        if isinstance(item, ClusterItem):
            self.change_cluster_color(item.clusteridx(), color)
        elif isinstance(item, GroupItem):
            self.change_group_color(item.groupidx(), color)
            
    def add_group_callback(self, checked=None):
        text, ok = QtGui.QInputDialog.getText(self, 
            "Group name", "Name group:",
            QtGui.QLineEdit.Normal, "New group")
        if ok:
            self.add_group(text, self.selected_clusters())
        
    def remove_group_callback(self, checked=None):
        item = self.current_item
        if isinstance(item, GroupItem):
            self.remove_group(item.groupidx())
            
    def rename_group_callback(self, checked=None):
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
    
    def move_to_noise_callback(self, checked=None):
        clusters = self.selected_clusters()
        self.move_to_noise(clusters)
        
    def move_to_mua_callback(self, checked=None):
        clusters = self.selected_clusters()
        self.move_to_mua(clusters)
    
    def move_to_good_callback(self, checked=None):
        clusters = self.selected_clusters()
        self.move_to_good(clusters)
    
    
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
        self.clustersSelected.emit(np.array(clusters, dtype=np.int32),
            self.wizard)
        
        # if group_indices:
            # self.scrollTo(group_indices[-1].index)
        if len(self.clusters_selected_previous) <= 1:
            if len(clusters) == 1:
                self.scrollTo(self.model.get_cluster(clusters[0]).index)
            elif len(group_indices) == 1:
                self.scrollTo(group_indices[0].index)
    
        self.wizard = False
        self.clusters_selected_previous = clusters
        self.update_actions()
    
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
        elif ((ctrl or shift) and (key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down))):
            return
        else:
            return super(ClusterView, self).keyPressEvent(e)
        
    
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
        
        