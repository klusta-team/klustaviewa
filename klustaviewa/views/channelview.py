"""Channel View: show all channels and groups."""

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

from kwiklib.dataio.selection import get_indices, select
from klustaviewa.gui.icons import get_icon
from kwiklib.utils.colors import COLORMAP, random_color
from kwiklib.utils import logger as log
from klustaviewa import SETTINGS
from kwiklib.utils.persistence import encode_bytearray, decode_bytearray
from klustaviewa.views.treemodel import TreeModel, TreeItem

# Specific item classes
# ---------------------
class ChannelItem(TreeItem):
    def __init__(self, parent=None, name=None, channelidx=None, color=None, bgcolor=None):
        if color is None:
            color = 0
        if name is None:
           name = str(channelidx) if channelidx is not None else ''
        data = OrderedDict()
        # different columns fields
        data['color'] = color
        self.bgcolor = bgcolor
        # the index is the last column
        data['channelidx'] = channelidx
        data['name'] = name
        super(ChannelItem, self).__init__(parent=parent, data=data)
        
    def name(self):
        return self.item_data['name']

    def color(self):
        return self.item_data['color']
                
    def channelidx(self):
        return self.item_data['channelidx']


class GroupItem(TreeItem):
    def __init__(self, parent=None, name=None, groupidx=None, color=None):
        data = OrderedDict()
        # different columns fields
        data['name'] = name
        data['color'] = color
        # the index is the last column
        data['groupidx'] = groupidx
        super(GroupItem, self).__init__(parent=parent, data=data)

    def name(self):
        return self.item_data['name']

    def color(self):
        return self.item_data['color']
        
    def groupidx(self):
        return self.item_data['groupidx']
        
    def __repr__(self):
        return "<group {0:d} '{1:s}'>".format(self.groupidx(), self.name())
        

# Custom model
# ------------
class ChannelViewModel(TreeModel):
    headers = ['Channel', 'Color']
    channelsMoved = QtCore.pyqtSignal(np.ndarray, int)
    
    def __init__(self, **kwargs):
        """Initialize the tree model.
        
        Arguments:
          * channels: a Nspikes long array with the channel index for each
            spike.
          * channels_info: an Info object with fields names, colors,
            groups_info.
        
        """
        super(ChannelViewModel, self).__init__(self.headers)
        self.background = {}
        self.load(**kwargs)
        
    
    # I/O methods
    # -----------
    def load(self, channel_colors=None, channel_groups=None,
        channel_names=None, group_colors=None, group_names=None,
        background={}):
        
        if group_names is None or channel_colors is None:
            return
        
        # Create the tree.
        # go through all groups
        for groupidx, groupname in group_names.iteritems():
            groupitem = self.add_group_node(groupidx=groupidx, name=groupname,
                color=select(group_colors, groupidx))
        
        # go through all channels
        for channelidx, color in channel_colors.iteritems():
            # add channel
            bgcolor = background.get(channelidx, None)
            channelitem = self.add_channel(
                name=channel_names[channelidx],
                channelidx=channelidx,
                color=color,
                bgcolor=None,
                parent=self.get_group(select(channel_groups, channelidx)))
    
    
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
            # color
            elif col == self.columnCount() - 1:
                if role == QtCore.Qt.BackgroundRole:
                    if item.color() >= 0:
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(*color)
                elif role == QtCore.Qt.DisplayRole:
                    return ""
                
        # channel item
        if type(item) == ChannelItem:
            # name
            if col == 0:
                if role == QtCore.Qt.DisplayRole:
                    return str(item.name())
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
                item.item_data['color'] = data
            self.dataChanged.emit(index, index)
            return True
            
    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
               QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
    
    def set_background(self, background=None):
        """Set the background of some channels. The argument is a dictionary
        channelidx ==> color index."""
        if background is not None:
            # Record the changed channels.
            self.background.update(background)
            # Get all channels to update.
            keys = self.background.keys()
            # Reset the keys
            if not background:
                self.background = {}
        # If background is None, use the previous one.
        else:
            background = self.background
            keys = self.background.keys()
        for channelidx in keys:
            bgcolor = background.get(channelidx, None)
            groupidx = self.get_groupidx(channelidx)
            # If the channel does not exist yet in the view, just discard it.
            if groupidx is None:
                continue
            group = self.get_group(groupidx)
            channel = self.get_channel(channelidx)
            index = self.index(channel.row(), 0, parent=group.index)
            index1 = self.index(channel.row(), 1, parent=group.index)
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
            name=name, color=color)
        return groupitem
        
    def add_group_node(self, **kwargs):
        return self.add_node(item_class=GroupItem, **kwargs)
        
    def remove_group(self, group):
        """Remove an empty group. Raise an error if the group is not empty."""
        groupidx = group.groupidx()
        # check that the group is empty
        if self.get_channels_in_group(groupidx):
            raise ValueError("group %d is not empty, unable to delete it" % \
                    groupidx)
        groups = [g for g in self.get_groups() if g.groupidx() == groupidx]
        if groups:
            group = groups[0]
            self.remove_node(group)
        else:
            log.warn("Group %d does not exist0" % groupidx)
        
    def add_channel(self, parent=None, **kwargs):
        channel = self.add_node(item_class=ChannelItem, parent=parent, 
                            **kwargs)
        return channel
    
    def move_channels(self, sources, target):
        # Get the groupidx if the target is a group,
        if type(target) == GroupItem:
            groupidx = target.groupidx()
            target = None
        # else, if it is a channel, take the corresponding group.
        elif type(target) == ChannelItem:
            groupidx = self.get_groupidx(target.channelidx())
        else:
            return None
            
        # Move channels.
        target_group = self.get_group(groupidx)
        for node in sources:
            self._move_channel(node, target_group, target)
        
        self.update_group_sizes()
    
    def _move_channel(self, channel, parent_target, child_target=None):
        row = channel.row()
        parent_source = channel.parent()
        # Find the row where the channel needs to be inserted.
        if child_target is not None:
            child_target_row = child_target.row()
        else:
            child_target_row = parent_target.rowCount()
        # Begin moving the row.
        canmove = self.beginMoveRows(parent_source.index, row, row,
            parent_target.index, child_target_row)
        if canmove:
            # Create a new channel, clone of the old one.
            channel_new = ChannelItem(parent=parent_target,
                channelidx=channel.channelidx(),
                name=channel.name(),
                color=channel.color())
            # Create the index.
            channel_new.index = self.createIndex(child_target_row, 
                0, channel_new)
            # Insert the new channel.
            parent_target.insertChild(channel_new, child_target_row)
            # Remove the old channel.
            if parent_target == parent_source:
                if child_target_row < row:
                    row += 1
                parent_source.removeChildAt(row)
            else:
                parent_source.removeChild(channel)
            self.endMoveRows()
        
    
    # Drag and drop for moving channels
    # ---------------------------------
    def drag(self, target, sources):
        # Get source ChannelItem nodes.
        source_items = [node for node in self.all_nodes() 
            if (str(node) in sources and type(node) == ChannelItem )]
        # Find the target group.
        if type(target) == GroupItem:
            groupidx = target.groupidx()
        # else, if it is a channel, take the corresponding group.
        elif type(target) == ChannelItem:
            groupidx = self.get_groupidx(target.channelidx())
        # Emit internal signal to let TreeView emit a public signal, and to
        # effectively move the channels.
        self.channelsMoved.emit(np.array([channel.channelidx() 
            for channel in source_items]), groupidx)
    
    def rename_channel_group(self, group, name):
        self.setData(self.index(group.row(), 0), name)
        
    def rename_channel(self, channel, name):
        groupidx = self.get_groupidx(channel.channelidx())
        group = self.get_group(groupidx)
        self.setData(self.index(channel.row(), 0, parent=group.index), name)
        
    def change_group_color(self, group, color):
        self.setData(self.index(group.row(), 2), color)
        
    def change_channel_color(self, channel, color):
        groupidx = self.get_groupidx(channel.channelidx())
        group = self.get_group(groupidx)
        self.setData(self.index(channel.row(), 2, parent=group.index), color)
        
        
    # Getter methods
    # --------------
    def get_groups(self):
        return [group for group in self.get_descendants(self.root_item) \
            if (type(group) == GroupItem)]
        
    def get_group(self, groupidx):
        return [group for group in self.get_descendants(self.root_item) \
            if (type(group) == GroupItem) and \
                (group.groupidx() == groupidx)][0]
        
    def get_channels(self):
        return [channel for channel in self.get_descendants(self.root_item) \
          if (type(channel) == ChannelItem)]
            
    def get_channel(self, channelidx):
        l = [channel for channel in self.get_descendants(self.root_item) \
                  if (type(channel) == ChannelItem) and \
                        (channel.channelidx() == channelidx)]
        if l:
            return l[0]
                
    def get_channels_in_group(self, groupidx):
        group = self.get_group(groupidx)
        return [channel for channel in self.get_descendants(group) \
            if (type(channel) == ChannelItem)]
        
    def get_groupidx(self, channelidx):
        """Return the group index currently assigned to the specifyed channel
        index."""
        for group in self.get_groups():
            channelindices = [channel.channelidx() \
                            for channel in self.get_channels_in_group(group.groupidx())]
            if channelidx in channelindices:
                return group.groupidx()
        return None
          
        
# Top-level widget
# ----------------
class ChannelView(QtGui.QTreeView):
    # Signals
    # -------
    # Selection.
    # The boolean indicates whether the selection has been initiated externally
    # or not (internally by clicking on items in the view).
    channelsSelected = QtCore.pyqtSignal(np.ndarray, bool)
    # groupsSelected = QtCore.pyqtSignal(np.ndarray)
    
    # Channel and group info.
    channelColorChanged = QtCore.pyqtSignal(int, int)
    groupColorChanged = QtCore.pyqtSignal(int, int)
    groupRenamed = QtCore.pyqtSignal(int, object)
    channelRenamed = QtCore.pyqtSignal(int, object)

    channelsMoved = QtCore.pyqtSignal(np.ndarray, int)
    groupRemoved = QtCore.pyqtSignal(int)
    groupAdded = QtCore.pyqtSignal(int, str, int)
    
    class ChannelDelegate(QtGui.QStyledItemDelegate):
        def paint(self, painter, option, index):
            """Disable the color column so that the color remains the same even
            when it is selected."""
            # deactivate all columns except the first one, so that selection
            # is only possible in the first column
            if index.column() >= 1:
                if option.state and QtGui.QStyle.State_Selected:
                    option.state = option.state and QtGui.QStyle.State_Off
            super(ChannelView.ChannelDelegate, self).paint(painter, option, index)
    
    def __init__(self, parent, getfocus=None):
        super(ChannelView, self).__init__(parent)
        # Current item.
        self.current_item = None
        self.wizard = False
        self.channels_selected_previous = []
        
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
        self.setItemDelegate(self.ChannelDelegate())
        
        # Create menu. (hidden because it doesn't work - yet!)
        self.create_actions()
        self.create_context_menu()
        
        self.restore_geometry()
    
    
    # Data methods
    # ------------
    def set_data(self, **kwargs):
        
        # if channel_colors is None:
        if kwargs.get('channel_colors', None) is None:
            return
        
        self.model = ChannelViewModel(**kwargs)
        
        self.setModel(self.model)
        self.expandAll()
        
        # set spkcount column size
        self.header().resizeSection(1, 40)
        self.header().resizeSection(2, 60)
        # # set color column size
        # self.header().resizeSection(3, 40)
        
        # HACK: drag is triggered in the model, so connect it to move_channels
        # in this function
        self.model.channelsMoved.connect(self.move_channels)
        
    def clear(self):
        self.setModel(ChannelViewModel())
        
        
    
    # Public methods
    # --------------
    def select(self, channels, groups=None, wizard=False):
        """Select multiple channels from their indices."""
        self.wizard = wizard
        if channels is None:
            channels = []
        if groups is None:
            groups = []
        if isinstance(channels, (int, long, np.integer)):
            channels = [channels]
        if isinstance(groups, (int, long, np.integer)):
            groups = [groups]
        if len(channels) == len(groups) == 0:
            return
        # Convert to lists.
        channels = list(channels)
        groups = list(groups)
        selection_model = self.selectionModel()
        selection = QtGui.QItemSelection()
        # Select groups.
        for groupidx in groups:
            group = self.model.get_group(groupidx)
            if group is not None:
                selection.select(group.index, group.index)
        # Select channels.
        for channelidx in channels:
            channel = self.model.get_channel(channelidx)
            if channel is not None:
                selection.select(channel.index, channel.index)
        # Process selection.
        selection_model.select(selection, 
                selection_model.Clear |
                selection_model.Current |
                selection_model.Select | 
                selection_model.Rows 
                )
        if len(channels) > 0:
            channel = self.model.get_channel(channels[-1])
            if channel is not None:
                # Set current index in the selection.
                selection_model.setCurrentIndex(
                    channel.index,
                    QtGui.QItemSelectionModel.NoUpdate)
                # Scroll to that channel.
                self.scrollTo(channel.index)
                    
    def unselect(self):
        self.selectionModel().clear()
    
    def move_channels(self, channels, groupidx):
        if not hasattr(channels, '__len__'):
            channels = [channels]
        if len(channels) == 0:
            return
        # Signal.
        log.debug("Moving channels {0:s} to group {1:d}.".format(
            str(channels), groupidx))
        self.channelsMoved.emit(np.array(channels), groupidx)
    
    def add_group(self, name, channels=[]):
        color = random_color()
        group = self.model.add_group(name, color)
        groupidx = group.groupidx()
        # Signal.
        log.debug("Adding group {0:s}.".format(name))
        self.groupAdded.emit(groupidx, name, color)
        # Move the selected channels to the new group.
        if channels:
            self.move_channels(channels, groupidx)
        self.expandAll()
        return groupidx
    
    def rename_channel_group(self, groupidx, name):
        self.model.rename_channel_group(self.model.get_group(groupidx), name)
        # Signal.
        log.debug("Rename group {0:d} to {1:s}.".format(
            groupidx, name))
        self.groupRenamed.emit(groupidx, name)
        
    def rename_channel(self, channelidx, name):
        self.model.rename_channel(self.model.get_channel(channelidx), name)
        # Signal.
        log.debug("Rename channel {0:d} to {1:s}.".format(
            channelidx, name))
        self.channelRenamed.emit(channelidx, name)
        
    def remove_group(self, groupidx):
        self.model.remove_group(self.model.get_group(groupidx))
        # Signal.
        log.debug("Removed group {0:d}.".format(groupidx))
        self.groupRemoved.emit(groupidx)
        
    def change_channel_color(self, channelidx, color):
        self.model.change_channel_color(self.model.get_channel(channelidx), 
            color)
        # Signal.
        log.debug("Changed color of channel {0:d} to {1:d}.".format(
            channelidx, color))
        self.channelColorChanged.emit(channelidx, color)
        
    def change_group_color(self, groupidx, color):
        self.model.change_group_color(self.model.get_group(groupidx), color)
        # Signal.
        log.debug("Changed color of group {0:d} to {1:d}.".format(
            groupidx, color))
        self.groupColorChanged.emit(groupidx, color)
    
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
        
        self.rename_channel_group_action = QtGui.QAction("Re&name group", self)
        self.rename_channel_group_action.setShortcut("F2")
        self.rename_channel_group_action.triggered.connect(self.rename_channel_group_callback)

        self.rename_channel_action = QtGui.QAction("Rename c&hannel", self)
        self.rename_channel_action.setShortcut("F3")
        self.rename_channel_action.triggered.connect(self.rename_channel_callback)
        
        self.remove_group_action = QtGui.QAction("&Remove group", self)
        self.remove_group_action.triggered.connect(self.remove_group_callback)
        
        # Add actions to the widget.
        self.addAction(self.change_color_action)
        self.addAction(self.add_group_action)
        self.addAction(self.rename_channel_group_action)
        self.addAction(self.rename_channel_action)
        self.addAction(self.remove_group_action)

    def create_context_menu(self):
        self.create_color_dialog()
        
        self.context_menu = QtGui.QMenu(self)
        self.context_menu.addAction(self.change_color_action)
        self.context_menu.addSeparator()
        self.context_menu.addAction(self.add_group_action)
        self.context_menu.addAction(self.rename_channel_group_action)
        self.context_menu.addAction(self.rename_channel_action)
        self.context_menu.addAction(self.remove_group_action)
        
    def update_actions(self):
        channels = self.selected_channels()
        groups = self.selected_groups()
        
        if len(groups) > 0:
            self.rename_channel_group_action.setEnabled(True)
            # First three groups are not removable (noise and MUA and good).
            if 0 not in groups and 1 not in groups and 2 not in groups:
                self.remove_group_action.setEnabled(True)
            else:
                self.remove_group_action.setEnabled(False)
        else:
            self.rename_channel_group_action.setEnabled(False)
            self.remove_group_action.setEnabled(False)
            
        if len(channels) > 0:
            self.rename_channel_action.setEnabled(True)
        else:
            self.rename_channel_action.setEnabled(False)
        
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
        if isinstance(item, ChannelItem):
            self.change_channel_color(item.channelidx(), color)
        elif isinstance(item, GroupItem):
            self.change_group_color(item.groupidx(), color)
            
    def add_group_callback(self, checked=None):
        text, ok = QtGui.QInputDialog.getText(self, 
            "Group name", "Name group:",
            QtGui.QLineEdit.Normal, "New group")
        if ok:
            self.add_group(text, self.selected_channels())
        
    def remove_group_callback(self, checked=None):
        item = self.current_item
        if isinstance(item, GroupItem):
            self.remove_group(item.groupidx())
            
    def rename_channel_group_callback(self, checked=None):
        group = self.current_item
        if isinstance(group, GroupItem):
            groupidx = group.groupidx()
            name = group.name()
            text, ok = QtGui.QInputDialog.getText(self, 
                "Group name", "Rename group:",
                QtGui.QLineEdit.Normal, name)
            if ok:
                # Rename the group.
                self.rename_channel_group(groupidx, text)
                
    def rename_channel_callback(self, checked=None):
        channel = self.current_item
        if isinstance(channel, ChannelItem):
            channelidx = channel.channelidx()
            name = channel.name()
            text, ok = QtGui.QInputDialog.getText(self, 
                "Channel name", "Rename channel:",
                QtGui.QLineEdit.Normal, name)
            if ok:
                # Rename the group.
                self.rename_channel(channelidx, text)
    
    # Get methods
    # -----------
    def get_channel_indices(self):
        return [channel.channelidx() for channel in self.model.get_channels()]
    
    def get_group_indices(self):
        return [group.groupidx() for group in self.model.get_groups()]
    
    def get_channel_indices_in_group(self, groupidx):
        return [channel.channelidx() 
            for channel in self.model.get_channels_in_group(groupidx)]
    
    
    # Selection methods
    # -----------------
    def selectionChanged(self, selected, deselected):
        super(ChannelView, self).selectionChanged(selected, deselected)
        selected_channels = self.selected_channels()
        selected_groups = self.selected_groups()
        # All channels in selected groups minus selected channels.
        channels = [channel 
            for group in selected_groups
                for channel in self.get_channel_indices_in_group(group)
                    if channel not in selected_channels]
        # Add selected channels not in selected groups.
        channels.extend([channel
            for channel in selected_channels
                if (channel not in channels and
                    self.model.get_groupidx(channel) not in selected_groups)
            ])
        
        # Selected groups.
        group_indices = [self.model.get_group(groupidx)
            for groupidx in selected_groups]
        
        # log.debug("Selected {0:d} channels.".format(len(channels)))
        # log.debug("Selected channels {0:s}.".format(str(channels)))
        self.channelsSelected.emit(np.array(channels, dtype=np.int32),
            self.wizard)
        
        # if group_indices:
            # self.scrollTo(group_indices[-1].index)
        if len(self.channels_selected_previous) <= 1:
            if len(channels) == 1:
                self.scrollTo(self.model.get_channel(channels[0]).index)
            elif len(group_indices) == 1:
                self.scrollTo(group_indices[0].index)
    
        self.wizard = False
        self.channels_selected_previous = channels
        self.update_actions()
    
    # Selected items
    # --------------
    def selected_items(self):
        """Return the list of selected channel indices."""
        return [(v.internalPointer()) \
                    for v in self.selectedIndexes() \
                        if v.column() == 0]
                            
    def selected_channels(self):
        """Return the list of selected channel indices."""
        return [(v.internalPointer().channelidx()) \
                    for v in self.selectedIndexes() \
                        if v.column() == 0 and \
                           type(v.internalPointer()) == ChannelItem]
              
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
            self.select(self.get_channel_indices())
        else:
            return super(ChannelView, self).keyPressEvent(e)
        
    
    # Save and restore geometry
    # -------------------------
    def save_geometry(self):
        SETTINGS['channel_widget.geometry'] = encode_bytearray(
            self.saveGeometry())
        SETTINGS['channel_widget.header'] = encode_bytearray(
            self.header().saveState())
        
    def restore_geometry(self):
        g = SETTINGS['channel_widget.geometry']
        h = SETTINGS['channel_widget.header']
        if g:
            self.restoreGeometry(decode_bytearray(g))
        if h:
            self.header().restoreState(decode_bytearray(h))
    
    def closeEvent(self, e):
        # Save the window geometry when closing the software.
        self.save_geometry()
        return super(ChannelView, self).closeEvent(e)
        
        