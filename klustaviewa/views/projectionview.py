"""Projection View: GUI for selecting projection."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter
import operator
import time
from functools import partial

import numpy as np
import numpy.random as rdn
from matplotlib.path import Path

from galry import (Manager, PlotPaintManager, PlotInteractionManager, Visual,
    GalryWidget, QtGui, QtCore, show_window, enforce_dtype, RectanglesVisual,
    TextVisual, PlotVisual, AxesVisual)
from klustaviewa.io.selection import get_indices
from klustaviewa.io.tools import get_array
from klustaviewa.views.common import HighlightManager, KlustaViewaBindings
from klustaviewa.utils.colors import COLORMAP, HIGHLIGHT_COLORMAP
import klustaviewa.utils.logger as log
import klustaviewa


# -----------------------------------------------------------------------------
# Projection View
# -----------------------------------------------------------------------------
class ProjectionView(QtGui.QWidget):
    
    projectionChanged = QtCore.pyqtSignal(int, int, int)
    
    def __init__(self, parent, getfocus=None):
        super(ProjectionView, self).__init__(parent)
        # Focus policy.
        if getfocus:
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
        else:
            self.setFocusPolicy(QtCore.Qt.NoFocus)
            
        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle('ProjectionView')
            
        self.show()
        
        
    # Public methods.
    # ---------------
    def set_data(self, fetdim=None, nchannels=None, nextrafet=None):
        self.fetdim = fetdim
        self.nchannels = nchannels
        self.nextrafet = nextrafet
        
        box = self.create_widget()
        self.setLayout(box)    
    
    def select_feature(self, coord, feature):
        channel = self.projection[coord][0]
        
        feature = np.clip(feature, 0, self.fetdim - 1)
        
        self._change_projection(coord, channel, feature)
        
    def select_channel(self, coord, channel):
        channel = str(channel).lower()
        if channel.startswith('extra'):
            channel = channel[6:]
            extra = True
        else:
            extra = False
            
        try:
            channel = int(channel)
        except ValueError:
            log.debug("Unable to parse channel '{0:s}'".format(str(channel)))
            channel = self.projection[coord][0]
            
        if extra:
            channel += self.nchannels
            
        channel = np.clip(channel, 0, self.nchannels + self.nextrafet - 1)
            
        feature = self.projection[coord][1]
        self._change_projection(coord, channel, feature)
    
    def get_projection(self, coord):
        return self.projection[coord]
    
    
    # Widgets.
    # --------
    def create_widget(self):
        
        box = QtGui.QHBoxLayout()
        
        # coord => channel combo box
        self.channel_box = [None, None]
        # coord => (butA, butB, butC)
        self.feature_buttons = [[None] * self.fetdim, [None] * self.fetdim]
        
        # add feature widget
        self.feature_widget1 = self.create_feature_widget(0)
        box.addLayout(self.feature_widget1)
        
        # add feature widget
        self.feature_widget2 = self.create_feature_widget(1)
        box.addLayout(self.feature_widget2)
        
        self.setTabOrder(self.channel_box[0], self.channel_box[1])
        
        return box
    
    def create_feature_widget(self, coord=0):
        # coord => (channel, feature)
        self.projection = [(0, 0), (0, 1)]
        
        gridLayout = QtGui.QGridLayout()
        gridLayout.setSpacing(0)
        # HACK: pyside does not have this function
        if hasattr(gridLayout, 'setMargin'):
            gridLayout.setMargin(0)
        
        # channel selection
        comboBox = QtGui.QComboBox(self)
        comboBox.setEditable(True)
        comboBox.setInsertPolicy(QtGui.QComboBox.NoInsert)
        comboBox.addItems(["%d" % i for i in xrange(self.nchannels)])
        comboBox.addItems(["Extra %d" % i for i in xrange(self.nextrafet)])
        comboBox.editTextChanged.connect(partial(self.select_channel, coord))
        # comboBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.channel_box[coord] = comboBox
        gridLayout.addWidget(comboBox, 0, 0, 1, self.fetdim)
        
        # create 3 buttons for selecting the feature
        widths = [30] * self.fetdim
        labels = ['PC%d' % i for i in xrange(1, self.fetdim + 1)]
        
        # ensure exclusivity of the group of buttons
        pushButtonGroup = QtGui.QButtonGroup(self)
        for i in xrange(len(labels)):
            # selecting feature i
            pushButton = QtGui.QPushButton(labels[i], self)
            pushButton.setCheckable(True)
            if coord == i:
                pushButton.setChecked(True)
            pushButton.setMaximumSize(QtCore.QSize(widths[i], 20))
            pushButton.clicked.connect(partial(self.select_feature, coord, i))
            pushButtonGroup.addButton(pushButton, i)
            self.feature_buttons[coord][i] = pushButton
            gridLayout.addWidget(pushButton, 1, i)
        
        return gridLayout
        
    def update_feature_widget(self):
        for coord in [0, 1]:
            comboBox = self.channel_box[coord]
            # update the channels/features list only if necessary
            if comboBox.count() != self.nchannels + self.nextrafet:
                comboBox.blockSignals(True)
                comboBox.clear()
                comboBox.addItems(["%d" % i for i in xrange(self.nchannels)])
                comboBox.addItems(["Extra %d" % i for i in xrange(self.nextrafet)])
                comboBox.blockSignals(False)
        
        
    # Internal methods.
    # -----------------
    def _change_projection(self, coord, channel, feature):
        assert coord in (0, 1)
        assert (isinstance(channel, (int, long)) and 
            0 <= channel < self.nchannels + self.nextrafet)
        assert (isinstance(feature, (int, long)) and 0 <= feature < self.fetdim)
        
        # Update the widgets.
        self.channel_box[coord].blockSignals(True)
        # update the channel box
        self.channel_box[coord].setCurrentIndex(channel)
        # update the feature button
        if feature < len(self.feature_buttons[coord]):
            self.feature_buttons[coord][feature].setChecked(True)
        self.channel_box[coord].blockSignals(False)
        
        self.projection[coord] = (channel, feature)
        log.debug("Projection changed on coordinate {0:s} to {1:d}:{2:s}".
            format('xy'[coord], channel, 'ABCDEF'[feature]))
        self.projectionChanged.emit(coord, channel, feature)
        
    
    # Slots.
    # ------
    # def slotProjectionChanged(self, sender, coord, channel, feature):
        # """Process the ProjectionChanged signal."""
        
        # if self.view.data_manager.projection is None:
            # return
            
        # # feature == -1 means that it should be automatically selected as
        # # a function of the current projection
        # if feature < 0:
            # # current channel and feature in the other coordinate
            # ch_fet = self.view.data_manager.projection[1 - coord]
            # if ch_fet is not None:
                # other_channel, other_feature = ch_fet
            # else:
                # other_channel, other_feature = 0, 1
            # fetdim = self.fetdim
            # # first dimension: we force to 0
            # if coord == 0:
                # feature = 0
            # # other dimension: 0 if different channel, or next feature if the same
            # # channel
            # else:
                # # same channel case
                # if channel == other_channel:
                    # feature = np.mod(other_feature + 1, fetdim)
                # # different channel case
                # else:
                    # feature = 0
        
        # # print sender
        # log.debug("Projection changed in coord %s, channel=%d, feature=%d" \
            # % (('X', 'Y')[coord], channel, feature))
        # # record the new projection
        # self.projection[coord] = (channel, feature)
        
        # # prevent the channelbox to raise signals when we change its state
        # # programmatically
        # self.channel_box[coord].blockSignals(True)
        # # update the channel box
        # self.channel_box[coord].setCurrentIndex(channel)
        # # update the feature button
        # if feature < len(self.feature_buttons[coord]):
            # self.feature_buttons[coord][feature].setChecked(True)
        # # reactive signals for the channel box
        # self.channel_box[coord].blockSignals(False)
        
        