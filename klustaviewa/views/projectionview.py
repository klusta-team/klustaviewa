"""Projection View: GUI for selecting projection."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import operator
import time
from functools import partial

import numpy as np
import numpy.random as rdn

from qtools import QtGui, QtCore, show_window
from galry import (Manager, PlotPaintManager, PlotInteractionManager, Visual,
    GalryWidget, enforce_dtype, RectanglesVisual,
    TextVisual, PlotVisual, AxesVisual)
from kwiklib.dataio.selection import get_indices
from kwiklib.dataio.tools import get_array
from klustaviewa.gui.icons import get_icon
from klustaviewa.views.common import HighlightManager, KlustaViewaBindings
from kwiklib.utils import logger as log
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
            
        self.setWindowTitle('ProjectionView')
            
        self.show()
        
        
    # Public methods.
    # ---------------
    def _has_changed(self, fetdim, nchannels, nextrafet):
        return (fetdim != getattr(self, 'fetdim', None) or
                nchannels != getattr(self, 'nchannels', None) or
                nextrafet != getattr(self, 'nextrafet', None))
    
    def set_data(self, fetdim=None, nchannels=None, nextrafet=None, channels=None):
        if fetdim is None:
            fetdim = 3
        if nchannels is None:
            nchannels = 1
        if nextrafet is None:
            nextrafet = 0
        if channels is None:
            channels = range(nchannels)
        
        # No need to update the widget if the data has not changed.
        if not self._has_changed(fetdim, nchannels, nextrafet):
            return
        
        self.fetdim = fetdim
        self.nchannels = nchannels
        self.channels = channels
        self.nextrafet = nextrafet
        
        # Remove the existing layout.
        if self.layout():
            QtGui.QWidget().setLayout(self.layout())
        
        self.box = self.create_widget()
        self.setLayout(self.box)    
    
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
            # absolute to relative indexing
            channel = list(self.channels).index(channel)
        except ValueError:
            log.debug("Unable to parse channel '{0:s}'".format(str(channel)))
            channel = self.projection[coord][0]
            
        if extra:
            channel += self.nchannels
            
        # now, channel is relative.
        channel = np.clip(channel, 0, self.nchannels + self.nextrafet - 1)
            
        feature = self.projection[coord][1]
        self._change_projection(coord, channel, feature)
    
    def get_projection(self, coord):
        return self.projection[coord]
    
    def set_projection(self, coord, channel, feature, do_emit=True):
        channel = np.clip(channel, 0, self.nchannels + self.nextrafet - 1)
        feature = np.clip(feature, 0, self.fetdim - 1)
        self._change_projection(coord, channel, feature, do_emit=do_emit)
        
    
    # Widgets.
    # --------
    def create_widget(self):
        
        box = QtGui.QHBoxLayout()
        if hasattr(box, 'setMargin'):
            box.setContentsMargins(QtCore.QMargins(10, 2, 10, 2))
        
        # box.addSpacing(10)
        
        # coord => channel combo box
        self.channel_box = [None, None]
        # coord => (butA, butB, butC)
        self.feature_buttons = [[None] * self.fetdim, [None] * self.fetdim]
        
        # add feature widget
        self.feature_widget1 = self.create_feature_widget(0)
        box.addLayout(self.feature_widget1)
        
        box.addSpacing(10)
        
        # Switch button.
        # button = QtGui.QPushButton('Flip', self)
        button = QtGui.QPushButton(self)
        button.setIcon(get_icon('flip'))
        button.setMaximumWidth(40)
        button.clicked.connect(self.flip_projections_callback)
        box.addWidget(button)
        
        box.addSpacing(10)
        
        # add feature widget
        self.feature_widget2 = self.create_feature_widget(1)
        box.addLayout(self.feature_widget2)
        
        # box.addSpacing(10)
        
        self.setTabOrder(self.channel_box[0], self.channel_box[1])
        
        # self.setMaximumWidth(300)
        # self.setMaximumHeight(80)
        
        
        return box
    
    def create_feature_widget(self, coord=0):
        # coord => (channel, feature)
        self.projection = [(0, 0), (0, 1)]
        
        hbox = QtGui.QHBoxLayout()
        hbox.setSpacing(0)
        # HACK: pyside does not have this function
        if hasattr(hbox, 'setMargin'):
            hbox.setMargin(0)
        
        # channel selection
        comboBox = QtGui.QComboBox(self)
        comboBox.setEditable(True)
        comboBox.setMaximumWidth(100)
        comboBox.setInsertPolicy(QtGui.QComboBox.NoInsert)
        comboBox.addItems(["%d" % i for i in self.channels])
        comboBox.addItems(["Extra %d" % i for i in xrange(self.nextrafet)])
        comboBox.editTextChanged.connect(partial(self.select_channel, coord))
        # comboBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.channel_box[coord] = comboBox
        hbox.addWidget(comboBox)
        
        # create 3 buttons for selecting the feature
        widths = [30] * self.fetdim
        labels = ['PC%d' % i for i in xrange(1, self.fetdim + 1)]
        
        hbox.addSpacing(10)
        
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
            hbox.addWidget(pushButton)
        
        return hbox
        
    def update_feature_widget(self):
        for coord in [0, 1]:
            comboBox = self.channel_box[coord]
            # update the channels/features list only if necessary
            if comboBox.count() != self.nchannels + self.nextrafet:
                comboBox.blockSignals(True)
                comboBox.clear()
                comboBox.addItems(["%d" % i for i in self.channels])
                comboBox.addItems(["Extra %d" % i for i in xrange(self.nextrafet)])
                comboBox.blockSignals(False)
        
        
    # Internal methods.
    # -----------------
    def _change_projection(self, coord, channel, feature, do_emit=True):
        assert coord in (0, 1)
        assert isinstance(channel, (int, long, np.integer))# and 
            # 0 <= channel < self.nchannels + self.nextrafet)
        assert isinstance(feature, (int, long, np.integer))# and 0 <= feature < self.fetdim)
        
        # coord = np.clip(coord, 0, 1)
        feature = np.clip(feature, 0, self.fetdim - 1)
        channel = np.clip(channel, 0, self.nchannels + self.nextrafet - 1)
        
        
        # Update the widgets.
        self.channel_box[coord].blockSignals(True)
        # update the channel box
        self.channel_box[coord].setCurrentIndex(channel)
        # update the feature button
        if feature < len(self.feature_buttons[coord]):
            self.feature_buttons[coord][feature].setChecked(True)
        self.channel_box[coord].blockSignals(False)
        
        self.projection[coord] = (channel, feature)
        # log.debug("Projection changed on coordinate {0:s} to {1:d}:{2:s}".
            # format('xy'[coord], channel, 'ABCDEF'[feature]))
        if do_emit:
            self.projectionChanged.emit(coord, channel, feature)
        
    def flip_projections_callback(self, checked=None):
        c0, f0 = self.projection[0]
        c1, f1 = self.projection[1]
        self.set_projection(0, c1, f1)
        self.set_projection(1, c0, f0)
        
    def maximumSize(self):
        return QtCore.QSize(300, 40)
        