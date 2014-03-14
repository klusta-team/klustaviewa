"""Main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import time
import os
import inspect
from collections import OrderedDict
from functools import partial

import pandas as pd
import numpy as np
import numpy.random as rnd
from qtools import QtGui, QtCore
from qtools import inprocess, inthread, QT_BINDING

import klustaviewa.views as vw
from klustaviewa.gui.icons import get_icon
from klustaviewa.control.controller import Controller
from kwiklib.dataio.tools import get_array
from kwiklib.dataio import KlustersLoader
from klustaviewa.gui.buffer import Buffer
from klustaviewa.stats.cache import StatsCache
from klustaviewa.stats.correlations import normalize
from kwiklib.utils import logger as log
from kwiklib.utils.logger import FileLogger, register, unregister
from kwiklib.utils.persistence import encode_bytearray, decode_bytearray
from klustaviewa import USERPREF
from klustaviewa import SETTINGS
from klustaviewa import APPNAME, ABOUT
from klustaviewa.gui.threads import ThreadedTasks
import rcicons


# -----------------------------------------------------------------------------
# Dock widget class
# -----------------------------------------------------------------------------
class ViewDockWidget(QtGui.QDockWidget):
    closed = QtCore.pyqtSignal(object)
    def __init__(self, parent):
        super(ViewDockWidget, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
    
    def closeEvent(self, e):
        self.closed.emit(self)
        super(ViewDockWidget, self).closeEvent(e)
    
    def keyPressEvent(self, e):
        super(ViewDockWidget, self).keyPressEvent(e)
        # Notify the main window of the key events when the dock widget
        # is floating.
        if self.isFloating():
            self.parent().keyPressEvent(e)
        
    def keyReleaseEvent(self, e):
        super(ViewDockWidget, self).keyReleaseEvent(e)
        # Notify the main window of the key events when the dock widget
        # is floating.
        if self.isFloating():
            self.parent().keyReleaseEvent(e)
    
        

# -----------------------------------------------------------------------------
# Title bar for dock widgets
# -----------------------------------------------------------------------------
DOCKSTYLESHEET = """

QToolButton {
    margin: 0px 2px;
    padding: 2px;
    border: 0;
}

QToolButton:checked {
    background-color: #606060;
}


"""

class DockTitleBar(QtGui.QWidget):
    def __init__(self, parent=None, name=''):
        super(DockTitleBar, self).__init__(parent)
        self.name = name
        self.create_buttons()
        self.create_layout()
        self.show()
        
    def is_floatable(self):
        return self.parent().features() & QtGui.QDockWidget.DockWidgetFloatable
        
    def is_closable(self):
        return self.parent().features() & QtGui.QDockWidget.DockWidgetClosable
    
        
    # Layout.
    # -------
    def add_button(self, name, text, callback=None, shortcut=None,
            checkable=False, icon=None):
        # Creation action.
        action = QtGui.QAction(text, self)
        if callback is None:
            callback = getattr(self, name + '_callback', None)
        if callback:
            action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        if icon:
            action.setIcon(get_icon(icon))
        action.setCheckable(checkable)
        # Create button
        button = QtGui.QToolButton(self)
        button.setContentsMargins(*((5,)*4))
        button.setDefaultAction(action)
        return button
    
    def create_buttons(self):
        if self.is_floatable():
            self.dockable_button = self.add_button('dockable', 
                'Pin/Unpin', icon='pin', checkable=True)
            self.dock_button = self.add_button('dock', 
                'Dock/Undock', icon='dockable')
            self.maximize_button = self.add_button('maximize', 
                'Maximize', icon='fullscreen')
        if self.is_closable():
            self.close_button = self.add_button('close', 
                'Close', icon='close')
    
    def create_layout(self):
        
        self.setStyleSheet(DOCKSTYLESHEET)
        
        # Create the title layout.
        self.setContentsMargins(0, 0, 0, 0)
        box = QtGui.QHBoxLayout()
        box.setContentsMargins(0, 2, 0, 2)
        box.setSpacing(0)
        
        # Add the title.
        self.title_widget = QtGui.QLabel(self.name, self)
        box.addSpacing(5)
        box.addWidget(self.title_widget)
        
        # Add spacing.
        box.addStretch(1000)
        
        # Add the dock-related buttons.
        if self.is_floatable():
            box.addWidget(self.dockable_button)
            box.addWidget(self.maximize_button)
            box.addWidget(self.dock_button)
            
        # Add the close button.
        if self.is_closable():
            box.addWidget(self.close_button)
        
        self.setLayout(box)
    
    
    # Callbacks.
    # ----------
    def dockable_callback(self, checked=None):
        if checked is False:
            self.parent().setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        else:
            self.parent().setAllowedAreas(QtCore.Qt.NoDockWidgetArea)
        
    def dock_callback(self, checked=None):
        self.parent().setFloating(not(self.parent().isFloating()))
        
    def maximize_callback(self, checked=None):
        # Force floating.
        if not self.parent().isFloating():
            self.parent().setFloating(True)
            self.forced_float = True
        # and cancel floating for minimize if it was triggered by maximize
        # button.
        elif self.forced_float:
            self.forced_float = False
            self.parent().showNormal()
            self.parent().setFloating(False)
        if self.parent().isMaximized():
            self.parent().showNormal()
        else:
            self.parent().showMaximized()
        
    def close_callback(self, checked=None):
        self.parent().close()
    
    
    # Size.
    # -----
    # def sizeHint(self):
        # return QtCore.QSize(200, 24)
        
    def minimumSizeHint(self):
        return QtCore.QSize(50, 24)
    
        
