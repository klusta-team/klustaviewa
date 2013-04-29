from galry import QtGui
# from views import *
# import tools
import numpy as np
import numpy.random as rnd
# from dataio import MockDataProvider
# from tools import Info
from collections import OrderedDict
import re
# import klustaviewa.dataio as sdataio
# import klustaviewa.tools as stools
from klustaviewa.utils.settings import SETTINGS


__all__ = ['VisualizationWidget', ]


class VisualizationWidget(QtGui.QWidget):
    def __init__(self, main_window, dataholder):
        super(VisualizationWidget, self).__init__()
        self.dataholder = dataholder
        self.main_window = main_window
        self.view = self.create_view(dataholder)
        self.controller = self.create_controller()
        self.initialize()
        self.initialize_connections()

    def create_view(self, dataholder):
        """Create the view and return it.
        The view must be an instance of a class deriving from `QWidget`.
        
        To be overriden."""
        return None

    def create_controller(self):
        """Create the controller and return it.
        The controller must be an instance of a class deriving from `QLayout`.
        
        To be overriden."""
        
        # horizontal layout for the controller
        hbox = QtGui.QHBoxLayout()
        
        # we add the "isolated" checkbox
        # self.isolated_control = QtGui.QCheckBox("isolated")
        # hbox.addWidget(self.isolated_control, stretch=1, alignment=QtCore.Qt.AlignLeft)
        
        # # add the reset view button
        # self.reset_view_control = QtGui.QPushButton("reset view")
        # hbox.addWidget(self.reset_view_control, stretch=1, alignment=QtCore.Qt.AlignLeft)
        
        # # hbox.addWidget(QtGui.QCheckBox("test"), stretch=1, alignment=QtCore.Qt.AlignLeft)
        # # add lots of space to the right to make sure everything is aligned to 
        # # the left
        # hbox.addStretch(100)
        
        return hbox
        
    def initialize(self):
        """Initialize the user interface.
        
        By default, add the controller at the top, and the view at the bottom.
        
        To be overriden."""
        # put the controller and the view vertically
        vbox = QtGui.QVBoxLayout()
        # add the controller (which must be a layout)
        vbox.addLayout(self.controller)
        # add the view (which must be a widget, typically deriving from
        # GalryWidget)
        vbox.addWidget(self.view)
        # set the VBox as layout of the widget
        self.setLayout(vbox)

    def initialize_connections(self):
        """Initialize signals/slots connections."""
        pass

