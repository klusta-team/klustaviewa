"""Feature View: show spikes as 2D points in feature space."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import time

import numpy as np
import numpy.random as rdn

from qtools import QtGui, QtCore
    
from kwiklib.utils import logger as log
from klustaviewa.views.featureview import FeatureView
from klustaviewa.views.projectionview import ProjectionView


# -----------------------------------------------------------------------------
# Top-level widget
# -----------------------------------------------------------------------------
class FeatureProjectionView(QtGui.QWidget):
    spikesHighlighted = QtCore.pyqtSignal(np.ndarray)
    spikesSelected = QtCore.pyqtSignal(np.ndarray)
    projectionChanged = QtCore.pyqtSignal(int, int, int)
    
    def __init__(self, parent, getfocus=None):
        super(FeatureProjectionView, self).__init__(parent)
        # Focus policy.
        if getfocus:
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
        else:
            self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setWindowTitle('FeatureView')
        self.create_layout()
        self.show()
    
    def create_layout(self):
        self.projection_view = ProjectionView(self, getfocus=False)
        self.feature_view = FeatureView(self, getfocus=False)
        self.set_data()
        
        # Connect the FeatureView signal to the top-level widget signals.
        self.feature_view.spikesHighlighted.connect(self.spikesHighlighted)
        self.feature_view.spikesSelected.connect(self.spikesSelected)
        
        # Connect the bottom-level projectionChanged signals to the top-level
        # widget signals.
        self.feature_view.projectionChanged.connect(self.projectionChanged)
        self.projection_view.projectionChanged.connect(self.projectionChanged)
        
        # Interconnect the projectionChanged between the two bottom-level widgets.
        self.projection_view.projectionChanged.connect(
            self.projection_changed_projection_callback)
        self.feature_view.projectionChanged.connect(
            self.projection_changed_feature_callback)
        
        box = QtGui.QVBoxLayout()
        # HACK: pyside does not have this function
        if hasattr(box, 'setMargin'):
            box.setMargin(0)
            
        box.addWidget(self.projection_view)
        box.addWidget(self.feature_view)
        
        self.setLayout(box)
    
    def set_data(self, *args, **kwargs):
        fetdim = kwargs.get('fetdim', 3)
        nchannels = kwargs.get('nchannels', 1)
        nextrafet = kwargs.get('nextrafet', 0)
        channels = kwargs.get('channels', None)
        self.projection_view.set_data(fetdim=fetdim, nchannels=nchannels, 
            nextrafet=nextrafet, channels=channels)
        self.feature_view.set_data(*args, **kwargs)
    
    def projection_changed_projection_callback(self, *args):
        self.feature_view.set_projection(*args, do_emit=False)
    
    def projection_changed_feature_callback(self, *args):
        self.projection_view.set_projection(*args, do_emit=False)
    
    
    # FeatureView methods
    # -------------------
    def set_wizard_pair(self, *args, **kwargs):
        return self.feature_view.set_wizard_pair(*args, **kwargs)
    
    def highlight_spikes(self, *args, **kwargs):
        return self.feature_view.highlight_spikes(*args, **kwargs)
    
    def select_spikes(self, *args, **kwargs):
        return self.feature_view.select_spikes(*args, **kwargs)
    
    def toggle_mask(self, *args, **kwargs):
        return self.feature_view.toggle_mask(*args, **kwargs)
    
        
    # ProjectionView methods
    # ----------------------
    def select_feature(self, *args, **kwargs):
        return self.projection_view.select_feature(*args, **kwargs)
    
    def select_channel(self, *args, **kwargs):
        return self.projection_view.select_channel(*args, **kwargs)
    
    def get_projection(self, *args, **kwargs):
        return self.projection_view.get_projection(*args, **kwargs)
    
    def set_projection(self, coord, channel, feature, do_emit=True):
        if feature == -1:
            feature = self.feature_view.projection_manager.get_smart_feature(
                coord, channel)
        self.projection_view.set_projection(coord, channel, 
            feature, do_emit=do_emit)
        self.feature_view.set_projection(coord, channel, 
            feature, do_emit=do_emit)
    
    
    # Event methods
    # -------------
    def keyPressEvent(self, e):
        super(FeatureProjectionView, self).keyPressEvent(e)
        [view.keyPressEvent(e) for view in (self.feature_view, self.projection_view)]
        
    def keyReleaseEvent(self, e):
        super(FeatureProjectionView, self).keyReleaseEvent(e)
        [view.keyReleaseEvent(e) for view in (self.feature_view, self.projection_view)]
    
        