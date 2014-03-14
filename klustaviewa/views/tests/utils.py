"""Utils for unit tests for views package."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import threading
import time

from qtools import QtGui, QtCore
from qtools import show_window

import mock_data as md
from klustaviewa.stats.correlograms import NCORRBINS_DEFAULT, CORRBIN_DEFAULT
from klustaviewa import USERPREF
import klustaviewa.views as v


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def assert_fun(statement):
    assert statement
    
def get_data():
    """Return a dictionary with data variables, after the fixture setup
    has been called."""
    l = md.LOADER
    
    # Get full data sets.
    clusters_selected = [4, 2, 10]
    l.select(clusters=clusters_selected)
    
    data = dict(
        clusters_selected=clusters_selected,
        features=l.get_features(),
        features_background=l.get_features_background(),
        # features_full=l.get_features('all'),
        masks=l.get_masks(),
        waveforms=l.get_waveforms(),
        clusters=l.get_clusters(),
        
        cluster_colors=l.get_cluster_colors(),
        cluster_colors_full=l.get_cluster_colors('all'),
        
        cluster_groups=l.get_cluster_groups('all'),
        group_colors=l.get_group_colors('all'),
        group_names=l.get_group_names('all'),
        cluster_sizes=l.get_cluster_sizes('all'),
        
        # channel_names=l.get_channel_names(),
        # channel_colors=l.self.kwa(),
        # channel_groups=l.get_channel_groups(),
        # 
        # channel_group_colors=l.get_channel_group_colors(),
        # channel_group_names=l.get_channel_group_names(),
        
        spiketimes=l.get_spiketimes(),
        geometrical_positions=l.get_probe(),
        
        freq=l.freq,
        nchannels=l.nchannels,
        nsamples=l.nsamples,
        fetdim=l.fetdim,
        nextrafet=l.nextrafet,
        ncorrbins=NCORRBINS_DEFAULT,  #l.ncorrbins,
        duration=NCORRBINS_DEFAULT * CORRBIN_DEFAULT,  #l.get_duration(),
    )
    
    return data

    
# -----------------------------------------------------------------------------
# View functions
# -----------------------------------------------------------------------------
def show_view(view_class, **kwargs):
    
    operators = kwargs.pop('operators', None)
    
    # Display a view.
    class TestWindow(QtGui.QMainWindow):
        operatorStarted = QtCore.pyqtSignal(int)
        
        def __init__(self):
            super(TestWindow, self).__init__()
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
            self.setMouseTracking(True)
            self.setWindowTitle("KlustaViewa")
            self.view = view_class(self, getfocus=False)
            self.view.set_data(**kwargs)
            self.setCentralWidget(self.view)
            self.move(100, 100)
            self.show()
            
            # Start "operator" asynchronously in the main thread.
            if operators:
                self.operator_list = operators
                self.operatorStarted.connect(self.operator)
                self._thread = threading.Thread(target=self._run_operator)
                self._thread.start()
            
        def _run_operator(self):
            for i in xrange(len(self.operator_list)):
                # Call asynchronously operation #i, after a given delay.
                if type(self.operator_list[i]) == tuple:
                    dt = self.operator_list[i][1]
                else:
                    # Default delay.
                    dt = USERPREF['test_operator_delay'] or .1
                time.sleep(dt)
                self.operatorStarted.emit(i)
            
        def operator(self, i):
            # Execute operation #i.
            if type(self.operator_list[i]) == tuple:
                fun = self.operator_list[i][0]
            else:
                fun = self.operator_list[i]
            fun(self)
            
        def keyPressEvent(self, e):
            super(TestWindow, self).keyPressEvent(e)
            self.view.keyPressEvent(e)
            if e.key() == QtCore.Qt.Key_Q:
                self.close()
            
        def keyReleaseEvent(self, e):
            super(TestWindow, self).keyReleaseEvent(e)
            self.view.keyReleaseEvent(e)
                
        def closeEvent(self, e):
            if operators:
                self._thread.join()
            return super(TestWindow, self).closeEvent(e)
                
    window = show_window(TestWindow)
    return window
    
# def show_waveformview(loader, clusters, **kwargs):
    # loader.select(clusters=clusters)
    # data = vd.get_waveformview_data(loader)
    # kwargs.update(data)
    # show_view(v.WaveformView, **kwargs)
    
# def show_featureview(loader, clusters, **kwargs):
    # loader.select(clusters=clusters)
    # data = vd.get_featureview_data(loader)
    # kwargs.update(data)
    # show_view(v.FeatureView, **kwargs)
    
    
    