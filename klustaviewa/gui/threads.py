"""Tasks running in external threads or processes."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import time
from threading import Lock

import numpy as np
from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from klustaviewa.dataio import KlustersLoader
from klustaviewa.dataio.tools import get_array
from klustaviewa.wizard.wizard import Wizard
import klustaviewa.utils.logger as log
from klustaviewa.stats import compute_correlograms, compute_correlations


# -----------------------------------------------------------------------------
# Synchronisation
# -----------------------------------------------------------------------------
LOCK = Lock()


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------
class OpenTask(QtCore.QObject):
    dataOpened = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super(OpenTask, self).__init__(parent)
    
    def open(self, loader, path):
        loader.open(path)
        self.dataOpened.emit()

        
class SelectTask(QtCore.QObject):
    clustersSelected = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, parent=None):
        super(SelectTask, self).__init__(parent)
    
    def select(self, loader, clusters):
        # This delay makes the interface smoother and reduces the risk of 
        # thread-induced bugs when selecting lots of different clusters
        # quickly.
        # time.sleep(.05)
        with LOCK:
            loader.select(clusters=clusters)
        log.debug("Selected clusters {0:s}.".format(str(clusters)))
        self.clustersSelected.emit(np.array(clusters))
        
        
class CorrelogramsTask(QtCore.QObject):
    correlogramsComputed = QtCore.pyqtSignal(np.ndarray, object, int, float)
    
    def __init__(self, parent=None):
        super(CorrelogramsTask, self).__init__(parent)
    
    def compute(self, spiketimes, clusters, clusters_to_update=None,
            clusters_selected=None, ncorrbins=None, corrbin=None):
        log.debug("Computing correlograms for clusters {0:s}.".format(
            str(list(clusters_to_update))))
        if len(clusters_to_update) == 0:
            return {}
        clusters_to_update = np.array(clusters_to_update, dtype=np.int32)
        correlograms = compute_correlograms(spiketimes, clusters,
            clusters_to_update=clusters_to_update,
            ncorrbins=ncorrbins, corrbin=corrbin)
        return correlograms
        
    def compute_done(self, spiketimes, clusters, clusters_to_update=None,
            clusters_selected=None, ncorrbins=None, corrbin=None, _result=None):
        correlograms = _result
        self.correlogramsComputed.emit(np.array(clusters_selected),
            correlograms, ncorrbins, corrbin)

            
class SimilarityMatrixTask(QtCore.QObject):
    correlationMatrixComputed = QtCore.pyqtSignal(np.ndarray, object,
        np.ndarray)
    
    def __init__(self, parent=None):
        super(SimilarityMatrixTask, self).__init__(parent)
        
    def compute(self, features, clusters, masks, clusters_selected):
        log.debug("Computing correlation for clusters {0:s}.".format(
            str(list(clusters_selected))))
        if len(clusters_selected) == 0:
            return {}
        correlations = compute_correlations(features, clusters, masks, 
            clusters_selected)
        return correlations
        
    def compute_done(self, features, clusters, masks, clusters_selected,
        _result=None):
        correlations = _result
        self.correlationMatrixComputed.emit(np.array(clusters_selected),
            correlations, get_array(clusters, copy=True))
            

class WizardTask(QtCore.QObject):
    def __init__(self, parent=None):
        super(WizardTask, self).__init__(parent)
        self.wizard = Wizard()
        
    def set_data(self, **kwargs):
        self.wizard.set_data(**kwargs)
        
    def previous(self):
        return self.wizard.previous()
        
    def next(self):
        return self.wizard.next()
        
    def previous_cluster(self):
        return self.wizard.previous_cluster()
        
    def next_cluster(self):
        return self.wizard.next_cluster()


# -----------------------------------------------------------------------------
# Container
# -----------------------------------------------------------------------------
class ThreadedTasks(QtCore.QObject):
    def __init__(self, parent=None):
        super(ThreadedTasks, self).__init__(parent)
        # In external threads.
        self.open_task = inthread(OpenTask)()
        self.select_task = inthread(SelectTask)(impatient=True)
        # In external processes.
        self.correlograms_task = inprocess(CorrelogramsTask)(impatient=True)
        self.similarity_matrix_task = inprocess(SimilarityMatrixTask)(
            impatient=True)
        self.wizard_task = inprocess(WizardTask)(impatient=False)

    def join(self):
        self.open_task.join()
        self.select_task.join()
        self.correlograms_task.join()
        self.similarity_matrix_task.join()
        self.wizard_task.join()
        
    def terminate(self):
        self.correlograms_task.terminate()
        self.similarity_matrix_task.terminate()
        self.wizard_task.terminate()
    
    
        