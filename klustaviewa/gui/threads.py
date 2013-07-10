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
# LOCK = Lock()


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------
class OpenTask(QtCore.QObject):
    dataOpened = QtCore.pyqtSignal()
    dataOpenFailed = QtCore.pyqtSignal(str)
    
    def __init__(self, parent=None):
        super(OpenTask, self).__init__(parent)
    
    def open(self, loader, path):
        try:
            loader.open(path)
            self.dataOpened.emit()
        except Exception as e:
            self.dataOpenFailed.emit(e.message)
            

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
        np.ndarray, np.ndarray, object)
    
    def __init__(self, parent=None):
        super(SimilarityMatrixTask, self).__init__(parent)
        
    def compute(self, features, clusters, 
            cluster_groups, masks, clusters_selected, target_next=None,
            similarity_measure=None):
        log.debug("Computing correlation for clusters {0:s}.".format(
            str(list(clusters_selected))))
        if len(clusters_selected) == 0:
            return {}
        correlations = compute_correlations(features, clusters, 
            masks, clusters_selected, similarity_measure=similarity_measure)
        return correlations
        
    def compute_done(self, features, clusters, 
            cluster_groups, masks, clusters_selected, target_next=None,
            similarity_measure=None, _result=None):
        correlations = _result
        self.correlationMatrixComputed.emit(np.array(clusters_selected),
            correlations, 
            get_array(clusters, copy=True), 
            get_array(cluster_groups, copy=True),
            target_next)


# -----------------------------------------------------------------------------
# Container
# -----------------------------------------------------------------------------
class ThreadedTasks(QtCore.QObject):
    def __init__(self, parent=None):
        super(ThreadedTasks, self).__init__(parent)
        self.correlograms_task = inprocess(CorrelogramsTask)(
            impatient=True, use_master_thread=False)
        self.similarity_matrix_task = inprocess(SimilarityMatrixTask)(
            impatient=True, use_master_thread=False)

    def join(self):
        self.correlograms_task.join()
        self.similarity_matrix_task.join()
        
    def terminate(self):
        self.correlograms_task.terminate()
        self.similarity_matrix_task.terminate()
    
        