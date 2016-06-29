"""Tasks running in external threads or processes."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import time
import sys
import traceback
from threading import Lock

import numpy as np
from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.tools import get_array
from klustaviewa.wizard.wizard import Wizard
from kwiklib.utils import logger as log
from klustaviewa.stats import compute_correlograms, SimilarityMatrix
from recluster import run_klustakwik

# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------
class OpenTask(QtCore.QObject):
    dataOpened = QtCore.pyqtSignal()
    dataSaved = QtCore.pyqtSignal()
    dataOpenFailed = QtCore.pyqtSignal(str)

    def open(self, loader, path, shank=None):
        try:
            loader.close()
            loader.open(path, shank)
            self.dataOpened.emit()
        except Exception as e:
            self.dataOpenFailed.emit(traceback.format_exc())

    def save(self, loader):
        loader.save()
        self.dataSaved.emit()


class SelectionTask(QtCore.QObject):
    selectionDone = QtCore.pyqtSignal(object, bool, int)

    def set_loader(self, loader):
        self.loader = loader

    def select(self, clusters, wizard, channel_group=0):
        self.loader.select(clusters=clusters)

    def select_done(self, clusters, wizard, channel_group=0, _result=None):
        self.selectionDone.emit(clusters, wizard, channel_group)


class ReclusterTask(QtCore.QObject):
    reclusterDone = QtCore.pyqtSignal(int, object, object, object, object)

    def recluster(self, exp, channel_group=0, clusters=None, wizard=None):
        spikes, clu = run_klustakwik(exp, channel_group=channel_group,
                             clusters=clusters)
        return spikes, clu

    def recluster_done(self, exp, channel_group=0, clusters=None, wizard=None, _result=None):
        spikes, clu = _result
        self.reclusterDone.emit(channel_group, clusters, spikes, clu, wizard)


class CorrelogramsTask(QtCore.QObject):
    correlogramsComputed = QtCore.pyqtSignal(np.ndarray, object, int, float, float, object)

    # def __init__(self, parent=None):
        # super(CorrelogramsTask, self).__init__(parent)

    def compute(self, spiketimes, clusters, clusters_to_update=None,
            clusters_selected=None, ncorrbins=None, corrbin=None, sample_rate=None, wizard=None):
        log.debug("Computing correlograms for clusters {0:s}.".format(
            str(list(clusters_to_update))))
        if len(clusters_to_update) == 0:
            return {}
        clusters_to_update = np.array(clusters_to_update, dtype=np.int32)
        correlograms = compute_correlograms(spiketimes, clusters,
            clusters_to_update=clusters_to_update,
            ncorrbins=ncorrbins, corrbin=corrbin, sample_rate=sample_rate)
        return correlograms

    def compute_done(self, spiketimes, clusters, clusters_to_update=None,
            clusters_selected=None, ncorrbins=None, corrbin=None, sample_rate=None, wizard=None, _result=None):
        correlograms = _result
        self.correlogramsComputed.emit(np.array(clusters_selected),
            correlograms, ncorrbins, corrbin, float(sample_rate), wizard)


class SimilarityMatrixTask(QtCore.QObject):
    correlationMatrixComputed = QtCore.pyqtSignal(np.ndarray, object,
        np.ndarray, np.ndarray, object)

    def __init__(self, parent=None):
        super(SimilarityMatrixTask, self).__init__(parent)
        self.sm = None

    def compute(self, features, clusters,
            cluster_groups, masks, clusters_selected, target_next=None,
            similarity_measure=None):
        log.debug("Computing correlation for clusters {0:s}.".format(
            str(list(clusters_selected))))
        if len(clusters_selected) == 0:
            return {}
        if self.sm is None:
            self.sm = SimilarityMatrix(features, masks)
        correlations = self.sm.compute_matrix(clusters, clusters_selected)
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
        self.selection_task = inthread(SelectionTask)(
            impatient=True)
        self.recluster_task = inthread(ReclusterTask)(
            impatient=True)
        self.correlograms_task = inprocess(CorrelogramsTask)(
            impatient=True, use_master_thread=False)
        # HACK: the similarity matrix view does not appear to update on
        # some versions of Mac+Qt, but it seems to work with inthread
        if sys.platform == 'darwin':
            self.similarity_matrix_task = inthread(SimilarityMatrixTask)(
                impatient=True)
        else:
            self.similarity_matrix_task = inprocess(SimilarityMatrixTask)(
                impatient=True, use_master_thread=False)

    def join(self):
        self.selection_task.join()
        self.recluster_task.join()
        self.correlograms_task.join()
        self.similarity_matrix_task.join()

    def terminate(self):
        self.correlograms_task.terminate()
        # The similarity matrix is in an external process only
        # if the system is not a Mac.
        if sys.platform != 'darwin':
            self.similarity_matrix_task.terminate()


