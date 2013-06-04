"""Tasks graph in the GUI."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd

from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from klustaviewa.dataio.tools import get_array
from klustaviewa.stats.correlations import normalize
from klustaviewa.stats.correlograms import get_baselines
import klustaviewa.utils.logger as log
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.gui.threads import ThreadedTasks, LOCK


# -----------------------------------------------------------------------------
# Abstract task graph
# -----------------------------------------------------------------------------
class AbstractTaskGraph(QtCore.QObject):
    """Graph of successive tasks."""
    def __init__(self):#, **kwargs):
        # for name, value in kwargs.iteritems():
            # setattr(self, name, value)
        pass
        
    def run_single(self, action):
        """Take an action in input, execute it, and return the next action(s).
        """
        if isinstance(action, basestring):
            method = action
            args, kwargs = (), {}
        elif isinstance(action, tuple):
            if len(action) == 1:
                method, = action
                args, kwargs = (), {}
            elif len(action) == 2:
                method, args = action
                kwargs = {}
            elif len(action) == 3:
                method, args, kwargs = action
        else:
            method = None
        if method is not None:
            return getattr(self, method)(*args, **kwargs)
        else:
            return action
    
    def run(self, action_first):
        # Breadth-first search in the task dependency graph.
        queue = [action_first]
        marks = []
        while queue:
            action = queue.pop(0)
            # Execute the first action.
            outputs = self.run_single(action)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for output in outputs:
                if output not in marks:
                    marks.append(output)
                    queue.append(output)
        return outputs
        
    def __getattr__(self, name):
        if not hasattr(self, '_' + name):
            raise ValueError('_' + name)
        return lambda *args, **kwargs: self.run(('_' + name, args, kwargs))
        

# -----------------------------------------------------------------------------
# Specific task graph
# -----------------------------------------------------------------------------
class TaskGraph(AbstractTaskGraph):
    # clustersSelected = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, mainwindow):
        # Shortcuts for the main window.
        self.mainwindow = mainwindow
        self.get_view = self.mainwindow.get_view
        self.get_views = self.mainwindow.get_views
        self.loader = self.mainwindow.loader
        self.statscache = self.mainwindow.statscache
        # Create external threads/processes for long-lasting tasks.
        self.create_threads()
        
    def set(self, mainwindow):
        # Shortcuts for the main window.
        self.mainwindow = mainwindow
        self.get_view = self.mainwindow.get_view
        self.loader = self.mainwindow.loader
        self.statscache = self.mainwindow.statscache
        
    def create_threads(self):
        # Create the external threads.
        self.tasks = ThreadedTasks()
        # self.tasks.open_task.dataOpened.connect(self.open_done)
        # self.tasks.select_task.clustersSelected.connect(self.selection_done)
        self.tasks.correlograms_task.correlogramsComputed.connect(
            self.correlograms_computed_callback)
        self.tasks.similarity_matrix_task.correlationMatrixComputed.connect(
            self.similarity_matrix_computed_callback)
    
    def join(self):
         self.tasks.join()
        

    # Selection.
    # ----------
    def _select(self, clusters):
        self.loader.select(clusters=clusters)
        log.debug("Selected clusters {0:s}.".format(str(clusters)))
        
        self.mainwindow.update_action_enabled()
        
        return [
                ('_update_feature_view',),
                ('_update_waveform_view',),
                ('_show_selection_in_matrix', (clusters,)),
                ('_compute_correlograms', (clusters,),),
                ]
    
    
    # Computations.
    # -------------
    def correlograms_computed_callback(self, clusters, correlograms, ncorrbins, 
            corrbin):
        # Execute the callback function under the control of the task manager
        # (which handles the graph dependency).
        self.correlograms_computed(clusters, correlograms, ncorrbins, corrbin)
        
    def similarity_matrix_computed_callback(self, clusters_selected, matrix, 
        clusters, cluster_groups):
        # Execute the callback function under the control of the task manager
        # (which handles the graph dependency).
        self.similarity_matrix_computed(clusters_selected, matrix, clusters,
            cluster_groups)
            
    def _compute_correlograms(self, clusters_selected):
        # Get the correlograms parameters.
        spiketimes = get_array(self.loader.get_spiketimes('all'))
        # Make a copy of the array so that it does not change before the
        # computation of the correlograms begins.
        clusters = np.array(get_array(self.loader.get_clusters('all')))
        corrbin = self.loader.corrbin
        ncorrbins = self.loader.ncorrbins
        
        # Get cluster indices that need to be updated.
        clusters_to_update = (self.statscache.correlograms.
            not_in_key_indices(clusters_selected))
            
        # If there are pairs that need to be updated, launch the task.
        if len(clusters_to_update) > 0:
            # Set wait cursor.
            self.mainwindow.set_busy_cursor()
            # Launch the task.
            self.tasks.correlograms_task.compute(spiketimes, clusters,
                clusters_to_update=clusters_to_update, 
                clusters_selected=clusters_selected,
                ncorrbins=ncorrbins, corrbin=corrbin)    
        # Otherwise, update directly the correlograms view without launching
        # the task in the external process.
        else:
            # self.update_correlograms_view()
            return '_update_correlograms_view'
    
    def _compute_similarity_matrix(self, clusters_to_update=None):
        # Set wait cursor.
        # self.set_cursor(QtCore.Qt.BusyCursor)
        # Get the correlation matrix parameters.
        features = get_array(self.loader.get_features('all'))
        masks = get_array(self.loader.get_masks('all', full=True))
        clusters = get_array(self.loader.get_clusters('all'))
        cluster_groups = get_array(self.loader.get_cluster_groups('all'))
        clusters_all = self.loader.get_clusters_unique()
        # Get cluster indices that need to be updated.
        if clusters_to_update is None:
            clusters_to_update = (self.statscache.similarity_matrix.
                not_in_key_indices(clusters_all))
        # If there are pairs that need to be updated, launch the task.
        if len(clusters_to_update) > 0:
            # Launch the task.
            self.tasks.similarity_matrix_task.compute(features,
                clusters, cluster_groups, masks, clusters_to_update)
        # Otherwise, update directly the correlograms view without launching
        # the task in the external process.
        else:
            # self.update_similarity_matrix_view()
            return '_update_similarity_matrix_view'
    
    def _correlograms_computed(self, clusters, correlograms, ncorrbins, corrbin):
        clusters_selected = self.loader.get_clusters_selected()
        # Abort if the selection has changed during the computation of the
        # correlograms.
        # Reset the cursor.
        self.mainwindow.set_normal_cursor()
        if not np.array_equal(clusters, clusters_selected):
            log.debug("Skip update correlograms with clusters selected={0:s}"
            " and clusters updated={1:s}.".format(clusters_selected, clusters))
            return
        if self.statscache.ncorrbins != ncorrbins:
            log.debug(("Skip updating correlograms because ncorrbins has "
                "changed (from {0:d} to {1:d})".format(
                ncorrbins, self.statscache.ncorrbins)))
            return
        # Put the computed correlograms in the cache.
        self.statscache.correlograms.update(clusters, correlograms)
        # Update the view.
        # self.update_correlograms_view()
        return '_update_correlograms_view'
        
    def _similarity_matrix_computed(self, clusters_selected, matrix, clusters,
            cluster_groups):
        self.statscache.similarity_matrix.update(clusters_selected, matrix)
        self.statscache.similarity_matrix_normalized = normalize(
            self.statscache.similarity_matrix.to_array(copy=True))
        # Update the cluster view with cluster quality.
        quality = np.diag(self.statscache.similarity_matrix_normalized)
        self.statscache.cluster_quality = pd.Series(
            quality,
            index=self.statscache.similarity_matrix.indices,
            )
        self.get_view('ClusterView').set_quality(
            self.statscache.cluster_quality)
        # Update the wizard.
        # self.update_wizard(clusters_selected, clusters)
        # self.tasks.wizard_task.set_data(
            # clusters=clusters,
            # cluster_groups=cluster_groups,
            # similarity_matrix=self.statscache.similarity_matrix_normalized,
            # )
        # Update the view.
        # self.update_similarity_matrix_view()
        return ('_update_similarity_matrix_view',)


    # View updates.
    # -------------
    def _update_correlograms_view(self):
        clusters_selected = self.loader.get_clusters_selected()
        correlograms = self.statscache.correlograms.submatrix(
            clusters_selected)
        # Compute the baselines.
        sizes = get_array(self.loader.get_cluster_sizes())
        duration = self.loader.get_duration()
        corrbin = self.loader.corrbin
        baselines = get_baselines(sizes, duration, corrbin)
        data = dict(
            correlograms=correlograms,
            baselines=baselines,
            clusters_selected=clusters_selected,
            cluster_colors=self.loader.get_cluster_colors(),
            ncorrbins=self.loader.ncorrbins,
            corrbin=self.loader.corrbin,
        )
        [view.set_data(**data) for view in self.get_views('CorrelogramsView')]
        
    def _update_similarity_matrix_view(self):
        if self.statscache is None:
            return
        # matrix = self.statscache.similarity_matrix
        similarity_matrix = self.statscache.similarity_matrix_normalized
        # Clusters in groups 0 or 1 to hide.
        cluster_groups = self.loader.get_cluster_groups('all')
        clusters_hidden = np.nonzero(np.in1d(cluster_groups, [0, 1]))[0]
        # Cluster quality.
        # similarity_matrix = normalize(matrix.to_array(copy=True))
        # cluster_quality = np.diag(similarity_matrix)
        data = dict(
            # WARNING: copy the matrix here so that we don't modify the
            # original matrix while normalizing it.
            similarity_matrix=similarity_matrix,
            cluster_colors_full=self.loader.get_cluster_colors('all'),
            clusters_hidden=clusters_hidden,
        )
        [view.set_data(**data) 
            for view in self.get_views('SimilarityMatrixView')]
        
    def _update_feature_view(self):
        data = dict(
            features=self.loader.get_some_features(),
            masks=self.loader.get_masks(),
            clusters=self.loader.get_clusters(),
            clusters_selected=self.loader.get_clusters_selected(),
            cluster_colors=self.loader.get_cluster_colors(),
            nchannels=self.loader.nchannels,
            fetdim=self.loader.fetdim,
            nextrafet=self.loader.nextrafet,
            freq=self.loader.freq,
            autozoom=False,  # TODO
            duration=self.loader.get_duration(),
            alpha_selected=USERPREF.get('feature_selected_alpha', .75),
            alpha_background=USERPREF.get('feature_background_alpha', .1),
            time_unit=USERPREF['features_info_time_unit'] or 'second',
        )
        [view.set_data(**data) for view in self.get_views('FeatureView')]
        
    def _update_waveform_view(self):
        data = dict(
            waveforms=self.loader.get_waveforms(),
            clusters=self.loader.get_clusters(),
            cluster_colors=self.loader.get_cluster_colors(),
            clusters_selected=self.loader.get_clusters_selected(),
            masks=self.loader.get_masks(),
            geometrical_positions=self.loader.get_probe(),
            autozoom=False,  # TODO
        )
        [view.set_data(**data) for view in self.get_views('WaveformView')]
        
    def _update_cluster_view(self):
        """Update the cluster view using the data stored in the loader
        object."""
        data = dict(
            cluster_colors=self.loader.get_cluster_colors('all',
                can_override=False),
            cluster_groups=self.loader.get_cluster_groups('all'),
            group_colors=self.loader.get_group_colors('all'),
            group_names=self.loader.get_group_names('all'),
            cluster_sizes=self.loader.get_cluster_sizes('all'),
            cluster_quality=self.statscache.cluster_quality,
        )
        self.get_view('ClusterView').set_data(**data)
    
    def _update_projection_view(self):
        """Update the cluster view using the data stored in the loader
        object."""
        data = dict(
            nchannels=self.loader.nchannels,
            fetdim=self.loader.fetdim,
            nextrafet=self.loader.nextrafet,
        )
        self.get_view('ProjectionView').set_data(**data)
        
    def _show_selection_in_matrix(self, clusters):
        if clusters is not None and 1 <= len(clusters) <= 2:
            self.get_view('SimilarityMatrixView').show_selection(
                clusters[0], clusters[-1])
        
            
    # Override colors.
    # ----------------
    def _override_color(self, override_color):
        self.loader.set_override_color(override_color)
        return ['_update_feature_view', '_update_waveform_view', '_update_correlograms_view']
    
    
    # Change correlograms parameter.
    # ------------------------------
    def _change_correlograms_parameters(self, ncorrbins=None, corrbin=None):
        # Update the correlograms parameters.
        if ncorrbins is not None:
            self.loader.ncorrbins = ncorrbins
        if corrbin is not None:
            self.loader.corrbin = corrbin
        # Reset the cache.
        self.statscache.reset(self.loader.ncorrbins)
        # Update the correlograms.
        clusters = self.loader.get_clusters_selected()
        return ('_compute_correlograms', (clusters,))
    
    
    
    
    
    
    
    
    
    