"""Tasks graph in the GUI."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from kwiklib.dataio import get_array, pandaize
from klustaviewa.stats.correlations import normalize
from klustaviewa.stats.correlograms import get_baselines, get_excerpts
from kwiklib.utils import logger as log
from klustaviewa import USERPREF
from klustaviewa import SETTINGS
from kwiklib.utils.colors import random_color
from klustaviewa.gui.threads import ThreadedTasks
import klustaviewa.views.viewdata as vd


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
        # print method
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
    def __init__(self, mainwindow):
        # Shortcuts for the main window.
        self.set(mainwindow)
        # Create external threads/processes for long-lasting tasks.
        self.create_threads()

    def set(self, mainwindow):
        # Shortcuts for the main window.
        self.mainwindow = mainwindow
        self.get_view = self.mainwindow.get_view
        self.get_views = self.mainwindow.get_views
        self.loader = self.mainwindow.loader
        self.experiment = self.loader.experiment
        self.wizard = self.mainwindow.wizard
        self.controller = self.mainwindow.controller
        self.statscache = self.mainwindow.statscache

    def create_threads(self):
        # Create the external threads.
        self.tasks = ThreadedTasks()
        self.tasks.selection_task.set_loader(self.loader)
        self.tasks.selection_task.selectionDone.connect(
            self.selection_done_callback)
        self.tasks.recluster_task.reclusterDone.connect(
            self.recluster_done_callback)
        self.tasks.correlograms_task.correlogramsComputed.connect(
            self.correlograms_computed_callback)
        self.tasks.similarity_matrix_task.correlationMatrixComputed.connect(
            self.similarity_matrix_computed_callback)

    def join(self):
         self.tasks.join()


    # Selection.
    # ----------
    def _select(self, clusters, wizard=False,):
        self.tasks.selection_task.select(clusters, wizard,)

    def _select_done(self, clusters, wizard=False,):
        if wizard:
            target = (self.wizard.current_target(),)
        else:
            target = ()
        # self.loader.select(clusters=clusters)
        log.debug("Selected clusters {0:s}.".format(str(clusters)))
        return [
                ('_update_feature_view', target, dict()),
                ('_update_waveform_view', (), dict(wizard=wizard,)),
                ('_show_selection_in_matrix', (clusters,),),
                ('_compute_correlograms', (clusters,), dict(wizard=wizard,)),
                ]

    def _select_in_cluster_view(self, clusters, groups=[], wizard=False):
        self.get_view('ClusterView').select(clusters, groups=groups,
            wizard=wizard)


    # Callbacks.
    # ----------
    def selection_done_callback(self, clusters, wizard,):
        self.select_done(clusters, wizard=wizard,)

    def recluster_done_callback(self, channel_group, clusters, spikes, clu, wizard):
        self.recluster_done(channel_group=channel_group,
                            clusters=clusters,
                            spikes=spikes, clu=clu, wizard=wizard)

    def correlograms_computed_callback(self, clusters, correlograms, ncorrbins,
            corrbin, sample_rate, wizard):
        # Execute the callback function under the control of the task manager
        # (which handles the graph dependency).
        self.correlograms_computed(clusters, correlograms, ncorrbins, corrbin, sample_rate, wizard)

    def similarity_matrix_computed_callback(self, clusters_selected, matrix,
        clusters, cluster_groups, target_next=None):
        # Execute the callback function under the control of the task manager
        # (which handles the graph dependency).
        self.similarity_matrix_computed(clusters_selected, matrix, clusters,
            cluster_groups, target_next=target_next)


    # Computations.
    # -------------
    def _compute_correlograms(self, clusters_selected, wizard=None):
        # Get the correlograms parameters.
        spiketimes = get_array(self.loader.get_spiketimes('all'))
        sample_rate = self.loader.freq
        # print spiketimes.dtype
        # Make a copy of the array so that it does not change before the
        # computation of the correlograms begins.
        clusters = np.array(get_array(self.loader.get_clusters('all')))

        # Get excerpts
        nexcerpts = USERPREF.get('correlograms_nexcerpts', 50)
        excerpt_size = USERPREF.get('correlograms_excerpt_size', 10000)
        spiketimes_excerpts = get_excerpts(spiketimes,
            nexcerpts=nexcerpts, excerpt_size=excerpt_size)
        clusters_excerpts = get_excerpts(clusters,
            nexcerpts=nexcerpts, excerpt_size=excerpt_size)

        # corrbin = self.loader.corrbin
        # ncorrbins = self.loader.ncorrbins
        corrbin = SETTINGS.get('correlograms.corrbin', .001)
        ncorrbins = SETTINGS.get('correlograms.ncorrbins', 101)

        # Ensure ncorrbins is odd.
        if ncorrbins % 2 == 0:
            ncorrbins += 1

        # Get cluster indices that need to be updated.
        # clusters_to_update = (self.statscache.correlograms.
        #     not_in_key_indices(clusters_selected))
        clusters_to_update = clusters_selected

        # If there are pairs that need to be updated, launch the task.
        if len(clusters_to_update) > 0:
            # Set wait cursor.
            self.mainwindow.set_busy(computing_correlograms=True)
            # Launch the task.
            self.tasks.correlograms_task.compute(
                spiketimes_excerpts,
                clusters_excerpts,
                clusters_to_update=clusters_to_update,
                clusters_selected=clusters_selected,
                ncorrbins=ncorrbins, corrbin=corrbin,
                sample_rate=sample_rate,
                wizard=wizard,
            )
        # Otherwise, update directly the correlograms view without launching
        # the task in the external process.
        else:
            # self.update_correlograms_view()
            return ('_update_correlograms_view', (wizard,), {})

    def _recluster(self):
        exp = self.loader.experiment
        channel_group = self.loader.shank
        clusters_selected = self.loader.get_clusters_selected()
        self.tasks.recluster_task.recluster(exp, channel_group=channel_group,
                             clusters=clusters_selected)

    def _recluster_done(self, channel_group=0, clusters=None,
                        spikes=None, clu=None, wizard=False):
        return [('_split2', (spikes, clu, wizard))]

    def _compute_similarity_matrix(self, target_next=None):
        exp = self.experiment
        channel_group = self.loader.shank
        clustering = 'main'  # TODO
        fetdim = exp.application_data.spikedetekt.n_features_per_channel

        clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
        spikes_data = exp.channel_groups[channel_group].spikes
        cluster_groups_data = getattr(exp.channel_groups[channel_group].cluster_groups, clustering)
        clusters_all = sorted(clusters_data.keys())
        cluster_groups = pd.Series([clusters_data[cl].cluster_group or 0
                                   for cl in clusters_all], index=clusters_all)

        spikes_selected, fm = spikes_data.load_features_masks(fraction=.1)
        clusters = getattr(spikes_data.clusters, clustering)[:][spikes_selected]

        fm = np.atleast_3d(fm)
        features = fm[:, :, 0]

        if features.shape[1] <= 1:
            return []

        # masks = fm[:, ::fetdim, 1]
        if fm.shape[2] > 1:
            masks = fm[:, :, 1]
        else:
            masks = None

        # features = pandaize(features, spikes_selected)
        # masks = pandaize(masks, spikes_selected)

        # Get cluster indices that need to be updated.
        # if clusters_to_update is None:
        # NOTE: not specifying explicitely clusters_to_update ensures that
        # all clusters that need to be updated are updated.
        # Allows to fix a bug where the matrix is not updated correctly
        # when multiple calls to this functions are called quickly.
        clusters_to_update = (self.statscache.similarity_matrix.
            not_in_key_indices(clusters_all))

        log.debug("Clusters to update: {0:s}".format(str(clusters_to_update)))

        # If there are pairs that need to be updated, launch the task.
        if len(clusters_to_update) > 0:
            self.mainwindow.set_busy(computing_matrix=True)
            # Launch the task.
            self.tasks.similarity_matrix_task.compute(features,
                clusters, cluster_groups, masks, clusters_to_update,
                target_next=target_next, similarity_measure=None)
        # Otherwise, update directly the correlograms view without launching
        # the task in the external process.
        else:
            return [('_wizard_update', (target_next,)),
                    ('_update_similarity_matrix_view',),
                    ]

    def _correlograms_computed(self, clusters, correlograms, ncorrbins, corrbin, sample_rate, wizard):
        clusters_selected = self.loader.get_clusters_selected()
        # Abort if the selection has changed during the computation of the
        # correlograms.
        # Reset the cursor.
        self.mainwindow.set_busy(computing_correlograms=False)
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
        return ('_update_correlograms_view', (), dict(wizard=wizard))

    def _similarity_matrix_computed(self, clusters_selected, matrix, clusters,
            cluster_groups, target_next=None):
        self.mainwindow.set_busy(computing_matrix=False)
        # spikes_slice = _get_similarity_matrix_slice(
            # self.loader.nspikes,
            # len(self.loader.get_clusters_unique()))
        # clusters_now = self.loader.get_clusters(
            # spikes=self.loader.background_spikes)
        # if not np.array_equal(clusters, clusters_now):
            # return False
        if len(matrix) == 0:
            return []
        self.statscache.similarity_matrix.update(clusters_selected, matrix)
        self.statscache.similarity_matrix_normalized = normalize(
            self.statscache.similarity_matrix.to_array(copy=True))
        # Update the cluster view with cluster quality.
        quality = np.diag(self.statscache.similarity_matrix_normalized).copy()
        self.statscache.cluster_quality = pd.Series(
            quality,
            index=self.statscache.similarity_matrix.indices,
            )
        self.get_view('ClusterView').set_quality(
            self.statscache.cluster_quality)
        return [('_wizard_update', (target_next,)),
                ('_update_similarity_matrix_view',),
                ]

    def _invalidate(self, clusters):
        self.statscache.invalidate(clusters)


    # View updates.
    # -------------
    def _update_correlograms_view(self, wizard=None):
        clu = self.loader.get_clusters_selected()
        # HACK: work around a bug with some GPU drivers and empty selections
        if len(clu)==0:
            return
        data = vd.get_correlogramsview_data(self.experiment,
            self.statscache.correlograms,
            clusters=clu,
            channel_group=self.loader.shank,
            wizard=wizard,
            )
        [view.set_data(**data) for view in self.get_views('CorrelogramsView')]

    def _update_similarity_matrix_view(self):
        data = vd.get_similaritymatrixview_data(self.experiment,
            self.statscache.similarity_matrix_normalized,
            channel_group=self.loader.shank,)
        [view.set_data(**data)
            for view in self.get_views('SimilarityMatrixView')]
        # Show selected clusters when the matrix has been updated.
        clusters = self.loader.get_clusters_selected()
        return ('_show_selection_in_matrix', (clusters,))

    def _update_feature_view(self, autozoom=None):
        clu = self.loader.clusters_selected
        # HACK: work around a bug with some GPU drivers and empty selections
        if len(clu)==0:
            return
        data = vd.get_featureview_data(self.experiment,
            clusters=clu,
            autozoom=autozoom,
            channel_group=self.loader.shank)
        [view.set_data(**data) for view in self.get_views('FeatureView')]

    def _update_waveform_view(self, autozoom=None, wizard=None):
        clu = self.loader.clusters_selected
        # HACK: work around a bug with some GPU drivers and empty selections
        if len(clu)==0:
            return
        data = vd.get_waveformview_data(self.experiment,
            clusters=clu,
            autozoom=autozoom,
            wizard=wizard,
            channel_group=self.loader.shank
            )
        [view.set_data(**data) for view in self.get_views('WaveformView')]

    def _update_trace_view(self):
        data = vd.get_traceview_data(self.experiment,
            channel_group=self.loader.shank)
        [view.set_data(**data) for view in self.get_views('TraceView')]

    def _update_cluster_view(self, clusters=None):
        """Update the cluster view using the data stored in the loader
        object."""
        data = vd.get_clusterview_data(self.experiment, self.statscache,
                                       channel_group=self.loader.shank)
        self.get_view('ClusterView').set_data(**data)
        if clusters is not None:
            return

    def _show_selection_in_matrix(self, clusters):
        if clusters is not None and 1 <= len(clusters) <= 2:
            [view.show_selection(clusters[0], clusters[-1])
                for view in self.get_views('SimilarityMatrixView')]


    # Override colors.
    # ----------------
    def _override_color(self, override_color):
        self.loader.set_override_color(override_color)
        return ['_update_feature_view', '_update_waveform_view', '_update_correlograms_view']


    # Change correlograms parameter.
    # ------------------------------
    def _change_correlograms_parameters(self, ncorrbins=None, corrbin=None):
        if ncorrbins % 2 == 0:
            ncorrbins += 1
        # Update the correlograms parameters.
        if ncorrbins is not None:
            SETTINGS['correlograms.ncorrbins'] = ncorrbins
        if corrbin is not None:
            SETTINGS['correlograms.corrbin'] = corrbin
        # Reset the cache.
        self.statscache.reset(ncorrbins)
        # Update the correlograms.
        clusters = self.loader.get_clusters_selected()
        return ('_compute_correlograms', (clusters,))


    # Merge/split actions.
    # --------------------
    def _merge(self, clusters, wizard=False):
        if len(clusters) >= 2:
            action, output = self.controller.merge_clusters(clusters)
            # Tell the next nodes whether the merge occurred after a wizard
            # selection or not, so that the merged cluster background is
            # highlighted or not.
            output['wizard'] = wizard
            return after_merge(output)

    def _split(self, clusters, spikes_selected, wizard=False):
        if len(spikes_selected) >= 1:
            action, output = self.controller.split_clusters(clusters,
                spikes_selected)
            output['wizard'] = wizard
            return after_split(output)

    def _split2(self, spikes, clusters, wizard=False):
        if len(spikes) >= 1:
            action, output = self.controller.split2_clusters(spikes, clusters)
            output['wizard'] = wizard
            return after_split(output)

    def _undo(self, wizard=False):
        undo = self.controller.undo()
        if undo is None:
            return
        action, output = undo
        output['wizard'] = wizard
        if action == 'merge_clusters_undo':
            return after_merge_undo(output)
        elif action == 'split_clusters_undo':
            return after_split_undo(output)
        elif action == 'split2_clusters_undo':
            return after_split_undo(output)
        elif action == 'change_cluster_color_undo':
            return after_cluster_color_changed_undo(output)
        elif action == 'change_group_color_undo':
            return after_group_color_changed(output)
        elif action == 'move_clusters_undo':
            return after_clusters_moved_undo(output)
        elif action == 'add_group_undo':
            return after_group_added(output)
        elif action == 'rename_group_undo':
            return after_group_renamed(output)
        elif action == 'remove_group_undo':
            return after_group_removed(output)

    def _redo(self, wizard=False):
        redo = self.controller.redo()
        if redo is None:
            return
        action, output = redo
        output['wizard'] = wizard
        if action == 'merge_clusters':
            return after_merge(output)
        elif action == 'split_clusters':
            return after_split(output)
        elif action == 'split2_clusters':
            return after_split(output)
        elif action == 'change_cluster_color':
            return after_cluster_color_changed(output)
        elif action == 'change_group_color':
            return after_group_color_changed(output)
        elif action == 'move_clusters':
            return after_clusters_moved(output)
        elif action == 'add_group':
            return after_group_added(output)
        elif action == 'rename_group':
            return after_group_renamed(output)
        elif action == 'remove_group':
            return after_group_removed(output)


    # Other actions.
    # --------------
    def _cluster_color_changed(self, cluster, color, wizard=True):
        action, output = self.controller.change_cluster_color(cluster, color)
        # if cluster == self.wizard.current_target():
        output['wizard'] = wizard
        return after_cluster_color_changed(output)

    def _group_color_changed(self, group, color):
        action, output = self.controller.change_group_color(group, color)
        return after_group_color_changed(output)

    def _group_renamed(self, group, name):
        action, output = self.controller.rename_group(group, name)
        return after_group_renamed(output)

    def _clusters_moved(self, clusters, group, wizard=False,):
        action, output = self.controller.move_clusters(clusters, group)
        output['wizard'] = wizard
        return after_clusters_moved(output)

    def _group_removed(self, group):
        action, output = self.controller.remove_group(group)
        return after_group_removed(output)

    def _group_added(self, group, name, color):
        action, output = self.controller.add_group(group, name, color)
        return after_group_added(output)


    # Wizard.
    # -------
    def _wizard_update(self, target=None, update_matrix=True):
        if update_matrix:
            self.wizard.set_data(
                cluster_groups=self.loader.get_cluster_groups('all'),
                similarity_matrix=self.statscache.similarity_matrix_normalized,
                )
        else:
            self.wizard.set_data(
                cluster_groups=self.loader.get_cluster_groups('all'),
                )
        self.wizard.update_candidates(target)

    def _wizard_change_color(self, clusters):
        if clusters is not None:
            # Set the background color in the cluster view for the wizard
            # target and candidate.
            self.get_view('ClusterView').set_background(
                {cluster: {0: 'target', 1: 'candidate'}.get(i, None)
                    for i, cluster in enumerate(clusters[:2])})

    def _wizard_change_candidate_color(self):
        candidate = self.wizard.current_candidate()
        target = self.wizard.current_target()
        # color = self.loader.get_cluster_color(candidate)
        return ('_cluster_color_changed', (candidate, random_color(),))

    def _wizard_show_pair(self, target=None, candidate=None):
        if target is None:
            target = (self.wizard.current_target(),
                      self.loader.get_cluster_color(self.wizard.current_target()))
        if candidate is None:
            try:
                candidate = (self.wizard.current_candidate(),
                             get_array(self.loader.get_cluster_color(self.wizard.current_candidate()))[0])
            # HACK: this can fail because when merging clusters, the merged
            # cluster (candidate) is deleted, and its color does not exist
            # anymore.
            except:
                candidate = (self.wizard.current_candidate(),
                             0)
        [view.set_wizard_pair(target, candidate)
            for view in self.get_views('FeatureView')]

    # Navigation.
    def _wizard_reset(self):
        clusters = self.wizard.reset()
        return ['_wizard_update', '_wizard_current_candidate']

    def _wizard_previous_candidate(self):
        clusters = self.wizard.previous_pair()
        return after_wizard_selection(clusters)

    def _wizard_current_candidate(self):
        clusters = self.wizard.current_pair()
        return after_wizard_selection(clusters)

    def _wizard_next_candidate(self):
        clusters = self.wizard.next_pair()
        return after_wizard_selection(clusters)

    def _wizard_skip_target(self):
        # Skip the current target and go the next target.
        self.wizard.skip_target()
        return [('_wizard_update', ()),
                ('_wizard_next_candidate',),]

    def _wizard_reset_skipped(self):
        self.wizard.reset_skipped()

    # Control.
    def _wizard_move_and_next(self, what, group):
        """Move target, candidate, or both, to a given group, and go to
        the next proposition."""
        # Current proposition.
        clusters = self.wizard.current_pair()
        if clusters is None:
            return
        target, candidate = clusters
        # Select appropriate clusters to move.
        if what == 'candidate':
            clusters = [candidate]
            # Keep the current target.
            target_next = target
            reset_skipped = False
        elif what == 'target':
            clusters = [target]
            # Go to the next best target cluster.
            target_next = None
            reset_skipped = True
        elif what == 'both':
            clusters = [candidate, target]
            # Go to the next best target cluster.
            target_next = None
            reset_skipped = True
        # Move clusters, and select next proposition.
        r = [('_clusters_moved', (clusters, group, True)),
            ]
        if reset_skipped:
            r += [('_wizard_reset_skipped',),]
        r += [('_wizard_update', (target_next,)),
              ('_wizard_next_candidate',),
              ]
        return r


# -----------------------------------------------------------------------------
# Tasks after actions
# -----------------------------------------------------------------------------
def union(*clusters_list):
    return sorted(set([item for sublist in clusters_list for item in sublist]))

# Merge/split actions.
def after_merge(output):
    if output.get('wizard', False):
        r = [('_invalidate', (output['clusters_to_merge'],)),
             # We specify here that the target in the wizard must be the
             # merged cluster.
             ('_compute_similarity_matrix', (output['cluster_merged'],)),
             ('_update_cluster_view'),
             ('_select_in_cluster_view', (output['cluster_merged'], [], True)),
             ('_wizard_change_color', ([output['cluster_merged']],)),
             ('_wizard_show_pair', ((output['cluster_merged'],
                                     output['cluster_merged_colors'][0]),)),
            ]
    else:
        r = [('_invalidate', (output['clusters_to_merge'],)),
             ('_compute_similarity_matrix',),
             ('_update_cluster_view'),
             ('_select_in_cluster_view', (output['cluster_merged'],)),
            ]
    return r

def after_merge_undo(output):
    clusters_to_invalidate = union(output['clusters_to_merge'], [output['cluster_merged']])
    if output.get('wizard', False):
        r = [('_invalidate', (clusters_to_invalidate,)),
             ('_compute_similarity_matrix', ()),
             # Update the wizard, but not the similarity matrix yet which
             # is being computed in an external process.
             # ('_wizard_update', (None, False)),
             ('_update_cluster_view'),
             ('_select_in_cluster_view', (output['clusters_to_merge'], [], True)),
             ('_wizard_change_color', (output['clusters_to_merge'],)),
             ('_wizard_show_pair', ((output['clusters_to_merge'][0],
                                     output['cluster_to_merge_colors'][0]),
                                    (output['clusters_to_merge'][1],
                                     output['cluster_to_merge_colors'][1])),
                                     ),
            ]
    else:
        r = [('_invalidate', (clusters_to_invalidate,)),
             ('_compute_similarity_matrix', ()),
             ('_update_cluster_view'),
             ('_select_in_cluster_view', (output['clusters_to_merge'],)),
            ]
    return r

def after_split(output):
    clusters_to_update = sorted(set(output['clusters_to_split']).union(set(
        output['clusters_split'])) - set(output['clusters_empty']))
    if output.get('wizard', False):
        r = [('_invalidate', (output['clusters_to_split'],)),
             ('_compute_similarity_matrix', (True,)),
             # Update the wizard, but not the similarity matrix yet which
             # is being computed in an external process.
             # ('_wizard_update', (True, False)),
             ('_update_cluster_view'),
             ('_select_in_cluster_view', (clusters_to_update, [], True)),
             ('_wizard_change_color', (output['clusters_to_split'],)),
            ]
    else:
        r = [ ('_invalidate', (output['clusters_to_split'],)),
             ('_compute_similarity_matrix', (True,)),
             ('_update_cluster_view'),
             ('_select_in_cluster_view', (clusters_to_update,)),
            ]
    return r

def after_split_undo(output):
    clusters_to_invalidate = union(output['clusters_to_split'], output['clusters_split'])
    if output.get('wizard', False):
        r = [('_invalidate', (clusters_to_invalidate,)),
             ('_compute_similarity_matrix', (True,)),
             # Update the wizard, but not the similarity matrix yet which
             # is being computed in an external process.
             # ('_wizard_update', (True, False)),
             ('_update_cluster_view'),
             ('_select_in_cluster_view', (output['clusters_to_split'], [], True)),
             ('_wizard_change_color', (output['clusters_to_split'],)),
            ]
    else:
        r = [('_invalidate', (clusters_to_invalidate,)),
             ('_compute_similarity_matrix', (True,)),
             ('_update_cluster_view'),
             ('_select_in_cluster_view', (output['clusters_to_split'],)),
            ]
    return r


# Other actions.
def after_cluster_color_changed(output):
    if output.get('wizard', False):
        return [('_update_cluster_view'),
                ('_select_in_cluster_view', (output['clusters'], [], True)),
                ('_wizard_change_color', (output['clusters'],)),
                ('_wizard_show_pair',),# (output['cluster'],
                                         # output['color_new'])),
                ]
    else:
        return [('_update_cluster_view'),
                ('_select_in_cluster_view', (output['clusters'],)),
                ]

def after_cluster_color_changed_undo(output):
    if output.get('wizard', False):
        return [('_update_cluster_view'),
                ('_select_in_cluster_view', (output['clusters'], [], True)),
                ('_wizard_change_color', (output['clusters'],)),
                ('_wizard_show_pair',),# (output['cluster'],
                                        # output['color_old'])),
                ]
    else:
        return [('_update_cluster_view'),
                ('_select_in_cluster_view', (output['clusters'],)),
                ]

def after_group_color_changed(output):
    return [('_update_cluster_view'),
            ('_select_in_cluster_view', ([],), dict(groups=output['groups']),),]

def after_clusters_moved(output):
    r = [ ('_update_cluster_view'),
          ('_update_similarity_matrix_view'),
          ]
    # If the wizard is active, it will be updated later so do not update it
    # now.
    if not output.get('wizard', False):
        r += [('_wizard_update',),]
        if 'next_cluster' in output:
            clusters = [output['next_cluster']]
        else:
            clusters = output['clusters']
        # When deleting clusters, selecting the next one in the same group.
        r += [('_select_in_cluster_view', (clusters,)),]
    return r

def after_clusters_moved_undo(output):
    if 'next_cluster' in output:
        clusters = [output['next_cluster']]
    else:
        clusters = output['clusters']
    r = [('_update_cluster_view'),
         ('_update_similarity_matrix_view'),
         ('_wizard_update',),
         ('_select_in_cluster_view', (clusters,)),]
    return r

def after_group_added(output):
    return [('_update_cluster_view')]

def after_group_renamed(output):
    return [('_update_cluster_view')]

def after_group_removed(output):
    return [('_update_cluster_view')]


# Wizard.
def after_wizard_selection(clusters):
    if clusters is None:
        return None
    else:
        return [
                ('_select_in_cluster_view', (clusters, (), True)),
                ('_wizard_change_color', (clusters,)),
                ('_wizard_show_pair',),
                ]



