"""Main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import time
import os
import inspect
from collections import OrderedDict, Counter
from functools import partial

import pandas as pd
import numpy as np
import numpy.random as rnd
from galry import QtGui, QtCore
from qtools import inprocess, inthread, QT_BINDING

import klustaviewa.views as vw
from klustaviewa.gui.icons import get_icon
from klustaviewa.control.controller import Controller
from klustaviewa.dataio.tools import get_array
from klustaviewa.dataio.loader import KlustersLoader
from klustaviewa.gui.buffer import Buffer
from klustaviewa.gui.dock import ViewDockWidget, DockTitleBar
from klustaviewa.stats.cache import StatsCache
from klustaviewa.stats.correlations import normalize
from klustaviewa.stats.correlograms import get_baselines
import klustaviewa.utils.logger as log
from klustaviewa.utils.logger import FileLogger, register, unregister
from klustaviewa.utils.persistence import encode_bytearray, decode_bytearray
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.utils.settings import SETTINGS
from klustaviewa.utils.globalpaths import APPNAME, ABOUT, get_global_path
from klustaviewa.gui.threads import ThreadedTasks, LOCK
import rcicons

    
# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
class MainWindow(QtGui.QMainWindow):
    
    def __init__(self, parent=None, dolog=True):
        super(MainWindow, self).__init__(parent)
        
        self.dolog = dolog
        if self.dolog:
            create_file_logger()
        
        log.debug("Using {0:s}.".format(QT_BINDING))
        
        # Main window options.
        self.move(50, 50)
        self.setWindowTitle('KlustaViewa')
        
        # Focus options.
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setMouseTracking(True)
        # self.installEventFilter(EventFilter(self))
        
        # Dock widgets options.
        self.setDockNestingEnabled(True)
        self.setAnimated(False)
        
        # Initialize some variables.
        self.statscache = None
        self.loader = KlustersLoader()
        self.loader.progressReported.connect(self.open_progress_reported)
        self.controller = None
        self.spikes_highlighted = []
        self.spikes_selected = []
        self.wizard_active = False
        self.need_save = False
        # self.last_selection_time = time.clock()
        self.busy_cursor = QtGui.QCursor(QtCore.Qt.BusyCursor)
        self.normal_cursor = QtGui.QCursor(QtCore.Qt.ArrowCursor)
        self.override_color = False
        # self.do_renumber = False
        
        # Create the main window.
        self.create_views()
        self.create_file_actions()
        self.create_view_actions()
        self.create_correlograms_actions()
        self.create_control_actions()
        self.create_wizard_actions()
        self.create_help_actions()
        self.create_menu()
        self.create_toolbar()
        self.create_open_progress_dialog()
        self.create_threads()
        
        # Update action enabled/disabled property.
        self.update_action_enabled()
        
        # Show the main window.
        self.set_styles()
        self.restore_geometry()
        self.show()
    
    def set_styles(self):
        # set stylesheet
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "styles.css")
        with open(path, 'r') as f:
            stylesheet = f.read()
        stylesheet = stylesheet.replace('%ACCENT%', '#cdcdcd')
        stylesheet = stylesheet.replace('%ACCENT2%', '#a0a0a0')
        stylesheet = stylesheet.replace('%ACCENT3%', '#909090')
        stylesheet = stylesheet.replace('%ACCENT4%', '#cdcdcd')
        self.setStyleSheet(stylesheet)
    
    def set_busy_cursor(self):
        cursor = QtGui.QApplication.overrideCursor()
        if cursor is None or cursor.shape() != QtCore.Qt.BusyCursor:
            QtGui.QApplication.setOverrideCursor(self.busy_cursor)
        
    def set_normal_cursor(self):
        # QtGui.QApplication.setOverrideCursor(self.normal_cursor)
        QtGui.QApplication.restoreOverrideCursor()
    
    
    # Actions.
    # --------
    def add_action(self, name, text, callback=None, shortcut=None,
            checkable=False, icon=None):
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
        setattr(self, name + '_action', action)
        
    def create_file_actions(self):
        # Open actions.
        self.add_action('open', '&Open', shortcut='Ctrl+O', icon='open')
        
        # Open last file action
        path = SETTINGS['main_window.last_data_file']
        if path:
            lastfile = os.path.basename(path)
            if len(lastfile) > 30:
                lastfile = '...' + lastfile[-30:]
            self.add_action('open_last', 'Open &last ({0:s})'.format(
                lastfile), shortcut='Ctrl+Alt+O')
        else:
            self.add_action('open_last', 'Open &last', shortcut='Ctrl+Alt+O')
            self.open_last_action.setEnabled(False)
            
        self.add_action('save', '&Save', shortcut='Ctrl+S', icon='save')
        
        self.add_action('renumber', '&Renumber when saving', checkable=True)
        
        # Quit action.
        self.add_action('quit', '&Quit', shortcut='Ctrl+Q')
        
    def create_view_actions(self):
        self.add_action('add_feature_view', 'Add FeatureView')
        self.add_action('add_waveform_view', 'Add WaveformView')
        self.add_action('add_similarity_matrix_view',
            'Add SimilarityMatrixView')
        self.add_action('add_correlograms_view', 'Add CorrelogramsView')
        self.add_action('add_ipython_view', 'Add IPythonView')
        
        self.add_action('override_color', 'Override cluster &color',
            icon='override_color', shortcut='C')
    
    def create_control_actions(self):
        self.add_action('undo', '&Undo', shortcut='Ctrl+Z', icon='undo')
        self.add_action('redo', '&Redo', shortcut='Ctrl+Y', icon='redo')
        
        self.add_action('merge', '&Merge', shortcut='G', icon='merge')
        self.add_action('split', '&Split', shortcut='K', icon='split')

    def create_correlograms_actions(self):
        self.add_action('change_ncorrbins', 'Change time &window')
        self.add_action('change_corrbin', 'Change &bin size')
        
        self.add_action('change_corr_normalization', 'Change &normalization')
        
    def create_wizard_actions(self):
        self.add_action('reset_navigation', '&Reset navigation')
            
        self.add_action('next_candidate', '&Next candidate', 
            shortcut='Space')
        self.add_action('previous_candidate', '&Previous candidate', 
            shortcut='CTRL+Space')
        self.add_action('delete_candidate', '&Delete candidate', 
            shortcut='Backspace')
            
        self.add_action('next_target', '&Next target', 
            shortcut='Return')
        self.add_action('previous_target', '&Previous target', 
            shortcut='CTRL+Return')
        self.add_action('delete_target', '&Delete target', 
            shortcut='SHIFT+Backspace')
        
    def create_help_actions(self):
        self.add_action('about', '&About')
        self.add_action('shortcuts', 'Show &shortcuts')
        self.add_action('refresh_preferences', '&Refresh preferences',
            shortcut='CTRL+R')
        
    def create_menu(self):
        # File menu.
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_last_action)
        file_menu.addSeparator()
        file_menu.addAction(self.renumber_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        
        # Views menu.
        views_menu = self.menuBar().addMenu("&Views")
        views_menu.addAction(self.add_feature_view_action)
        views_menu.addAction(self.add_waveform_view_action)
        views_menu.addAction(self.add_correlograms_view_action)
        views_menu.addAction(self.add_similarity_matrix_view_action)
        views_menu.addSeparator()
        if vw.IPYTHON:
            views_menu.addAction(self.add_ipython_view_action)
            views_menu.addSeparator()
        views_menu.addAction(self.override_color_action)
        
        # Correlograms menu.
        correlograms_menu = self.menuBar().addMenu("&Correlograms")
        correlograms_menu.addAction(self.change_ncorrbins_action)
        correlograms_menu.addAction(self.change_corrbin_action)
        correlograms_menu.addSeparator()
        correlograms_menu.addAction(self.change_corr_normalization_action)
        
        # Actions menu.
        actions_menu = self.menuBar().addMenu("&Actions")
        actions_menu.addAction(self.undo_action)
        actions_menu.addAction(self.redo_action)
        actions_menu.addSeparator()
        actions_menu.addAction(self.get_view('ClusterView').move_to_mua_action)
        actions_menu.addAction(self.get_view('ClusterView').move_to_noise_action)
        actions_menu.addSeparator()
        actions_menu.addAction(self.merge_action)
        actions_menu.addAction(self.split_action)
        
        # Wizard menu.
        wizard_menu = self.menuBar().addMenu("&Wizard")
        wizard_menu.addAction(self.reset_navigation_action)
        wizard_menu.addSeparator()
        wizard_menu.addAction(self.next_candidate_action)
        wizard_menu.addAction(self.previous_candidate_action)
        wizard_menu.addAction(self.delete_candidate_action)
        wizard_menu.addSeparator()
        wizard_menu.addAction(self.next_target_action)
        wizard_menu.addAction(self.previous_target_action)
        wizard_menu.addAction(self.delete_target_action)
        
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.refresh_preferences_action)
        help_menu.addSeparator()
        help_menu.addAction(self.shortcuts_action)
        help_menu.addAction(self.about_action)
        
    def create_toolbar(self):
        self.toolbar = self.addToolBar("KlustaViewaToolbar")
        self.toolbar.setObjectName("KlustaViewaToolbar")
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.save_action)
        # self.toolbar.addAction(self.saveas_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.merge_action)
        self.toolbar.addAction(self.split_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.get_view('ClusterView').move_to_mua_action)
        self.toolbar.addAction(self.get_view('ClusterView').move_to_noise_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.undo_action)
        self.toolbar.addAction(self.redo_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.override_color_action)
        
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolbar)
        
    def create_open_progress_dialog(self):
        self.open_progress = QtGui.QProgressDialog("Loading...", "Cancel", 0, 0, self)
        self.open_progress.setWindowModality(QtCore.Qt.WindowModal)
        self.open_progress.setValue(0)
        self.open_progress.setCancelButton(None)
        self.open_progress.setMinimumDuration(0)
        
    def update_action_enabled(self):
        self.undo_action.setEnabled(self.can_undo())
        self.redo_action.setEnabled(self.can_redo())
        self.merge_action.setEnabled(self.can_merge())
        self.split_action.setEnabled(self.can_split())
    
    def can_undo(self):
        if self.controller is None:
            return False
        return self.controller.can_undo()
    
    def can_redo(self):
        if self.controller is None:
            return False
        return self.controller.can_redo()
    
    def can_merge(self):
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        return len(clusters) >= 2
        
    def can_split(self):
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        spikes_selected = self.spikes_selected
        return len(spikes_selected) >= 1
    
    
    # File menu callbacks.
    # --------------------
    def open_callback(self, checked=None):
        # HACK: Force release of Ctrl key.
        self.force_key_release()
        
        folder = SETTINGS['main_window.last_data_dir']
        path = QtGui.QFileDialog.getOpenFileName(self, 
            "Open a file (.clu or other)", folder)[0]
        # If a file has been selected, open it.
        if path:
            # Launch the loading task in the background asynchronously.
            self.tasks.open_task.open(self.loader, path)
            # Save the folder.
            folder = os.path.dirname(path)
            SETTINGS['main_window.last_data_dir'] = folder
            SETTINGS['main_window.last_data_file'] = path
            
    def save_callback(self, checked=None):
        folder = SETTINGS.get('main_window.last_data_file')
        self.loader.save(renumber=self.renumber_action.isChecked())
        self.need_save = False
        
        # # ask a new file name
        # filename = QtGui.QFileDialog.getSaveFileName(self, "Save a CLU file",
            # os.path.join(folder, default_filename))[0]
        # if filename:
            # # save the new file name
            # self.save_filename = filename
            # # save
            # self.provider.save(filename)
        
            
    def open_last_callback(self, checked=None):
        path = SETTINGS['main_window.last_data_file']
        if path:
            self.tasks.open_task.open(self.loader, path)
            
    def quit_callback(self, checked=None):
        self.close()
    
    
    # Views menu callbacks.
    # ---------------------
    def add_feature_view_callback(self, checked=None):
        self.add_feature_view()
        
    def add_waveform_view_callback(self, checked=None):
        self.add_waveform_view()
        
    def add_similarity_matrix_view_callback(self, checked=None):
        self.add_similarity_matrix_view()
        
    def add_correlograms_view_callback(self, checked=None):
        self.add_correlograms_view()
    
    def add_ipython_view_callback(self, checked=None):
        self.add_ipython_view()
    
    def override_color_callback(self, checked=None):
        self.override_color = not self.override_color
        self.loader.set_override_color(self.override_color)
        # view = self.get_view('ClusterView')
        # clusters = view.selected_clusters()
        # groups = view.selected_groups()
        # if groups:
            # view.select_groups(groups)
        # if clusters:
            # view.select(clusters)
        self.update_views()
    
    
    # Correlograms callbacks.
    # -----------------------
    def change_correlograms_parameters(self, ncorrbins=None, corrbin=None):
        # Update the correlograms parameters.
        if ncorrbins is not None:
            self.loader.ncorrbins = ncorrbins
        if corrbin is not None:
            self.loader.corrbin = corrbin
        # Reset the cache.
        self.statscache.reset(self.loader.ncorrbins)
        # Update the correlograms.
        clusters = self.loader.get_clusters_selected()
        self.start_compute_correlograms(clusters)
        
    def change_ncorrbins_callback(self, checked=None):
        if not self.loader:
            return
        corrbin = self.loader.corrbin
        duration = self.loader.get_correlogram_window()
        duration_new, ok = QtGui.QInputDialog.getDouble(self,
            "Correlograms time window", "Half width (ms):", 
            duration / 2 * 1000, 1, 1000, 1)
        if ok:
            duration_new = duration_new * .001 * 2
            ncorrbins_new = 2 * int(np.ceil(.5 * duration_new / corrbin))
            # ncorrbins_new = int(duration_new / corrbin * .001)
            self.change_correlograms_parameters(ncorrbins=ncorrbins_new)
    
    def change_corrbin_callback(self, checked=None):
        if not self.loader:
            return
        ncorrbins = self.loader.ncorrbins
        corrbin = self.loader.corrbin
        duration = self.loader.get_correlogram_window()
        # duration = self.loader.get_correlogram_window() * 1000
        corrbin_new, ok = QtGui.QInputDialog.getDouble(self,
            "Correlograms bin size", "Bin size (ms):", 
            corrbin * 1000, .1, 10, 2)
        if ok:
            corrbin_new = corrbin_new * .001
            # ncorrbins_new = int(duration / corrbin_new)
            ncorrbins_new = 2 * int(np.ceil(.5 * duration/ corrbin_new))
            # print corrbin_new, ncorrbins_new, duration
            self.change_correlograms_parameters(corrbin=corrbin_new,
                ncorrbins=ncorrbins_new)
    
    def change_corr_normalization_callback(self, checked=None):
        self.get_view('CorrelogramsView').change_normalization()
    
    
    # Actions callbacks.
    # ------------------
    def update_cluster_selection(self, clusters, groups=[]):
        self.update_action_enabled()
        self.update_cluster_view()
        self.get_view('ClusterView').select(clusters, 
            groups=groups)
    
    def action_processed(self, action, to_select=[], to_invalidate=[],
        to_compute=None, groups_to_select=None):
        """Called after an action has been processed. Used to update the 
        different views and launch tasks."""
        if isinstance(to_select, (int, long, np.integer)):
            to_select = [to_select]
        if isinstance(to_invalidate, (int, long, np.integer)):
            to_invalidate = [to_invalidate]
        if isinstance(to_compute, (int, long, np.integer)):
            to_compute = [to_compute]
        # Select clusters to be selected.
        if len(to_select) > 0:
            self.update_cluster_selection(to_select, groups=groups_to_select)
        elif len(groups_to_select) > 0:
            self.update_cluster_selection([], groups=groups_to_select)
        # Invalidate clusters.
        if len(to_invalidate) > 0:
            self.statscache.invalidate(to_invalidate)
        # Compute the correlation matrix for the requested clusters.
        if to_compute is not None:
            self.start_compute_similarity_matrix(to_compute)
            
        # print action, to_select, to_invalidate, to_compute, groups_to_select
            
        self.need_save = True
        
    def merge_callback(self, checked=None):
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        if len(clusters) >= 2:
            with LOCK:
                action, output = self.controller.merge_clusters(clusters)
            self.action_processed(action, **output)
            # Inform the wizard.
            cluster_merged = output['to_select']
            clusters_to_merge = output['to_invalidate']
            clusters_to_merge.remove(cluster_merged)
            self.tasks.wizard_task.merged(clusters_to_merge, cluster_merged)
            
    def split_callback(self, checked=None):
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        spikes_selected = self.spikes_selected
        if len(spikes_selected) >= 1:
            with LOCK:
                action, output = self.controller.split_clusters(
                    clusters, spikes_selected)
            self.action_processed(action, **output)
            # Cancel the selection after the split.
            self.spikes_selected = []
            # Inform the wizard.
            # TODO
            
    def undo_callback(self, checked=None):
        with LOCK:
            action, output = self.controller.undo()
        if output is None:
            output = {}
        self.action_processed(action, **output)
        if action == 'merge_clusters_undo':
            # Inform the wizard.
            clusters_to_merge = list(output['to_select'])
            self.tasks.wizard_task.merged_undo(clusters_to_merge)
        
    def redo_callback(self, checked=None):
        with LOCK:
            action, output = self.controller.redo()
        if output is None:
            output = {}
        self.action_processed(action, **output)
        
    def cluster_color_changed_callback(self, cluster, color):
        with LOCK:
            action, output = self.controller.change_cluster_color(cluster, color)
        self.action_processed(action, **output)
        
    def group_color_changed_callback(self, group, color):
        with LOCK:
            action, output = self.controller.change_group_color(group, color)
        self.action_processed(action, **output)
        
    def group_renamed_callback(self, group, name):
        with LOCK:
            action, output = self.controller.rename_group(group, name)
        self.action_processed(action, **output)
        
    def clusters_moved_callback(self, clusters, group):
        with LOCK:
            action, output = self.controller.move_clusters(clusters, group)
        self.action_processed(action, **output)
        # Update the wizard.
        self.tasks.wizard_task.moved(clusters, group)
        
        
    def group_removed_callback(self, group):
        with LOCK:
            action, output = self.controller.remove_group(group)
        self.action_processed(action, **output)
        
    def group_added_callback(self, group, name, color):
        with LOCK:
            action, output = self.controller.add_group(group, name, color)
        self.action_processed(action, **output)
    
    
    # Help callbacks.
    # ---------------
    def about_callback(self, checked=None):
        QtGui.QMessageBox.about(self, "KlustaViewa", ABOUT)
    
    def shortcuts_callback(self, checked=None):
        e = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, 
                             QtCore.Qt.Key_H,
                             QtCore.Qt.NoModifier,)
        self.keyPressEvent(e)
    
    def refresh_preferences_callback(self, checked=None):
        log.debug("Refreshing user preferences.")
        USERPREF.refresh()
        
    
    # Views callbacks.
    # ----------------
    def waveform_spikes_highlighted_callback(self, spikes):
        self.spikes_highlighted = spikes
        self.get_view('FeatureView').highlight_spikes(get_array(spikes))
        
    def features_spikes_highlighted_callback(self, spikes):
        self.spikes_highlighted = spikes
        self.get_view('WaveformView').highlight_spikes(get_array(spikes))
        
    def features_spikes_selected_callback(self, spikes):
        self.spikes_selected = spikes
        self.update_action_enabled()
        self.get_view('WaveformView').highlight_spikes(get_array(spikes))
        
    def waveform_box_clicked_callback(self, coord, cluster, channel):
        """Changed in waveform ==> change in feature"""
        self.get_view('FeatureView').set_projection(coord, channel, -1)
        
    def projection_changed_callback(self, coord, channel, feature):
        """Changed in projection ==> change in feature"""
        self.get_view('FeatureView').set_projection(coord, channel, feature)
        
    def features_projection_changed_callback(self, coord, channel, feature):
        """Changed in feature ==> change in projection"""
        self.get_view('ProjectionView').set_projection(coord, channel, feature,
            do_emit=False)
        
    
    # Task methods.
    # -------------
    def open_done(self):
        # Start the selection buffer.
        self.buffer = Buffer(self, delay_timer=.1, delay_buffer=.2)
        self.buffer.start()
        self.buffer.accepted.connect(self.buffer_accepted_callback)
        
        # HACK: force release of Control key.
        self.force_key_release()
        clusters = self.get_view('ClusterView').selected_clusters()
        if clusters:
            self.get_view('ClusterView').unselect()
        
        # Create the Controller.
        self.controller = Controller(self.loader)
        # Create the cache for the cluster statistics that need to be
        # computed in the background.
        self.statscache = StatsCache(self.loader.ncorrbins)
        # Update stats cache in IPython view.
        ipython = self.get_view('IPythonView')
        if ipython:
            ipython.set_data(stats=self.statscache)
        # Start computing the correlation matrix.
        self.start_compute_similarity_matrix()
        # Update the wizard.
        self.initialize_wizard()
        # Update the views.
        self.update_cluster_view()
        self.update_projection_view()
        
    def open_progress_reported(self, progress, progress_max):
        self.open_progress.setMaximum(progress_max)
        self.open_progress.setValue(progress)
        
    def start_compute_correlograms(self, clusters_selected):
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
            self.set_busy_cursor()
            # Launch the task.
            self.tasks.correlograms_task.compute(spiketimes, clusters,
                clusters_to_update=clusters_to_update, 
                clusters_selected=clusters_selected,
                ncorrbins=ncorrbins, corrbin=corrbin)    
        # Otherwise, update directly the correlograms view without launching
        # the task in the external process.
        else:
            self.update_correlograms_view()
        
    def start_compute_similarity_matrix(self, clusters_to_update=None):
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
            self.update_similarity_matrix_view()
        
    def correlograms_computed(self, clusters, correlograms, ncorrbins, corrbin):
        clusters_selected = self.loader.get_clusters_selected()
        # Abort if the selection has changed during the computation of the
        # correlograms.
        # Reset the cursor.
        self.set_normal_cursor()
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
        self.update_correlograms_view()
        
    def similarity_matrix_computed(self, clusters_selected, matrix, clusters,
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
        self.tasks.wizard_task.set_data(
            clusters=clusters,
            cluster_groups=cluster_groups,
            similarity_matrix=self.statscache.similarity_matrix_normalized,
            )
        # Update the view.
        self.update_similarity_matrix_view()
        
    
    # Selection methods.
    # ------------------
    def buffer_accepted_callback(self, clusters):
        # Highlight row and column of the selected cluster(s) in the
        # Similarity Matrix View.
        if clusters is not None and 1 <= len(clusters) <= 2:
            self.get_view('SimilarityMatrixView').show_selection(
                clusters[0], clusters[-1])
        self.tasks.select_task.select(self.loader, clusters)
        
    def selection_done(self, clusters_selected):
        """Called on the main thread once the clusters have been loaded 
        in the main thread."""
        if not np.array_equal(clusters_selected, 
            self.loader.get_clusters_selected()):
            log.debug(("Skip updating views with clusters_selected={0:s} and "
                "actual selected clusters={1:s}").format(
                    str(clusters_selected),
                    str(self.loader.get_clusters_selected())))
            return
            
        # Force refreshing the preference file.
        # USERPREF.refresh()
            
        # Launch the computation of the correlograms.
        self.start_compute_correlograms(clusters_selected)
        
        # Update the different views, with autozoom on if the selection has
        # been made by the wizard.
        with LOCK:
            self.update_feature_view(autozoom=self.wizard_active)
            self.update_waveform_view()#autozoom=self.wizard_active)
            
        # Update action enabled/disabled property.
        self.update_action_enabled()
        # self.wizard_active = False
    
    def clusters_selected_callback(self, clusters, external_call):
        self.wizard_active = external_call
        self.buffer.request(clusters)
    
    def cluster_pair_selected_callback(self, clusters):
        """Callback when the user clicks on a pair in the
        SimilarityMatrixView."""
        self.wizard_active = False
        self.get_view('ClusterView').select(clusters)
    
    
    # Wizard.
    # -------
    def initialize_wizard(self):
        self.tasks.wizard_task.reset()
        self.tasks.wizard_task.set_data(
            # Data.
            features=self.loader.get_features('all'),
            spiketimes=self.loader.get_spiketimes('all'),
            masks=self.loader.get_masks('all'),
            clusters=self.loader.get_clusters('all'),
            # clusters_unique=self.loader.get_clusters_unique(),
            cluster_groups=self.loader.get_cluster_groups('all'),
            # Statistics.
            correlograms=self.statscache.correlograms,
            similarity_matrix=self.statscache.similarity_matrix,
            )
    
    # def update_wizard(self, clusters, cluster_groups, similarity_matrix):
        # self.tasks.wizard_task.set_data(
            # clusters=clusters,
            # cluster_groups=cluster_groups,#get_array(self.loader.get_cluster_groups('all')),
            # similarity_matrix=similarity_matrix#self.statscache.similarity_matrix_normalized,
            # )
        
    def wizard_callback(self, f, *args, **kwargs):
        kwargs['_sync'] = True
        clusters = f(*args, **kwargs)[2]['_result']
        # log.info("The wizard proposes clusters {0:s}.".format(str(clusters)))
        if clusters is None or len(clusters) == 0:
            return
        self.wizard_active = True
        self.get_view('ClusterView').select(clusters, external_call=True)
    
    def reset_navigation_callback(self, checked=None):
        self.wizard_callback(self.tasks.wizard_task.reset_navigation)
    
    def previous_candidate_callback(self, checked=None):
        self.wizard_callback(self.tasks.wizard_task.previous)
        
    def next_candidate_callback(self, checked=None):
        self.wizard_callback(self.tasks.wizard_task.next)
    
    def previous_target_callback(self, checked=None):
        self.wizard_callback(self.tasks.wizard_task.previous_target)
        
    def next_target_callback(self, checked=None):
        # Get the current target and candidate.
        target, candidate = self.tasks.wizard_task.current(
            _sync=True)[2]['_result']
        # Move the target cluster to the good group.
        self.clusters_moved_callback([target], 2)
        # Go to the next target cluster.
        self.wizard_callback(self.tasks.wizard_task.next_target)
        
    def delete_candidate_callback(self, checked=None):
        # Get the current target and candidate.
        target, candidate = self.tasks.wizard_task.current(
            _sync=True)[2]['_result']
        # Delete the cluster, and update the wizard
        self.clusters_moved_callback([candidate], 1)
        # Inform the wizard of the deletion.
        # self.wizard_callback(self.tasks.wizard_task.candidate_deleted, candidate)
        self.next_candidate_callback()
        
    def delete_target_callback(self, checked=None):
        # Get the current target and candidate.
        target, candidate = self.tasks.wizard_task.current(
            _sync=True)[2]['_result']
        # Delete the cluster, and update the wizard.
        self.clusters_moved_callback([target], 1)
        # Inform the wizard of the deletion.
        # self.wizard_callback(self.tasks.wizard_task.target_deleted, target)
        self.next_target_callback()
    
    
    # Threads.
    # --------
    def create_threads(self):
        # Create the external threads.
        self.tasks = ThreadedTasks()
        self.tasks.open_task.dataOpened.connect(self.open_done)
        self.tasks.select_task.clustersSelected.connect(self.selection_done)
        self.tasks.correlograms_task.correlogramsComputed.connect(
            self.correlograms_computed)
        self.tasks.similarity_matrix_task.correlationMatrixComputed.connect(
            self.similarity_matrix_computed)
    
    def join_threads(self):
         self.tasks.join()
    
    
    # View methods.
    # -------------
    def create_view(self, view_class, position=None, 
        closable=True, floatable=True, index=0, floating=None, **kwargs):
        """Add a widget to the main window."""
        view = view_class(self, getfocus=False)
        view.set_data(**kwargs)
        # if not position:
            # position = QtCore.Qt.LeftDockWidgetArea
            
        # Create the dock widget.
        name = view_class.__name__ + '_' + str(index)
        dockwidget = ViewDockWidget(view_class.__name__)
        # dockwidget = ViewDockWidget(name)
        dockwidget.setObjectName(name)
        dockwidget.setWidget(view)
        dockwidget.closed.connect(self.dock_widget_closed)
        
        # Set dock widget options.
        options = QtGui.QDockWidget.DockWidgetMovable
        if closable:
            options = options | QtGui.QDockWidget.DockWidgetClosable
        if floatable:
            options = options | QtGui.QDockWidget.DockWidgetFloatable
            
        dockwidget.setFeatures(options)
        dockwidget.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea |
            QtCore.Qt.RightDockWidgetArea |
            QtCore.Qt.TopDockWidgetArea |
            QtCore.Qt.BottomDockWidgetArea)
        # dockwidget.setAttribute(QtCore.Qt.WA_DeleteOnClose )
        
        dockwidget.visibilityChanged.connect(partial(
            self.dock_visibility_changed_callback, view))
            
        if position is not None:
            # Add the dock widget to the main window.
            self.addDockWidget(position, dockwidget)
        
        if floating is not None:
            dockwidget.setFloating(floating)
        
        dockwidget.setTitleBarWidget(DockTitleBar(dockwidget, view_class.__name__))
            
        # Return the view widget.
        return view
    
    def add_cluster_view(self):
        view = self.create_view(vw.ClusterView,
            position=QtCore.Qt.LeftDockWidgetArea,
            index=len(self.views['ClusterView']),
            closable=False, 
            # floatable=False
            )
            
        # Connect callback functions.
        view.clustersSelected.connect(self.clusters_selected_callback)
        view.clusterColorChanged.connect(self.cluster_color_changed_callback)
        view.groupColorChanged.connect(self.group_color_changed_callback)
        view.groupRenamed.connect(self.group_renamed_callback)
        view.clustersMoved.connect(self.clusters_moved_callback)
        view.groupAdded.connect(self.group_added_callback)
        view.groupRemoved.connect(self.group_removed_callback)
        
        self.views['ClusterView'].append(view)
        
    def add_projection_view(self):
        view = self.create_view(vw.ProjectionView,
            index=len(self.views['ProjectionView']),
            position=QtCore.Qt.LeftDockWidgetArea, 
            closable=False, 
            # floatable=False
            )
            
        # Connect callback functions.
        view.projectionChanged.connect(self.projection_changed_callback)
        
        self.views['ProjectionView'].append(view)
        
    def dock_visibility_changed_callback(self, view, visibility):
        # Register dock widget visibility.
        view.visibility = visibility
        
    def restore_last_view(self, name):
        """Return True if the last view was successfully restored,
        False if the view needs to be restored manually by creating a new
        view."""
        # No existing view: need to create a new view.
        if not self.views[name]:
            return False
        view = self.views[name][-1]
        # A view exists and it is hidden: restore it.
        if getattr(view, 'visibility', None) is False:
            view.parent().toggleViewAction().activate(QtGui.QAction.Trigger)
            return True
        # A view exists but it is not hidden: just add a new view.
        else:
            return False
        
    def add_similarity_matrix_view(self):
        # Try restoring the last view if it exists and it is hidden, and if
        # successfully restored, do nothing more. Otherwise, need to create 
        # a new view.
        if self.restore_last_view('SimilarityMatrixView'):
            return
        view = self.create_view(vw.SimilarityMatrixView,
            index=len(self.views['SimilarityMatrixView']),
            position=QtCore.Qt.LeftDockWidgetArea,)
        view.clustersSelected.connect(self.cluster_pair_selected_callback)
        self.views['SimilarityMatrixView'].append(view)
        self.update_similarity_matrix_view()
    
    def add_waveform_view(self):
        view = self.create_view(vw.WaveformView,
            index=len(self.views['WaveformView']),
            position=QtCore.Qt.RightDockWidgetArea,)
        view.spikesHighlighted.connect(
            self.waveform_spikes_highlighted_callback)
        view.boxClicked.connect(self.waveform_box_clicked_callback)
        self.views['WaveformView'].append(view)
        
    def add_feature_view(self):
        view = self.create_view(vw.FeatureView,
            index=len(self.views['FeatureView']),
            position=QtCore.Qt.RightDockWidgetArea,)
        view.spikesHighlighted.connect(
            self.features_spikes_highlighted_callback)
        view.spikesSelected.connect(
            self.features_spikes_selected_callback)
        view.projectionChanged.connect(
            self.features_projection_changed_callback)
        self.views['FeatureView'].append(view)
            
    def add_ipython_view(self):
        view = self.create_view(vw.IPythonView,
            index=len(self.views['IPythonView']),
            position=QtCore.Qt.BottomDockWidgetArea,
            floating=True)
        # Create namespace for the interactive session.
        namespace = dict(
            window=self,
            select=self.get_view('ClusterView').select,
            loader=self.loader,
            stats=self.statscache,
            )
        view.set_data(**namespace)
        # Load all .py files in the code directory.
        paths = USERPREF['ipython_import_paths'] or []
        if isinstance(paths, basestring):
            paths = [paths]
        for path in paths:
            path = os.path.realpath(os.path.expanduser(path))
            if os.path.exists(path):
                files = [file for file in os.listdir(path) if file.endswith('.py')]
                for file in files:
                    log.debug("Running {0:s}".format(file))
                    view.run_file(os.path.join(path, file))
        self.views['IPythonView'].append(view)
        
    def add_correlograms_view(self):
        view = self.create_view(vw.CorrelogramsView,
            index=len(self.views['CorrelogramsView']),
            position=QtCore.Qt.RightDockWidgetArea,)
        self.views['CorrelogramsView'].append(view)
            
    def get_view(self, name, index=0):
        views = self.views[name] 
        if not views:
            return None
        else:
            return views[index]
            
    def get_views(self, name):
        return self.views[name]
            
    def create_views(self):
        """Create all views at initialization."""
        
        # Create the default layout.
        self.views = dict(
            ClusterView=[],
            ProjectionView=[],
            SimilarityMatrixView=[],
            WaveformView=[],
            FeatureView=[],
            CorrelogramsView=[],
            IPythonView=[],
            )
        
        self.add_projection_view()
        self.add_cluster_view()
        self.add_similarity_matrix_view()
            
        self.splitDockWidget(
            self.get_view('ProjectionView').parentWidget(), 
            self.get_view('ClusterView').parentWidget(), 
            QtCore.Qt.Vertical
            )
            
        self.splitDockWidget(
            self.get_view('ClusterView').parentWidget(), 
            self.get_view('SimilarityMatrixView').parentWidget(), 
            QtCore.Qt.Vertical
            )
            
        self.add_waveform_view()
        self.add_feature_view()
            
        self.splitDockWidget(
            self.get_view('WaveformView').parentWidget(), 
            self.get_view('FeatureView').parentWidget(), 
            QtCore.Qt.Horizontal
            )
            
        self.add_correlograms_view()
            
        self.splitDockWidget(
            self.get_view('FeatureView').parentWidget(), 
            self.get_view('CorrelogramsView').parentWidget(), 
            QtCore.Qt.Vertical
            )
    
    def dock_widget_closed(self, dock):
        for key in self.views.keys():
            views = self.views[key]
            for i in xrange(len(views)):
                if views[i].parent() == dock:
                    # self.views[view][i] = None
                    del self.views[key][i]
    
    
    # Update view methods.
    # --------------------
    def update_cluster_view(self):
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
    
    def update_projection_view(self):
        """Update the cluster view using the data stored in the loader
        object."""
        data = dict(
            nchannels=self.loader.nchannels,
            fetdim=self.loader.fetdim,
            nextrafet=self.loader.nextrafet,
        )
        self.get_view('ProjectionView').set_data(**data)
    
    def update_waveform_view(self, autozoom=None):
        data = dict(
            waveforms=self.loader.get_waveforms(),
            clusters=self.loader.get_clusters(),
            cluster_colors=self.loader.get_cluster_colors(
                wizard=self.wizard_active),
            clusters_selected=self.loader.get_clusters_selected(),
            masks=self.loader.get_masks(),
            geometrical_positions=self.loader.get_probe(),
            autozoom=autozoom,
        )
        [view.set_data(**data) for view in self.get_views('WaveformView')]
    
    def update_feature_view(self, autozoom=None):
        data = dict(
            features=self.loader.get_some_features(),
            masks=self.loader.get_masks(),
            clusters=self.loader.get_clusters(),
            clusters_selected=self.loader.get_clusters_selected(),
            cluster_colors=self.loader.get_cluster_colors(
                wizard=self.wizard_active),
            nchannels=self.loader.nchannels,
            fetdim=self.loader.fetdim,
            nextrafet=self.loader.nextrafet,
            freq=self.loader.freq,
            autozoom=autozoom,
            duration=self.loader.get_duration(),
            alpha_selected=USERPREF.get('feature_selected_alpha', .75),
            alpha_background=USERPREF.get('feature_background_alpha', .1),
            time_unit=USERPREF['features_info_time_unit'] or 'second',
        )
        [view.set_data(**data) for view in self.get_views('FeatureView')]
        
    def update_correlograms_view(self):
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
            cluster_colors=self.loader.get_cluster_colors(
                wizard=self.wizard_active),
            ncorrbins=self.loader.ncorrbins,
            corrbin=self.loader.corrbin,
        )
        [view.set_data(**data) for view in self.get_views('CorrelogramsView')]
    
    def update_similarity_matrix_view(self):
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
    
    def update_views(self):
        self.update_feature_view()
        self.update_waveform_view()
        self.update_correlograms_view()
    
    
    # Geometry.
    # ---------
    def save_geometry(self):
        """Save the arrangement of the whole window."""
        # SETTINGS['main_window.dockwidgets'] = {name: len(self.get_views(name))
            # for name in self.views.keys()}
        SETTINGS['main_window.geometry'] = encode_bytearray(
            self.saveGeometry())
        SETTINGS['main_window.state'] = encode_bytearray(self.saveState())
        
    def restore_geometry(self):
        """Restore the arrangement of the whole window."""
        # count = SETTINGS['main_window.dockwidgets']
        g = SETTINGS['main_window.geometry']
        s = SETTINGS['main_window.state']
        if s:
            self.restoreState(decode_bytearray(s))
        if g:
            self.restoreGeometry(decode_bytearray(g))
    
    
    # Event handlers.
    # ---------------
    def force_key_release(self):
        """HACK: force release of Ctrl when opening a dialog with a keyboard
        shortcut."""
        self.keyReleaseEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyRelease,
            QtCore.Qt.Key_Control, QtCore.Qt.NoModifier))
    
    def contextMenuEvent(self, e):
        """Disable the context menu in the main window."""
        return
        
    def keyPressEvent(self, e):
        super(MainWindow, self).keyPressEvent(e)
        for views in self.views.values():
            [view.keyPressEvent(e) for view in views]
        
    def keyReleaseEvent(self, e):
        super(MainWindow, self).keyReleaseEvent(e)
        for views in self.views.values():
            [view.keyReleaseEvent(e) for view in views]
            
    def closeEvent(self, e):
        prompt_save_on_exit = USERPREF['prompt_save_on_exit']
        if prompt_save_on_exit is None:
            prompt_save_on_exit = True
        if self.need_save and prompt_save_on_exit:
            reply = QtGui.QMessageBox.question(self, 'Save',
            "Do you want to save?",
            (
            QtGui.QMessageBox.Save | 
             QtGui.QMessageBox.Close |
             QtGui.QMessageBox.Cancel 
             ),
            QtGui.QMessageBox.Save)
            if reply == QtGui.QMessageBox.Save:
                folder = SETTINGS.get('main_window.last_data_file')
                self.loader.save()
            elif reply == QtGui.QMessageBox.Cancel:
                e.ignore()
                return
            elif reply == QtGui.QMessageBox.Close:
                pass
        
        # Save the window geometry when closing the software.
        self.save_geometry()
        
        # End the threads.
        self.join_threads()
        
        # Close the loader.
        self.loader.close()
        
        # Close all views.
        for views in self.views.values():
            for view in views:
                if hasattr(view, 'closeEvent'):
                    view.closeEvent(e)
        
        # Close the logger file.
        if self.dolog:
            close_file_logger()
        
        # Close the main window.
        return super(MainWindow, self).closeEvent(e)
            
            
            
    def sizeHint(self):
        return QtCore.QSize(1200, 800)
        
        
# -----------------------------------------------------------------------------
# File logger
# -----------------------------------------------------------------------------
def create_file_logger():
    global LOGGER_FILE
    LOGFILENAME = get_global_path('logfile.txt')
    LOGGER_FILE = FileLogger(LOGFILENAME, name='file', 
        level=USERPREF['loglevel_file'])
    register(LOGGER_FILE)

def close_file_logger():
    unregister(LOGGER_FILE)
    
    