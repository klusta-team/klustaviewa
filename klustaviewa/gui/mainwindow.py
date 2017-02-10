"""Main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import time
from StringIO import StringIO
import os
import sys
import inspect
import logging
from collections import OrderedDict
from functools import partial
import webbrowser

import pandas as pd
import numpy as np
import numpy.random as rnd
from qtools import QtGui, QtCore
from qtools import inprocess, inthread, QT_BINDING

import klustaviewa.views as vw
from klustaviewa.gui.icons import get_icon
from klustaviewa.control.controller import Controller
from klustaviewa.wizard.wizard import Wizard
from kwiklib.dataio.tools import get_array
from kwiklib.dataio import KlustersLoader, KwikLoader, read_clusters
from klustaviewa.gui.buffer import Buffer
from klustaviewa.gui.dock import ViewDockWidget, DockTitleBar
from klustaviewa.stats.cache import StatsCache
from klustaviewa.stats.correlograms import NCORRBINS_DEFAULT, CORRBIN_DEFAULT
from klustaviewa.stats.correlations import normalize
from kwiklib.utils import logger as log
from kwiklib.utils.logger import FileLogger, register, unregister
from kwiklib.utils.persistence import encode_bytearray, decode_bytearray
from klustaviewa import USERPREF
from klustaviewa import SETTINGS
from klustaviewa import APPNAME, ABOUT, get_global_path
from klustaviewa import get_global_path
from klustaviewa.gui.threads import ThreadedTasks, OpenTask
from klustaviewa.gui.taskgraph import TaskGraph
import rcicons


# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, dolog=True, filename=None, shank=None):
        self.views = {}
        super(MainWindow, self).__init__(parent)
        self.views = {}

        # HACK: display the icon in Windows' taskbar.
        if os.name == 'nt':
            try:
                import ctypes
                myappid = 'klustateam.klustaviewa'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except:
                pass

        self.dolog = dolog
        if self.dolog:
            create_file_logger()

        self.initialize_view_logger()

        log.debug("Using {0:s}.".format(QT_BINDING))

        # Main window options.
        self.move(50, 50)
        self.setWindowTitle('KlustaViewa')

        # Focus options.
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setMouseTracking(True)

        # Dock widgets options.
        self.setDockNestingEnabled(True)
        self.setAnimated(False)
        self.setWindowIcon(get_icon('logo'))

        # Initialize some variables.
        self.statscache = None
        # self.loader = KlustersLoader()
        self.loader = KwikLoader(userpref=USERPREF)
        self.loader.progressReported.connect(self.open_progress_reported)
        self.loader.saveProgressReported.connect(self.save_progress_reported)
        self.wizard = Wizard()
        self.controller = None
        self.spikes_highlighted = []
        self.spikes_selected = []
        self._wizard = False
        self.is_file_open = False
        self.need_save = False
        self.taskgraph = TaskGraph(self)
        self.busy_cursor = QtGui.QCursor(QtCore.Qt.BusyCursor)
        self.normal_cursor = QtGui.QCursor(QtCore.Qt.ArrowCursor)
        self.is_busy = False
        self.override_color = False
        self.computing_correlograms = False
        self.computing_matrix = False

        # Create the main window.
        self.create_views()
        self.create_file_actions()
        self.create_edit_actions()
        self.create_view_actions()
        self.create_correlograms_actions()
        self.create_control_actions()
        self.create_wizard_actions()
        self.create_help_actions()
        self.create_menu()
        self.create_toolbar()
        self.create_open_progress_dialog()
        self.create_save_progress_dialog()
        self.create_threads()

        # Update action enabled/disabled property.
        self.update_action_enabled()

        # Show the main window.
        self.set_styles()
        self.restore_geometry()

        # Automatically load a file upon startup if requested.
        if filename:
            filename = os.path.realpath(filename)
            self.open_task.open(self.loader, filename, shank)

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

    def set_busy(self, computing_correlograms=None, computing_matrix=None):
        if computing_correlograms is not None:
            self.computing_correlograms = computing_correlograms
        if computing_matrix is not None:
            self.computing_matrix = computing_matrix
        busy = self.computing_correlograms or self.computing_matrix
        if busy:
            self.set_busy_cursor()
            self.is_busy = True
        else:
            self.set_normal_cursor()
            self.is_busy = False

    def initialize_view_logger(self):
        # Initialize the view logger.
        viewlogger = vw.ViewLogger(name='viewlogger', fmt='%(message)s',
            level=USERPREF['loglevel'], print_caller=False)
        register(viewlogger)
        viewlogger.outlog.writeRequested.connect(self.log_view_write_callback)
        self.view_logger_text = StringIO()


    # Actions.
    # --------
    def add_action(self, name, text, callback=None, shortcut=None,
            checkable=False, checked=False, icon=None):
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
        action.setChecked(checked)
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

        self.add_action('switch', 'S&witch shank')
        self.add_action('import', '&Import clustering')
        self.add_action('reset', '&Reset clustering')
        self.add_action('save', '&Save', shortcut='Ctrl+S', icon='save')
        # self.add_action('renumber', 'Save &renumbered')
        self.add_action('close', '&Close file')

    def create_edit_actions(self):
        # Undo/redo actions.
        self.add_action('undo', '&Undo', shortcut='Ctrl+Z', icon='undo')
        self.add_action('redo', '&Redo', shortcut='Ctrl+Y', icon='redo')

        # self.add_action('reset', 'Re&set')

        # Quit action.
        self.add_action('quit', '&Quit', shortcut='Ctrl+Q')

    def create_view_actions(self):
        self.add_action('add_feature_view', 'Add &FeatureView')
        self.add_action('add_waveform_view', 'Add &WaveformView')
        self.add_action('add_similarity_matrix_view',
            'Add &SimilarityMatrixView')
        self.add_action('add_correlograms_view', 'Add &CorrelogramsView')
        self.add_action('add_ipython_view', 'Add &IPythonView')
        self.add_action('add_log_view', 'Add &LogView')
        # self.add_action('add_trace_view', 'Add &TraceView')
        self.add_action('reset_views', '&Reset views')
        self.add_action('toggle_fullscreen', 'Toggle fullscreen', shortcut='F')

        self.add_action('override_color', 'Override cluster &color',
            icon='override_color')#, shortcut='C')

    def create_control_actions(self):
        self.add_action('merge', '&Merge', shortcut='G', icon='merge')
        self.add_action('split', '&Split', shortcut='K', icon='split')
        self.add_action('recluster', '&Recluster', shortcut='CTRL+R')

    def create_correlograms_actions(self):
        self.add_action('change_ncorrbins', 'Change time &window')
        self.add_action('change_corrbin', 'Change &bin size')

        self.add_action('change_corr_normalization', 'Change &normalization')

    def create_wizard_actions(self):
        self.add_action('reset_navigation', '&Reinitialize wizard',
            shortcut='CTRL+ALT+Space')
        self.add_action('automatic_projection', '&Automatic projection',
            checkable=True, checked=True)
        self.add_action('change_candidate_color',
            'Change &color of the closest match',
            shortcut='C')

        self.add_action('previous_candidate', '&Previous closest match',
            shortcut='SHIFT+Space')
        self.add_action('next_candidate', '&Skip closest match',
            shortcut='Space')
        self.add_action('skip_target', '&Skip best unsorted',
            # shortcut='Space'
            )
        self.add_action('delete_candidate', 'Move closest match to &MUA',
            shortcut='CTRL+M')
        self.add_action('delete_candidate_noise', 'Move closest match to &noise',
            shortcut='CTRL+N')

        self.add_action('next_target', 'Move best unsorted to &good',
            shortcut='ALT+G')
        self.add_action('delete_target', 'Move best unsorted to &MUA',
            shortcut='ALT+M')
        self.add_action('delete_target_noise', 'Move best unsorted to &noise',
            shortcut='ALT+N')

        self.add_action('delete_both', 'Move &both to MUA',
            shortcut='CTRL+ALT+M')
        self.add_action('delete_both_noise', 'Move both to noise',
            shortcut='CTRL+ALT+N')

    def create_help_actions(self):
        self.add_action('about', '&About')
        self.add_action('manual', 'Show &manual')
        self.add_action('shortcuts', 'Show &shortcuts')
        self.add_action('open_preferences', '&Open preferences')
        self.add_action('refresh_preferences', '&Refresh preferences')

    def create_menu(self):
        # File menu.
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_last_action)
        file_menu.addSeparator()
        file_menu.addAction(self.reset_action)
        # file_menu.addAction(self.import_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        # file_menu.addAction(self.renumber_action)
        file_menu.addSeparator()
        file_menu.addAction(self.switch_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_action)
        file_menu.addAction(self.quit_action)

        # Edit menu.
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        # View menu.
        views_menu = self.menuBar().addMenu("&View")
        views_menu.addAction(self.add_feature_view_action)
        views_menu.addAction(self.add_waveform_view_action)
        views_menu.addAction(self.add_correlograms_view_action)
        views_menu.addAction(self.add_similarity_matrix_view_action)
        # views_menu.addAction(self.add_trace_view_action)
        views_menu.addSeparator()
        views_menu.addAction(self.add_log_view_action)
        if vw.IPYTHON:
            views_menu.addAction(self.add_ipython_view_action)
            views_menu.addSeparator()
        views_menu.addAction(self.override_color_action)
        views_menu.addSeparator()
        views_menu.addAction(self.reset_views_action)
        views_menu.addAction(self.toggle_fullscreen_action)

        # Correlograms menu.
        correlograms_menu = self.menuBar().addMenu("&Correlograms")
        correlograms_menu.addAction(self.change_ncorrbins_action)
        correlograms_menu.addAction(self.change_corrbin_action)
        correlograms_menu.addSeparator()
        correlograms_menu.addAction(self.change_corr_normalization_action)

        # Actions menu.
        actions_menu = self.menuBar().addMenu("&Actions")

        actions_menu.addSeparator()
        actions_menu.addAction(self.get_view('ClusterView').move_to_mua_action)
        actions_menu.addAction(self.get_view('ClusterView').move_to_noise_action)
        actions_menu.addSeparator()
        actions_menu.addAction(self.merge_action)
        actions_menu.addAction(self.split_action)
        # actions_menu.addSeparator()
        # actions_menu.addAction(self.recluster_action)

        # Wizard menu.
        wizard_menu = self.menuBar().addMenu("&Wizard")
        # Previous/skip candidate.
        wizard_menu.addAction(self.next_candidate_action)
        wizard_menu.addAction(self.previous_candidate_action)
        wizard_menu.addSeparator()
        wizard_menu.addAction(self.skip_target_action)
        wizard_menu.addSeparator()
        # Good group.
        # wizard_menu.addSeparator()
        # Delete.
        wizard_menu.addAction(self.delete_candidate_action)
        wizard_menu.addAction(self.delete_candidate_noise_action)
        wizard_menu.addSeparator()
        wizard_menu.addAction(self.next_target_action)
        wizard_menu.addAction(self.delete_target_action)
        wizard_menu.addAction(self.delete_target_noise_action)
        wizard_menu.addSeparator()
        wizard_menu.addAction(self.delete_both_action)
        wizard_menu.addAction(self.delete_both_noise_action)
        wizard_menu.addSeparator()
        # Misc.
        wizard_menu.addAction(self.change_candidate_color_action)
        wizard_menu.addAction(self.automatic_projection_action)
        wizard_menu.addAction(self.reset_navigation_action)

        # Help menu.
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.open_preferences_action)
        help_menu.addAction(self.refresh_preferences_action)
        help_menu.addSeparator()
        help_menu.addAction(self.shortcuts_action)
        help_menu.addAction(self.manual_action)
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
        # self.toolbar.addSeparator()
        # self.toolbar.addAction(self.override_color_action)

        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolbar)

    def create_open_progress_dialog(self):
        self.open_progress = QtGui.QProgressDialog("Converting to Kwik...",
            "Cancel", 0, 0, self, QtCore.Qt.Tool)
        self.open_progress.setWindowModality(QtCore.Qt.WindowModal)
        self.open_progress.setValue(0)
        self.open_progress.setWindowTitle('Loading')
        self.open_progress.setCancelButton(None)
        self.open_progress.setMinimumDuration(0)

    def create_save_progress_dialog(self):
        self.save_progress = QtGui.QProgressDialog("Saving...",
            "Cancel", 0, 0, self, QtCore.Qt.Tool)
        self.save_progress.setWindowModality(QtCore.Qt.WindowModal)
        self.save_progress.setValue(0)
        self.save_progress.setWindowTitle('Saving')
        self.save_progress.setCancelButton(None)
        self.save_progress.setMinimumDuration(0)


    # Action enabled.
    # ---------------
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


    # View methods.
    # -------------
    def create_view(self, view_class, position=None,
        closable=True, floatable=True, index=0, floating=None, title=None,
        **kwargs):
        """Add a widget to the main window."""
        view = view_class(self, getfocus=False)
        view.set_data(**kwargs)

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

        dockwidget.visibilityChanged.connect(partial(
            self.dock_visibility_changed_callback, view))

        if position is not None:
            # Add the dock widget to the main window.
            self.addDockWidget(position, dockwidget)

        if floating is not None:
            dockwidget.setFloating(floating)

        if title is None:
            title = view_class.__name__
        dockwidget.setTitleBarWidget(DockTitleBar(dockwidget, title))

        # Return the view widget.
        return view

    def add_cluster_view(self, do_update=None, floating=False):
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

        if do_update:
            self.taskgraph.update_cluster_view()

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

    def add_similarity_matrix_view(self, do_update=None, floating=False):
        # Try restoring the last view if it exists and it is hidden, and if
        # successfully restored, do nothing more. Otherwise, need to create
        # a new view.
        if self.restore_last_view('SimilarityMatrixView'):
            return
        view = self.create_view(vw.SimilarityMatrixView,
            index=len(self.views['SimilarityMatrixView']),
            position=QtCore.Qt.LeftDockWidgetArea,
            floating=floating)
        view.clustersSelected.connect(self.cluster_pair_selected_callback)
        self.views['SimilarityMatrixView'].append(view)
        if do_update and self.is_file_open:
            self.taskgraph.update_similarity_matrix_view()

    def add_waveform_view(self, do_update=None, floating=False):
        view = self.create_view(vw.WaveformView,
            index=len(self.views['WaveformView']),
            position=QtCore.Qt.RightDockWidgetArea,
            floating=floating)
        view.spikesHighlighted.connect(
            self.waveform_spikes_highlighted_callback)
        view.boxClicked.connect(self.waveform_box_clicked_callback)
        self.views['WaveformView'].append(view)
        if do_update and self.is_file_open and self.loader.has_selection():
            self.taskgraph.update_waveform_view()

    def add_feature_view(self, do_update=None, floating=False):
        view = self.create_view(vw.FeatureProjectionView,
            index=len(self.views['FeatureView']),
            position=QtCore.Qt.RightDockWidgetArea,
            floating=floating,
            title='FeatureView')
        view.spikesHighlighted.connect(
            self.features_spikes_highlighted_callback)
        view.spikesSelected.connect(
            self.features_spikes_selected_callback)
        self.views['FeatureView'].append(view)
        if do_update and self.is_file_open and self.loader.has_selection():
            self.taskgraph.update_feature_view()

    def add_ipython_view(self, floating=None):
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
            wizard=self.wizard,
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

    def add_log_view(self, floating=None):
        if len(self.views['LogView']) >= 1:
            return
        view = self.create_view(vw.LogView,
            text=self.view_logger_text.getvalue(),
            position=QtCore.Qt.BottomDockWidgetArea,
            floating=True)
        self.views['LogView'].append(view)

    def log_view_write_callback(self, message):
        view = self.get_view('LogView')
        if view:
            view.append(message)
        self.view_logger_text.write(message)

    def add_correlograms_view(self, do_update=None, floating=False):
        view = self.create_view(vw.CorrelogramsView,
            index=len(self.views['CorrelogramsView']),
            position=QtCore.Qt.RightDockWidgetArea,
            floating=floating)
        self.views['CorrelogramsView'].append(view)
        if do_update and self.is_file_open and self.loader.has_selection():
            self.taskgraph.update_correlograms_view()

    # def add_trace_view(self, do_update=None, floating=None):
    #     # if len(self.views['TraceView']) >= 1:
    #     #     return
    #     view = self.create_view(vw.TraceView,
    #         index=len(self.views['TraceView']),
    #         position=QtCore.Qt.BottomDockWidgetArea,
    #         floating=True)
    #     self.views['TraceView'].append(view)
    #     if do_update and self.is_file_open:
    #         self.taskgraph.update_trace_view()

    def get_view(self, name, index=0):
        views = self.views.get(name, [])
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
            SimilarityMatrixView=[],
            WaveformView=[],
            FeatureView=[],
            CorrelogramsView=[],
            IPythonView=[],
            TraceView=[],
            LogView=[],
            )

        count = SETTINGS['main_window.views']
        if count is None:
            self.create_default_views()
        else:
            self.create_custom_views(count)

    def create_default_views(self, do_update=None, floating=False):
        self.add_cluster_view(do_update=do_update, floating=floating)
        self.add_similarity_matrix_view(do_update=do_update, floating=floating)

        self.splitDockWidget(
            self.get_view('ClusterView').parentWidget(),
            self.get_view('SimilarityMatrixView').parentWidget(),
            QtCore.Qt.Vertical
            )

        self.add_waveform_view(do_update=do_update, floating=floating)
        self.add_feature_view(do_update=do_update, floating=floating)

        self.splitDockWidget(
            self.get_view('WaveformView').parentWidget(),
            self.get_view('FeatureView').parentWidget(),
            QtCore.Qt.Horizontal
            )

        self.add_correlograms_view(do_update=do_update, floating=floating)

        self.splitDockWidget(
            self.get_view('FeatureView').parentWidget(),
            self.get_view('CorrelogramsView').parentWidget(),
            QtCore.Qt.Vertical
            )

    def create_custom_views(self, count):
        [self.add_cluster_view() for _ in xrange(count.get('ClusterView', 0))]
        [self.add_similarity_matrix_view() for _ in xrange(count.get('SimilarityMatrixView', 0))]
        [self.add_waveform_view() for _ in xrange(count.get('WaveformView', 0))]
        [self.add_feature_view() for _ in xrange(count.get('FeatureView', 0))]
        [self.add_log_view() for _ in xrange(count.get('LogView', 0))]
        [self.add_ipython_view() for _ in xrange(count.get('IPythonView', 0))]
        [self.add_correlograms_view() for _ in xrange(count.get('CorrelogramsView', 0))]
        #[self.add_trace_view() for _ in xrange(count.get('TraceView', 0))]

    def dock_widget_closed(self, dock):
        for key in self.views.keys():
            self.views[key] = [view for view in self.views[key] if view.parent() != dock]


    # Threads.
    # --------
    def create_threads(self):
        # Create the external threads.
        self.open_task = inthread(OpenTask)()
        self.open_task.dataOpened.connect(self.open_done)
        self.open_task.dataSaved.connect(self.save_done)
        self.open_task.dataOpenFailed.connect(self.open_failed)

    def join_threads(self):
         self.open_task.join()
         self.taskgraph.join()


    # File menu callbacks.
    # --------------------
    def open_callback(self, checked=None):
        # HACK: Force release of Ctrl key.
        self.force_key_release()

        folder = SETTINGS['main_window.last_data_dir']
        path = QtGui.QFileDialog.getOpenFileName(self,
            "Open a .kwik file", folder)[0]
        # If a file has been selected, open it.
        if path:
            # Launch the loading task in the background asynchronously.
            self._path = path
            self.open_task.open(self.loader, path)
            # Save the folder.
            folder = os.path.dirname(path)
            SETTINGS['main_window.last_data_dir'] = folder
            SETTINGS['main_window.last_data_file'] = path

    def import_callback(self, checked=None):
        folder = SETTINGS['main_window.last_data_dir']
        path = QtGui.QFileDialog.getOpenFileName(self,
            "Open a .kwik file", folder)[0]
        # If a file has been selected, open it.
        if path and self.loader is not None:
            clu = read_clusters(path)
            # TODO
            self.open_done()

    def save_callback(self, checked=None):
        self.open_task.save(self.loader)

    def reset_callback(self, checked=None):
        # reply = QtGui.QMessageBox.question(self, 'Reset clustering',
            # "Do you *really* want to erase permanently your manual clustering and reset it to the original (automatic) clustering? You won't be able to undo this operation!",
            # (
            # QtGui.QMessageBox.Yes |
             # QtGui.QMessageBox.Cancel
             # ),
            # QtGui.QMessageBox.Cancel)
        # if reply == QtGui.QMessageBox.Yes:
        clustering_name, ok = QtGui.QInputDialog.getText(self, "Clustering name", "Copy from (you'll lose the current clustering):",
                                   QtGui.QLineEdit.Normal, 'original')
        if ok:
            self.loader.copy_clustering(clustering_from=clustering_name,
                                        clustering_to='main')
            # Reload the file.
            self.loader.close()
            self.open_task.open(self.loader, self._path)
        # elif reply == QtGui.QMessageBox.Cancel:
            # return

    # def renumber_callback(self, checked=None):
        # # folder = SETTINGS.get('main_window.last_data_file')
        # self.loader.save(renumber=True)
        # # self.need_save = False
        # self.open_last_callback()

    def open_last_callback(self, checked=None):
        path = SETTINGS['main_window.last_data_file']
        if path:
            self._path = path
            self.open_task.open(self.loader, path)

    def close_callback(self, checked=None):
        # clusters = self.get_view('ClusterView').selected_clusters()
        # if clusters:
            # self.get_view('ClusterView').unselect()
            # time.sleep(.25)

        # Clear the views.
        self.clear_view('ClusterView')
        self.clear_view('SimilarityMatrixView')
        self.clear_view('FeatureView')
        self.clear_view('WaveformView')
        self.clear_view('CorrelogramsView')
        self.clear_view('TraceView')

        self.loader.close()
        self.is_file_open = False

    def switch_callback(self, checked=None):
        shank, ok = QtGui.QInputDialog.getInt(self,
            "Shank number", "Shank number:",
            self.loader.shank,
            min(self.loader.shanks),
            max(self.loader.shanks),
            1)
        if ok:
            if shank in self.loader.shanks:
                self.loader.set_shank(shank)
                self.open_done()
            else:
                QtGui.QMessageBox.warning(self, "Wrong shank number",
                ("The selected shank '{0:d}' is not in "
                 "the list of shanks: {1:s}.").format(shank,
                                                    str(self.loader.shanks)),
                    QtGui.QMessageBox.Ok, QtGui.QMessageBox.Ok)

    def clear_view(self, view_name):
        for v in self.get_views(view_name):
            v.set_data()
            if hasattr(v, 'clear'):
                v.clear()

    def quit_callback(self, checked=None):
        self.close()


    # Open callbacks.
    # --------------
    def open_done(self):
        self.is_file_open = True
        self.setWindowTitle('KlustaViewa: {0:s}'.format(
            os.path.basename(self.loader.filename)
        ))

        register(FileLogger(self.loader.log_filename, name='kwik',
                 level=logging.INFO))

        # Start the selection buffer.
        self.buffer = Buffer(self,
            # delay_timer=.1, delay_buffer=.2
            delay_timer=USERPREF['delay_timer'],
            delay_buffer=USERPREF['delay_buffer']
            )
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
        self.statscache = StatsCache(SETTINGS.get('correlograms.ncorrbins', NCORRBINS_DEFAULT))
        # Update stats cache in IPython view.
        ipython = self.get_view('IPythonView')
        if ipython:
            ipython.set_data(stats=self.statscache)

        # Initialize the wizard.
        self.wizard = Wizard()

        # Update the task graph.
        self.taskgraph.set(self)
        # self.taskgraph.update_projection_view()
        self.taskgraph.update_cluster_view()
        self.taskgraph.compute_similarity_matrix()
        # self.taskgraph.update_trace_view()

    def open_failed(self, message):
        self.open_progress.setValue(0)
        QtGui.QMessageBox.warning(self, "Error while opening the file",
            "An error occurred: {0:s}".format(message),
            QtGui.QMessageBox.Ok, QtGui.QMessageBox.Ok)

    def open_progress_reported(self, progress, progress_max):
        self.open_progress.setMaximum(progress_max)
        self.open_progress.setValue(progress)

    def save_progress_reported(self, progress, progress_max):
        self.save_progress.setMaximum(progress_max)
        self.save_progress.setValue(progress)

    def save_done(self):
        self.need_save = False


    # Selection methods.
    # ------------------
    def buffer_accepted_callback(self, (clusters, wizard)):
        self._wizard = wizard
        # The wizard boolean specifies whether the autozoom is activated or not.
        self.taskgraph.select(clusters, wizard and
            self.automatic_projection_action.isChecked(), )

    def clusters_selected_callback(self, clusters, wizard=False):
        self.buffer.request((clusters, wizard))

    def cluster_pair_selected_callback(self, clusters):
        """Callback when the user clicks on a pair in the
        SimilarityMatrixView."""
        self.get_view('ClusterView').select(clusters,)


    # Views menu callbacks.
    # ---------------------
    def add_feature_view_callback(self, checked=None):
        self.add_feature_view(do_update=True, floating=True)

    def add_waveform_view_callback(self, checked=None):
        self.add_waveform_view(do_update=True, floating=True)

    def add_similarity_matrix_view_callback(self, checked=None):
        self.add_similarity_matrix_view(do_update=True, floating=True)

    def add_correlograms_view_callback(self, checked=None):
        self.add_correlograms_view(do_update=True, floating=True)

    # def add_trace_view_callback(self, checked=None):
    #         self.add_trace_view(do_update=True, floating=True)

    def add_log_view_callback(self, checked=None):
        self.add_log_view()

    def add_ipython_view_callback(self, checked=None):
        self.add_ipython_view()

    def reset_views_callback(self, checked=None):
        # Delete all views.
        for key, views in self.views.iteritems():
            for view in views:
                self.removeDockWidget(view.parent())
            self.views[key] = []
        # Re-create the default views.
        self.create_default_views(do_update=self.is_file_open, floating=False)

    def toggle_fullscreen_callback(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()


    # Override color callback.
    # ------------------------
    def override_color_callback(self, checked=None):
        self.override_color = not self.override_color
        self.taskgraph.override_color(self.override_color)


    # Correlograms callbacks.
    # -----------------------
    def change_ncorrbins_callback(self, checked=None):
        if not self.loader:
            return
        corrbin = SETTINGS.get('correlograms.corrbin', CORRBIN_DEFAULT)
        ncorrbins = SETTINGS.get('correlograms.ncorrbins', NCORRBINS_DEFAULT)
        duration = corrbin * ncorrbins
        # duration = self.loader.get_correlogram_window()
        duration_new, ok = QtGui.QInputDialog.getDouble(self,
            "Correlograms time window", "Half width (ms):",
            duration / 2 * 1000, 1, 100000, 1)
        if ok:
            duration_new = duration_new * .001 * 2
            ncorrbins_new = 2 * int(np.ceil(.5 * duration_new / corrbin))
            # ncorrbins_new = int(duration_new / corrbin * .001)
            SETTINGS['correlograms.ncorrbins'] = ncorrbins_new
            self.taskgraph.change_correlograms_parameters(ncorrbins=ncorrbins_new)

    def recluster_callback(self, checked=None):
        self.taskgraph.recluster()

    def change_corrbin_callback(self, checked=None):
        if not self.loader:
            return
        # ncorrbins = self.loader.ncorrbins
        # corrbin = self.loader.corrbin
        # duration = self.loader.get_correlogram_window()
        corrbin = SETTINGS.get('correlograms.corrbin', CORRBIN_DEFAULT)
        ncorrbins = SETTINGS.get('correlograms.ncorrbins', NCORRBINS_DEFAULT)
        duration = corrbin * ncorrbins
        corrbin_new, ok = QtGui.QInputDialog.getDouble(self,
            "Correlograms bin size", "Bin size (ms):",
            corrbin * 1000, .01, 1000, 2)
        if ok:
            corrbin_new = corrbin_new * .001
            ncorrbins_new = 2 * int(np.ceil(.5 * duration/ corrbin_new))
            SETTINGS['correlograms.corrbin'] = corrbin_new
            SETTINGS['correlograms.ncorrbins'] = ncorrbins_new
            self.taskgraph.change_correlograms_parameters(corrbin=corrbin_new,
                ncorrbins=ncorrbins_new)

    def change_corr_normalization_callback(self, checked=None):
        [view.change_normalization() for view in self.get_views('CorrelogramsView')]


    # Actions callbacks.
    # ------------------
    def merge_callback(self, checked=None):
        if self.is_busy:
            return
        self.need_save = True
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        self.taskgraph.merge(clusters, self._wizard)
        self.update_action_enabled()

    def split_callback(self, checked=None):
        if self.is_busy:
            return
        self.need_save = True
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        spikes_selected = self.spikes_selected
        # Cancel the selection after the split.
        self.spikes_selected = []
        self.taskgraph.split(clusters, spikes_selected, self._wizard)
        self.update_action_enabled()

    def undo_callback(self, checked=None):
        if self.is_busy:
            return
        self.taskgraph.undo(self._wizard)
        self.update_action_enabled()

    def redo_callback(self, checked=None):
        if self.is_busy:
            return
        self.taskgraph.redo(self._wizard)
        self.update_action_enabled()

    def cluster_color_changed_callback(self, cluster, color):
        self.taskgraph.cluster_color_changed(cluster, color, self._wizard)
        self.update_action_enabled()

    def group_color_changed_callback(self, group, color):
        self.taskgraph.group_color_changed(group, color)
        self.update_action_enabled()

    def group_renamed_callback(self, group, name):
        self.taskgraph.group_renamed(group, name)
        self.update_action_enabled()

    def clusters_moved_callback(self, clusters, group):
        self.taskgraph.clusters_moved(clusters, group)
        self.update_action_enabled()

    def group_removed_callback(self, group):
        self.taskgraph.group_removed(group)
        self.update_action_enabled()

    def group_added_callback(self, group, name, color):
        self.taskgraph.group_added(group, name, color)
        self.update_action_enabled()


    # Wizard callbacks.
    # -----------------
    def reset_navigation_callback(self, checked=None):
        self.taskgraph.wizard_reset()

    def previous_candidate_callback(self, checked=None):
        # Previous candidate.
        self.taskgraph.wizard_previous_candidate()

    def next_candidate_callback(self, checked=None):
        if self.is_busy:
            return
        # Skip candidate.
        self.taskgraph.wizard_next_candidate()

    def skip_target_callback(self, checked=None):
        if self.is_busy:
            return
        # Skip target.
        self.taskgraph.wizard_skip_target()

    def next_target_callback(self, checked=None):
        if self.is_busy:
            return
        # Move target to Good group, and select next target.
        self.taskgraph.wizard_move_and_next('target', 2)

    def delete_candidate_noise_callback(self, checked=None):
        self.taskgraph.wizard_move_and_next('candidate', 0)

    def delete_candidate_callback(self, checked=None):
        self.taskgraph.wizard_move_and_next('candidate', 1)

    def delete_target_noise_callback(self, checked=None):
        self.taskgraph.wizard_move_and_next('target', 0)

    def delete_target_callback(self, checked=None):
        self.taskgraph.wizard_move_and_next('target', 1)

    def delete_both_noise_callback(self, checked=None):
        self.taskgraph.wizard_move_and_next('both', 0)

    def delete_both_callback(self, checked=None):
        self.taskgraph.wizard_move_and_next('both', 1)

    def change_candidate_color_callback(self, checked=None):
        self.taskgraph.wizard_change_candidate_color()
        self.update_action_enabled()


    # Views callbacks.
    # ----------------
    def waveform_spikes_highlighted_callback(self, spikes):
        self.spikes_highlighted = spikes
        [view.highlight_spikes(get_array(spikes)) for view in self.get_views('FeatureView')]

    def features_spikes_highlighted_callback(self, spikes):
        self.spikes_highlighted = spikes
        [view.highlight_spikes(get_array(spikes)) for view in self.get_views('WaveformView')]

    def features_spikes_selected_callback(self, spikes):
        self.spikes_selected = spikes
        self.update_action_enabled()
        [view.highlight_spikes(get_array(spikes)) for view in self.get_views('WaveformView')]

    def waveform_box_clicked_callback(self, coord, cluster, channel):
        """Changed in waveform ==> change in feature"""
        [view.set_projection(coord, channel, -1) for view in self.get_views('FeatureView')]


    # Help callbacks.
    # ---------------
    def manual_callback(self, checked=None):
        url = "https://github.com/klusta-team/klustaviewa/tree/master/docs/manual.md"
        webbrowser.open(url)

    def about_callback(self, checked=None):
        QtGui.QMessageBox.about(self, "KlustaViewa", ABOUT)

    def shortcuts_callback(self, checked=None):
        e = QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                             QtCore.Qt.Key_H,
                             QtCore.Qt.NoModifier,)
        self.keyPressEvent(e)
        self.keyReleaseEvent(e)

    def open_preferences_callback(self, checked=None):
        url = USERPREF.filepath
        log.debug("Opening preferences file at '{0:s}'".format(url))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl('file:///' + url))

    def refresh_preferences_callback(self, checked=None):
        log.debug("Refreshing user preferences.")
        USERPREF.refresh()


    # Geometry.
    # ---------
    def save_geometry(self):
        """Save the arrangement of the whole window."""
        SETTINGS['main_window.views'] = {name: len(self.get_views(name))
            for name in self.views.keys()}
        SETTINGS['main_window.geometry'] = encode_bytearray(
            self.saveGeometry())
        SETTINGS['main_window.state'] = encode_bytearray(self.saveState())

    def restore_geometry(self):
        """Restore the arrangement of the whole window."""
        g = SETTINGS['main_window.geometry']
        s = SETTINGS['main_window.state']
        if s:
            self.restoreState(decode_bytearray(s))
        if g:
            self.restoreGeometry(decode_bytearray(g))


    # Event handlers.
    # ---------------
    def force_key_release(self):
        """HACK: force release of Ctrl, Shift and Alt when focus out."""
        self.keyReleaseEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyRelease,
            QtCore.Qt.Key_Control, QtCore.Qt.NoModifier))
        self.keyReleaseEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyRelease,
            QtCore.Qt.Key_Shift, QtCore.Qt.NoModifier))
        self.keyReleaseEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyRelease,
            QtCore.Qt.Key_Alt, QtCore.Qt.NoModifier))

    def event(self, e):
        if e.type() == QtCore.QEvent.WindowActivate:
            pass
        elif e.type() == QtCore.QEvent.WindowDeactivate:
            self.force_key_release()
        return super(MainWindow, self).event(e)

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
