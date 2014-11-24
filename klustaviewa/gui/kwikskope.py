"""Main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import time
import os
import inspect
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
from kwiklib.dataio import KlustersLoader, KwikLoader
from klustaviewa.gui.buffer import Buffer
from klustaviewa.gui.dock import ViewDockWidget, DockTitleBar
from klustaviewa.stats.correlations import normalize
from kwiklib.utils import logger as log
from kwiklib.utils.logger import FileLogger, register, unregister
from kwiklib.utils.persistence import encode_bytearray, decode_bytearray
from klustaviewa import USERPREF
from klustaviewa import SETTINGS
from klustaviewa import APPNAME, ABOUT, get_global_path
from klustaviewa.gui.threads import ThreadedTasks, OpenTask
import klustaviewa.views.viewdata as vd
import rcicons

    
# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
class KwikSkope(QtGui.QMainWindow):
    
    def __init__(self, parent=None, dolog=True, filename=None):
        super(KwikSkope, self).__init__(parent)

        # HACK: display the icon in Windows' taskbar.
        if os.name == 'nt':
            try:
                import ctypes
                myappid = 'klustateam.kwikskope'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except:
                pass
        
        self.dolog = dolog
        if self.dolog:
            create_file_logger()
        
        log.debug("Using {0:s}.".format(QT_BINDING))
        
        # Main window options.
        self.move(50, 50)
        self.setWindowTitle('KwikSkope')
        
        # Focus options.
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setMouseTracking(True)
        
        # Dock widgets options.
        self.setDockNestingEnabled(True)
        self.setAnimated(False)
        self.setWindowIcon(get_icon('logo'))
        
        # Initialize some variables.
        # self.statscache = None
        # self.loader = KlustersLoader()
        self.loader = KwikLoader()
        self.loader.progressReported.connect(self.open_progress_reported)
        self.loader.saveProgressReported.connect(self.save_progress_reported)
        self.wizard = Wizard()
        
        self.controller = None
        self.spikes_highlighted = []
        self.spikes_selected = []
        self._wizard = False
        self.is_file_open = False
        self.need_save = False
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
            self.open_task.open(self.loader, filename)
        
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
            
        self.add_action('save', '&Save', shortcut='Ctrl+S', icon='save')
        
        self.add_action('close', '&Close file')
        
        # Quit action.
        self.add_action('quit', '&Quit', shortcut='Ctrl+Q')

    def create_edit_actions(self):
        self.add_action('undo', '&Undo', shortcut='Ctrl+Z', icon='undo')
        self.add_action('redo', '&Redo', shortcut='Ctrl+Y', icon='redo')
            
    def create_view_actions(self):
        self.add_action('add_ipython_view', 'Add &IPythonView')
        self.add_action('reset_views', '&Reset views')
        self.add_action('toggle_fullscreen', 'Toggle fullscreen', shortcut='F')
        
        self.add_action('override_color', 'Override channel &color',
            icon='override_color')#, shortcut='C')
        
    def create_help_actions(self):
        self.add_action('about', '&About')
        self.add_action('manual', 'Show &manual')
        self.add_action('shortcuts', 'Show &shortcuts')
        self.add_action('open_preferences', '&Open preferences')
        self.add_action('refresh_preferences', '&Refresh preferences',
            shortcut='CTRL+R')
        
    def create_menu(self):
        # File menu.
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_last_action)
        file_menu.addSeparator()
        # file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        # Edit menu.
        # edit_menu = self.menuBar().addMenu("&Edit")
        # edit_menu.addAction(self.undo_action)
        # edit_menu.addAction(self.redo_action)
        
        # View menu.
        views_menu = self.menuBar().addMenu("&View")
        if vw.IPYTHON:
            views_menu.addAction(self.add_ipython_view_action)
            views_menu.addSeparator()
        # views_menu.addAction(self.override_color_action)
        # views_menu.addSeparator()
        views_menu.addAction(self.reset_views_action)
        views_menu.addAction(self.toggle_fullscreen_action)
        
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
    
    def can_undo(self):
        if self.controller is None:
            return False
        return self.controller.can_undo()
    
    def can_redo(self):
        if self.controller is None:
            return False
        return self.controller.can_redo()
    
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
    
    def add_channel_view(self, do_update=None, floating=False):
        view = self.create_view(vw.ChannelView,
            position=QtCore.Qt.LeftDockWidgetArea,
            index=len(self.views['ChannelView']),
            closable=False, 
            # floatable=False
            )
            
        # Connect callback functions.
        # view.channelsSelected.connect(self.channels_selected_callback)
        # view.channelColorChanged.connect(self.channel_color_changed_callback)
        # view.groupColorChanged.connect(self.group_color_changed_callback)
        # view.groupRenamed.connect(self.group_renamed_callback)
        # view.channelsMoved.connect(self.channels_moved_callback)
        # view.groupAdded.connect(self.group_added_callback)
        # view.groupRemoved.connect(self.group_removed_callback)
        
        self.views['ChannelView'].append(view)
        
        if do_update:
            self.update_channel_view()
        
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
            
    def add_ipython_view(self, floating=None):
        view = self.create_view(vw.IPythonView,
            index=len(self.views['IPythonView']),
            position=QtCore.Qt.BottomDockWidgetArea,
            floating=True)
        # Create namespace for the interactive session.
        namespace = dict(
            window=self,
            select=self.get_view('ChannelView').select,
            loader=self.loader,
            # stats=self.statscache,
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
            
    def add_trace_view(self, do_update=None, floating=False):
         if len(self.views['TraceView']) >= 1:
             return
         view = self.create_view(vw.TraceView,
             index=len(self.views['TraceView']),
             position=QtCore.Qt.RightDockWidgetArea,
             floating=floating)
         self.views['TraceView'].append(view)
         if do_update and self.is_file_open:
             self.update_trace_view()
            
    def update_trace_view(self):
        data = vd.get_traceview_data(self.loader)
        [view.set_data(**data) for view in self.get_views('TraceView')]

    def update_channel_view(self, channels=None):
        """Update the channel view using the data stored in the loader
        object."""
        data = vd.get_channelview_data(self.loader, channels=channels)
        self.get_view('ChannelView').set_data(**data)
        if channels is not None:
            return
            
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
            ChannelView=[],
            IPythonView=[],
            TraceView=[],
            )
        
        # count = SETTINGS['main_window.views']
        count = None
        if count is None:
            self.create_default_views()
        else:
            self.create_custom_views(count)
    
    def create_default_views(self, do_update=None, floating=False):
        self.add_channel_view(do_update=do_update)
        self.add_trace_view(do_update=do_update)
    
    def create_custom_views(self, count):
        [self.add_channel_view() for _ in xrange(count['ChannelView'])]
        [self.add_ipython_view() for _ in xrange(count['IPythonView'])]
        [self.add_trace_view() for _ in xrange(count['TraceView'])]
    
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
            self.open_task.open(self.loader, path)
            # Save the folder.
            folder = os.path.dirname(path)
            SETTINGS['main_window.last_data_dir'] = folder
            SETTINGS['main_window.last_data_file'] = path
            
    def save_callback(self, checked=None):
        self.open_task.save(self.loader)
        
    def open_last_callback(self, checked=None):
        path = SETTINGS['main_window.last_data_file']
        if path:
            self.open_task.open(self.loader, path)
            
    def close_callback(self, checked=None):
        self.is_file_open = False
        channels = self.get_view('ChannelView').selected_channels()
        if channels:
            self.get_view('ChannelView').unselect()
            time.sleep(.25)

        # Update the views.
        self.update_channel_view()
        self.update_trace_view()

        # Clear the ChannelView.
        self.get_view('ChannelView').clear()
            
    def quit_callback(self, checked=None):
        self.close()
    
    
    # Open callbacks.
    # --------------
    def open_done(self):
        self.is_file_open = True
        
        # HACK: force release of Control key.
        self.force_key_release()
        channels = self.get_view('ChannelView').selected_channels()
        if channels:
            self.get_view('ChannelView').unselect()
        
        # Create the Controller.
        self.controller = Controller(self.loader)
        # Create the cache for the channel statistics that need to be
        # computed in the background.
        # self.statscache = StatsCache(self.loader.ncorrbins)
        # Update stats cache in IPython view.
        ipython = self.get_view('IPythonView')
        # if ipython:
            # ipython.set_data(stats=self.statscache)
        
        # Initialize the wizard.
        self.wizard = Wizard()
        
        # Update the views.
        self.update_channel_view()
        self.update_trace_view()
        
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
    # def buffer_accepted_callback(self, (channels, wizard)):
    #     self._wizard = wizard
    #     # The wizard boolean specifies whether the autozoom is activated or not.
    #     self.taskgraph.select(channels, wizard and 
    #         self.automatic_projection_action.isChecked())
        
    def channels_selected_callback(self, channels, wizard=False):
        self.buffer.request((channels, wizard))
    
    def channel_pair_selected_callback(self, channels):
        """Callback when the user clicks on a pair in the
        SimilarityMatrixView."""
        self.get_view('ChannelView').select(channels)
    
    
    # Views menu callbacks.
    # ---------------------
    
    def add_trace_view_callback(self, checked=None):
            self.add_trace_view(do_update=True, floating=True)
    
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
    # def override_color_callback(self, checked=None):
    #     self.override_color = not self.override_color
    #     self.taskgraph.override_color(self.override_color)
    
    
    # Actions callbacks.
    # ------------------
    #     
    # def undo_callback(self, checked=None):
    #     if self.is_busy:
    #         return
    #     self.taskgraph.undo(self._wizard)
    #     self.update_action_enabled()
    #     
    # def redo_callback(self, checked=None):
    #     if self.is_busy:
    #         return
    #     self.taskgraph.redo(self._wizard)
    #     self.update_action_enabled()
    #     
    # def channel_color_changed_callback(self, channel, color):
    #     self.taskgraph.channel_color_changed(channel, color, self._wizard)
    #     self.update_action_enabled()
    #     
    # def group_color_changed_callback(self, group, color):
    #     self.taskgraph.group_color_changed(group, color)
    #     self.update_action_enabled()
    #     
    # def group_renamed_callback(self, group, name):
    #     self.taskgraph.group_renamed(group, name)
    #     self.update_action_enabled()
    #     
    # def channels_moved_callback(self, channels, group):
    #     self.taskgraph.channels_moved(channels, group)
    #     self.update_action_enabled()
    #     
    # def group_removed_callback(self, group):
    #     self.taskgraph.group_removed(group)
    #     self.update_action_enabled()
    #     
    # def group_added_callback(self, group, name, color):
    #     self.taskgraph.group_added(group, name, color)
    #     self.update_action_enabled()
    # 
        
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
        # SETTINGS['main_window.views'] = {name: len(self.get_views(name))
        #     for name in self.views.keys()}
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
        return super(KwikSkope, self).event(e)
    
    def contextMenuEvent(self, e):
        """Disable the context menu in the main window."""
        return
        
    def keyPressEvent(self, e):
        super(KwikSkope, self).keyPressEvent(e)
        for views in self.views.values():
            [view.keyPressEvent(e) for view in views]
        
    def keyReleaseEvent(self, e):
        super(KwikSkope, self).keyReleaseEvent(e)
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
        return super(KwikSkope, self).closeEvent(e)
            
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
    
    