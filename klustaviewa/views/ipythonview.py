"""IPython View: interactive shell in the interface."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from kwiklib.utils import logger as log

try:
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
    from IPython.qt.inprocess import QtInProcessKernelManager
    from IPython.lib import guisupport
    IPYTHON = True
except Exception as e:
    IPYTHON = False
    log.debug(("You need IPython 1.0 if you want the IPython console in the"
    "application: " + e.message))
    
import galry
from qtools import QtGui, QtCore


# -----------------------------------------------------------------------------
# IPython view
# -----------------------------------------------------------------------------
class IPythonView(QtGui.QWidget):
    def __init__(self, parent=None, getfocus=None):
        super(IPythonView, self).__init__(parent)
        
        # Create an in-process kernel
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.shell = self.kernel.shell

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        self.control = RichIPythonWidget()
        self.control.set_default_style(colors='linux')
        self.control.kernel_manager = self.kernel_manager
        self.control.kernel_client = self.kernel_client
        self.control.exit_requested.connect(self.stop)
        
        # Enable Pylab mode.
        self.shell.enable_pylab()
        self.shell.automagic = True
        
        # Add some variables in the namespace.
        self.push(galry=galry)
        
        box = QtGui.QVBoxLayout()
        box.addWidget(self.control)
        
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)

        
        self.setLayout(box)

    def stop(self, *args):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()
    
    
    # Public methods.
    # ---------------
    def set_data(self, **kwargs):
        self.push(**kwargs)
    
    def push(self, **kwargs):
        """Inject variables in the interactive namespace."""
        self.shell.push(kwargs)
    
    def run_file(self, file):
        """Execute a Python file in the interactive namespace."""
        self.shell.safe_execfile(file, self.shell.user_global_ns)
    
    def run_cell(self, *args, **kwargs):
        """Execute a cell."""
        self.shell.run_cell(*args, **kwargs)
        
    
    