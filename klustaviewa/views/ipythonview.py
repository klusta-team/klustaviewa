import os

import klustaviewa.utils.logger as log

try:
    from IPython.frontend.qt.console.rich_ipython_widget import RichIPythonWidget
    from IPython.frontend.qt.inprocess import QtInProcessKernelManager
    from IPython.lib import guisupport
    IPYTHON = True
except Exception as e:
    IPYTHON = False
    log.debug(("You need IPython 1.0dev if you want the IPython console in the"
    "application: " + e.message))
    
    
from galry import QtGui, QtCore

class IPythonView(QtGui.QWidget):
    def __init__(self, parent=None, getfocus=None):
        super(IPythonView, self).__init__(parent)
        
        # Create an in-process kernel
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        self.control = RichIPythonWidget()
        self.control.set_default_style(colors='linux')
        self.control.kernel_manager = self.kernel_manager
        self.control.kernel_client = self.kernel_client
        self.control.exit_requested.connect(self.stop)
        
        # self.control
        
        box = QtGui.QVBoxLayout()
        box.addWidget(self.control)
        self.setLayout(box)

    def set_data(self, **kwargs):
        self.kernel.shell.push(kwargs)
        
    def stop(self, *args):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()

    