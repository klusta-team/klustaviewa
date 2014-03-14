"""Buffer allowing to delay multiple costly operations."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import timeit
from threading import Lock

import numpy as np

from qtools import QtGui, QtCore
from kwiklib.utils import logger as log


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def time():
    return timeit.default_timer()


# -----------------------------------------------------------------------------
# Buffer
# -----------------------------------------------------------------------------
class Buffer(QtCore.QObject):
    accepted = QtCore.pyqtSignal(object)
    
    def __init__(self, parent=None, delay_timer=None, delay_buffer=None):
        """Create a new buffer.
        
        The user can request an item at any time. The buffer will respond
        to the requests only after some delay where no request happened.
        This allows users to handle time-consuming requests smoothly, when the
        latency is not critical.
        
          * delay_timer: time interval during two visits.
          * delay_buffer: minimum time interval between two accepted requests.
        
        """
        super(Buffer, self).__init__(parent)
        self.delay_timer = delay_timer
        self.delay_buffer = delay_buffer
        
    
    # Internal methods.
    # -----------------
    def _accept(self):
        # log.debug("Accept")
        self.accepted.emit(self._buffer.pop())
        self._last_accepted = time()
        self._buffer = []
    
    def _visit(self):
        delay = time() - self._last_request
        n = len(self._buffer)
        # log.debug("Visit {0:d} {1:.5f}".format(n, delay))
        # Only accept items that have been put after a sufficiently long
        # idle time.
        if ((n == 1 and (delay >= self.delay_buffer / 2)) or 
           ((n >= 2) and (delay >= self.delay_buffer))):
            self._accept()
    
    
    # Public methods.
    # ---------------
    def start(self):
        self._buffer = []
        self._last_request = 0
        self._last_accepted = 0
        
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(self.delay_timer * 1000))
        self.timer.timeout.connect(self._visit)
        self.timer.start()
        
    def stop(self):
        self.timer.stop()
        
    def request(self, item):
        self._buffer.append(item)
        n = len(self._buffer)
        self._last_request = time()
    
    
    
    