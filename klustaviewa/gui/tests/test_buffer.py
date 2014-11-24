"""Unit tests for the buffer module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import time
import threading

from qtools import get_application, QtCore, QtGui

from kwiklib.utils import logger as log
from klustaviewa.gui.buffer import Buffer


# -----------------------------------------------------------------------------
# Test class
# -----------------------------------------------------------------------------
class BufferTest(QtCore.QObject):
    delay_timer = .025
    delay_buffer = .1
    
    def __init__(self, parent=None):
        super(BufferTest, self).__init__(parent)
        self.buffer = Buffer(delay_timer=self.delay_timer,
            delay_buffer=self.delay_buffer)
        self.buffer.accepted.connect(self.accepted)
        self.accepted_list = []
    
    def main(self):
        self.buffer.start()
        # Start _main in an external thread.
        self.thread = threading.Thread(target=self._main)
        self.thread.isDaemon = True
        self.thread.start()
        
    def _main(self):
        # Request first, delay after.
        delays = []
        d = self.delay_buffer
        delays.extend([d * 2] * 2)
        delays.extend([d / 2] * 10) # No accept here, except for the last one.
        delays.extend([d * 2] * 2)
        
        for i, delay in enumerate(delays):
            # log.debug((i, delay))
            self.buffer.request(i)
            time.sleep(delay)
        
        time.sleep(d * 2)
        
        # Stop the test.
        app, app_created = get_application()
        app.quit()
        
    def accepted(self, item):
        self.accepted_list.append(item)
        
        
# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_buffer():
    app, app_created = get_application()
    
    test = BufferTest()
    
    # Launch the application.
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.setInterval(100)
    timer.start()
    timer.timeout.connect(test.main)
    app.exec_()
    
    # print test.accepted_list
    assert test.accepted_list[0] == 0
    assert test.accepted_list[-1] == 13
    
    
    