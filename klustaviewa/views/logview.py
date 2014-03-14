"""Log View: display the log (stdout) of the software."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys

from qtools import QtGui, QtCore

from kwiklib.utils.logger import Logger


# -----------------------------------------------------------------------------
# Log view.
# -----------------------------------------------------------------------------
class OutLog(QtCore.QObject):
    # We use signals so that this stream object can write from another thread
    # and the QTextEdit widget will be updated in the main thread through
    # a slot connected in the main window.
    writeRequested = QtCore.pyqtSignal(str)
    
    def write(self, m):
        self.writeRequested.emit(m)


class ViewLogger(Logger):
    def __init__(self, **kwargs):
        self.outlog = OutLog()
        kwargs['stream'] = self.outlog
        super(ViewLogger, self).__init__(**kwargs)

        
class LogView(QtGui.QWidget):
    def __init__(self, parent=None, getfocus=None):
        super(LogView, self).__init__(parent)
        
        # Create the text edit widget.
        self.textedit = QtGui.QTextEdit()
        self.textedit.setReadOnly(True)
        
        # Add the text edit widget to the layout.
        box = QtGui.QVBoxLayout()
        box.addWidget(self.textedit)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.setLayout(box)
        
    def append(self, text=''):
        self.textedit.moveCursor(QtGui.QTextCursor.End)
        self.textedit.insertPlainText(text)
        
    def get_text(self):
        return self.textedit.toPlainText()
        
    def set_data(self, text=''):
        self.textedit.setText(text)
    
    
    