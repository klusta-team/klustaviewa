"""Log View: display the log (stdout) of the software."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys

import klustaviewa.utils.logger as log

from qtools import QtGui, QtCore


# -----------------------------------------------------------------------------
# Out log, stream which writes in a TextEdit with a color.
# -----------------------------------------------------------------------------
class OutLog(object):
    def __init__(self, edit, out=None, color=None):
        self.edit = edit
        self.out = None
        self.color = color

    def write(self, m):
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QtGui.QTextCursor.End)
        self.edit.insertPlainText(m)

        if self.color:
            self.edit.setTextColor(tc)

        if self.out:
            self.out.write(m)

            
# -----------------------------------------------------------------------------
# Log view.
# -----------------------------------------------------------------------------
class LogView(QtGui.QWidget):
    def __init__(self, parent=None, getfocus=None):
        super(LogView, self).__init__(parent)
        
        # Create the text edit widget.
        self.textedit = QtGui.QTextEdit()
        
        # Redirect standard output and error to the text edit.
        sys.stdout = OutLog(self.textedit, sys.stdout, QtGui.QColor(50, 50, 50))
        sys.stderr = OutLog(self.textedit, sys.stderr, QtGui.QColor(255, 50, 50))
        
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
        self.append(text)
    
    
    