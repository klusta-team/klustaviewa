"""Tasks graph in the GUI."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
from qtools import inthread, inprocess
from qtools import QtGui, QtCore


# -----------------------------------------------------------------------------
# Tasks manager
# -----------------------------------------------------------------------------
class TaskManager(QtCore.QObject):
    def __init__(self):
        pass
        
    def run_single(self, action):
        """Take an action in input, execute it, and return the next action(s).
        """
        method, args, kwargs = action
        return getattr(self, method)(*args, **kwargs)
    
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
        
