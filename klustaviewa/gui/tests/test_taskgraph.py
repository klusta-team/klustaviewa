"""Unit tests for the taskgraph module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import time
import threading

import numpy as np
from qtools import QtCore, QtGui, get_application

from klustaviewa.gui.taskgraph import AbstractTaskGraph


# -----------------------------------------------------------------------------
# Test class
# -----------------------------------------------------------------------------
class TestTaskGraph1(AbstractTaskGraph):
    def _fun1(self, x):
        return ('_fun2', (x * x,))
        
    def _fun2(self, xx):
        return xx - 1
        
class TestTaskGraph2(AbstractTaskGraph):
    def __init__(self):
        super(TestTaskGraph2, self).__init__()
        self.checkpoint = False
    
    def _fun1(self, x):
        return [('_fun11', (x * x,)),
                ('_fun12', (x * x * x,)),]
        
    def _fun11(self, xx):
        self.checkpoint = True
        return xx - 1
        
    def _fun12(self, xxx):
        return ('_fun2', (xxx + 2,))
        
    def _fun2(self, xxx2):
        return int(np.round((xxx2 - 2) ** (1. / 3)))
    
        
# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_taskgraph1():
    task = TestTaskGraph1()
    assert task.fun1(3)[0] == 8
    
def test_taskgraph2():
    task = TestTaskGraph2()
    assert task.fun1(3)[0] == 3
    assert task.checkpoint
    
    