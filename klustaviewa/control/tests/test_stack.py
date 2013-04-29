"""Unit tests for stack module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from klustaviewa.control.stack import Stack


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_stack1():
    s = Stack()
    
    assert s.get_current() == None
    assert s.can_undo() == False
    assert s.can_redo() == False
    
    s.add("action 0")
    
    assert s.can_undo() == True
    assert s.can_redo() == False
    
    s.add("action 1")
    s.add("action 2")
    
    assert s.get_current() == "action 2"
    assert s.can_undo() == True
    assert s.can_redo() == False
    
    s.undo()
    
    assert s.get_current() == "action 1"
    assert s.can_undo() == True
    assert s.can_redo() == True
    
    s.redo()
    
    assert s.get_current() == "action 2"
    assert s.can_undo() == True
    assert s.can_redo() == False
    
    s.undo()
    s.undo()
    s.add("action 1 bis")
    
    assert s.get_current() == "action 1 bis"
    assert s.can_undo() == True
    assert s.can_redo() == False
    
    s.undo()
    assert s.get_current() == "action 0"
    assert s.can_undo() == True
    assert s.can_redo() == True
    
def test_stack_maxsize():
    
    s = Stack(maxsize=10)
    
    [s.add("action {0:d}".format(i)) for i in xrange(20)]
    
    assert len(s.get_stack()) == 10
    assert s.get_current() == "action 19"
    assert s.can_undo() == True
    assert s.can_redo() == False
    
    [s.undo() for _ in xrange(10)]
    assert s.can_undo() == False
    assert s.can_redo() == True
    
    