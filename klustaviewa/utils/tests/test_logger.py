"""Unit tests for logger module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
import os
import logging

from kwiklib.utils.logger import (StringLogger, ConsoleLogger,
    StringStream, FileLogger, register)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_string_logger():
    l = StringLogger(fmt='')
    l.info("test 1")
    l.info("test 2")
    
    log = str(l)
    logs = log.split('\n')

    assert "test 1" in logs[0]
    assert "test 2" in logs[1]
    
def test_console_logger():
    l = ConsoleLogger(fmt='')
    l.info("test 1")
    l.info("test 2")
    
def test_file_logger():
    logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'log.txt')
    l = FileLogger(logfile, fmt='', level=logging.DEBUG, print_caller=False)
    l.debug("test file 1")
    l.debug("test file 2")
    l.close()
    
    with open(logfile, 'r') as f:
        contents = f.read()

    assert contents.strip().endswith("test file 1\ntest file 2")
    
    os.remove(logfile)
    
    