"""Unit tests for logger module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

from klustaviewa.utils.logger import StringLogger, ConsoleLogger, StringStream


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
    
def test_consoler_logger():
    l = ConsoleLogger(fmt='')
    l.info("test 1")
    l.info("test 2")
    