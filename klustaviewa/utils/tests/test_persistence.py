"""Unit tests for persistence module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

from qtools import QtCore

from kwiklib.utils.persistence import encode_bytearray, decode_bytearray


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_bytearray():
    str = "rueszhfghgfhfdfrtertrozerporkmlk"
    array = QtCore.QByteArray(str)
    
    encoded = encode_bytearray(array)
    decoded = decode_bytearray(encoded)
    
    assert str == decoded
    