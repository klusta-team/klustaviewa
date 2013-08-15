"""Persistence utility functions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import cPickle
import sys

from qtools import QtCore, QtGui


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def encode_bytearray(bytearray):
    """Encode a QByteArray in a string."""
    return str(bytearray.toBase64())

def decode_bytearray(bytearray_encoded):
    return QtCore.QByteArray.fromBase64(bytearray_encoded)

