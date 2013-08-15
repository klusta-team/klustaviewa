import os
import galry
from qtools import QtGui

__all__ = ['get_icon']

def get_icon(name):
    """Get an icon from the icons folder in the current package,
    or directly from galry."""
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "../icons/%s.png" % name)
    if not os.path.exists(path):
        return galry.get_icon(name)
    else:
        return QtGui.QIcon(path)

