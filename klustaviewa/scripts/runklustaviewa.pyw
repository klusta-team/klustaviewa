#!/usr/bin/env pythonw

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
import re

import numpy as np

from qtools import show_window
from klustaviewa.gui.mainwindow import MainWindow


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    shank = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            shank = int(sys.argv[2])
    else:
        filename = None
    window = show_window(MainWindow, filename=filename, shank=shank)
    return window

if __name__ == '__main__':
    main()
