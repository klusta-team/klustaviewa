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
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = None
    window = show_window(MainWindow, filename=filename)
    return window

if __name__ == '__main__':
    main()
    