#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
import re

import numpy as np

from galry import show_window
from klustaviewa.gui.mainwindow import MainWindow


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    window = show_window(MainWindow)
    return window

if __name__ == '__main__':
    main()
    