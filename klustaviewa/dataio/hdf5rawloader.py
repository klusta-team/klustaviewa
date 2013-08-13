import os

import numpy as np
import tables as tb

from qtools import QtCore
from klustersloader import find_filenames

class HDF5RawDataLoader(QtCore.QObject):
    def open(self, filename):
        
        filenames = find_filenames(filename)
        self.filename_raw = filenames['hdf5_raw']
        self.filename_kla = filenames['hdf5_kla']

        try:
            self.kld_raw = tb.openFile(self.filename_raw)
        except:
            self.kld_raw = None

    def get_rawdata(self):
        try:
            rawdata = self.kld_raw.root.data
        except:
            rawdata = None
        
        freq = 20000.
        dead_channels = np.arange(0,5,1)
        data = dict(
            rawdata=rawdata,
            freq=freq,
            dead_channels=dead_channels,
        )
        return data
    

