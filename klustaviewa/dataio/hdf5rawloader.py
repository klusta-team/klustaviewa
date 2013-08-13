from klustersloader import find_filenames
import numpy as np
import tables as tb
from qtools import QtCore

class HDF5RawDataLoader(QtCore.QObject):
    def open(self, filename):
        
        filenames = find_filenames(filename)
        self.filename_raw = filenames['hdf5_raw']
        self.filename_kla = filenames['hdf5_kla']

        self.kld_raw = tb.openFile(self.filename_raw)

    def get_rawdata(self):
        try:
            rawdata = self.kld_raw.root.RawData
        except:
            self.rawdata = self.kld_raw.root.raw_data
        
        freq = 20000.
        dead_channels = np.arange(0,5,1)
        data = dict(
            rawdata=self.rawdata,
            freq=freq,
            dead_channels=dead_channels,
        )
        return data
    

