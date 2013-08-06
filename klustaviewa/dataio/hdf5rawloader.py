from klustersloader import find_filenames
from hdf5tools import klusters_to_hdf5
from qtools import QtCore

class HDF5RawDataLoader(QtCore.QObject):
    def open(self, filename):
        # filename is blah.xxxx
        # TODO: find the blah.klxd
        filenames = find_filename(filename, 'klxd')
        filename_main = filenames['hdf5_main']
        self.filename_log = filenames['kvwlg']
        self.filename = filename_main
        self.read()
        
        pass

    def get_rawdata(self):
        # return slicable pytables object (EArray)
        pass
    
