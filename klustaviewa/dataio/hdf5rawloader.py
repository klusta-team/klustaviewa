from qtools import QtCore

class HDF5RawDataLoader(QtCore.QObject):
    def open(self, filename):
        # filename is blah.xxxx
        # TODO: find the blah.klxd
        filename = find_filename(filename, 'klxd')
        pass

    def get_raw_data(self):
        # return slicable pytables object (EArray)
        pass
    
