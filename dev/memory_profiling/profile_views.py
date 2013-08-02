from klustaviewa.dataio import HDF5Loader
from klustaviewa.views import WaveformView
from klustaviewa.gui import get_waveformview_data
from klustaviewa.views.tests import show_waveformview


filename = r"D:\\Spike sorting\\second\\testalignment_secondnearest_subset_129989"
loader = HDF5Loader(filename=filename)


show_waveformview(loader, [5, 6, 7])


loader.close()

