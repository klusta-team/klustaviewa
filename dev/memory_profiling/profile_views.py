from klustaviewa.dataio import HDF5Loader
from klustaviewa.views import WaveformView
from klustaviewa.views.tests import show_view


filename = r"D:\\Spike sorting\\second\\testalignment_secondnearest_subset_129989"

loader = HDF5Loader(filename=filename)

show_view(WaveformView)


loader.close()