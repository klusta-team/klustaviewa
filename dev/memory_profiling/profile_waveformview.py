from klustaviewa.dataio import HDF5Loader
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.views import WaveformView
from klustaviewa.gui import get_waveformview_data
from klustaviewa.views.tests import show_view


filename = r"D:\\Spike sorting\\second\\testalignment_secondnearest_subset_129989"
loader = HDF5Loader(filename=filename)

USERPREF['test_operator_delay'] = .1

loader.select(clusters=range(4, 20, 2))
data0 = get_waveformview_data(loader)

loader.select(clusters=range(5, 20, 2))
data1 = get_waveformview_data(loader)

operators = [
    lambda self: self.view.set_data(**data0),
    lambda self: self.view.set_data(**data1),
] * 10

show_view(WaveformView, operators=operators)


loader.close()

