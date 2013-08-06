from klustaviewa.dataio import HDF5Loader
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.views import WaveformView
from klustaviewa.gui import get_waveformview_data
from klustaviewa.views.tests import show_view


filename = r"D:\SpikeSorting\nick\137_34_shankA_27cat.res.1"
loader = HDF5Loader(filename=filename)

USERPREF['test_operator_delay'] = .25

# loader.select(clusters=range(5, 20, 2))
# data1 = get_waveformview_data(loader)

def get_data(clusters):
    loader.select(clusters=clusters)
    return get_waveformview_data(loader)

def get_data_fun(clusters):
    # return lambda clusters: get_data(clusters)
    return lambda self: self.view.set_data(**get_data(clusters))

operators = [
    # (lambda self: self.view.set_data(**get_data_fun(clusters)))
    get_data_fun(clusters)
    for clusters in range(4, 20)
]



show_view(WaveformView, operators=operators)


loader.close()

