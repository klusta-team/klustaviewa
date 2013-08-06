from klustaviewa.dataio import HDF5Loader
from klustaviewa.utils.userpref import USERPREF
from klustaviewa.views import FeatureView
from klustaviewa.gui import get_featureview_data
from klustaviewa.views.tests import show_view


filename = r"D:\\Spike sorting\\second\\testalignment_secondnearest_subset_129989"
loader = HDF5Loader(filename=filename)

USERPREF['test_operator_delay'] = .1

loader.select(clusters=range(4, 50, 2))
data0 = get_featureview_data(loader)

loader.select(clusters=range(5, 50, 2))
data1 = get_featureview_data(loader)

operators = [
    lambda self: self.view.set_data(**data0),
    lambda self: self.view.set_data(**data1),
] * 20

show_view(FeatureView, operators=operators)


loader.close()

