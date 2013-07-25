"""Raw Data View: show raw data traces."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import numpy.random as rdn
from numpy.lib.stride_tricks import as_strided
from collections import Counter
import operator
import time

from galry import (Manager, PlotPaintManager, PlotInteractionManager, Visual,
    GalryWidget, QtGui, QtCore, show_window, enforce_dtype, RectanglesVisual,
    TextVisual, process_coordinates, get_next_color, get_color)
from klustaviewa.views.common import KlustaViewaBindings, KlustaView
import klustaviewa.utils.logger as log
from klustaviewa.utils.settings import SETTINGS

__all__ = ['RawDataView']

# -----------------------------------------------------------------------------
# Data manager
# -----------------------------------------------------------------------------

class RawDataManager(Manager):
    info = {}
    def set_data(self, rawdata=None):
        
        samples = rawdata[:5000, :]
        position, shape = process_coordinates(samples.T)
        
        self.rawdata = rawdata
        self.samples = samples
        self.position = position
        self.shape = shape
        print shape
        
#    def update(self, data, xlimex, slice):
#        samples, bounds, size = get_undersampled_data(data, xlimex, slice)
#        nsamples = samples.shape[0]
#        color_array_index = np.repeat(np.arange(nchannels), nsamples / nchannels)
#        self.info = dict(position0=samples, bounds=bounds, size=size,
#            index=color_array_index)
            
# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class RawDataPaintManager(PlotPaintManager):
    def initialize(self):
        self.add_visual(MultiChannelVisual,
            position=self.data_manager.position,
            name='rawdata_waveforms',
            shape=self.data_manager.shape)

    def update(self):
        self.set_data(visual='rawdata_waveforms',
            position=self.data_manager.position)


# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------

class MultiChannelVisual(Visual):
    def initialize(self, color=None, point_size=1.0,
            position=None, shape=None, nprimitives=None, index=None,
            color_array_index=None, channel_height=0.25,
            options=None, autocolor=1):
        
        # register the size of the data
        self.size = np.prod(shape)
        
        # there is one plot per row
        if not nprimitives:
            nprimitives = shape[0]
            nsamples = shape[1]
        else:
            nsamples = self.size // nprimitives
        
        # register the bounds
        if nsamples <= 1:
            self.bounds = [0, self.size]
        else:
            self.bounds = np.arange(0, self.size + 1, nsamples)
        
        # automatic color with color map
        if autocolor is not None:
            if nprimitives <= 1:
                color = get_next_color(autocolor)
            else:
                color = np.array([get_next_color(i + autocolor) for i in xrange(nprimitives)])
            
        # set position attribute
        self.add_attribute("position0", ndim=2, data=position, autonormalizable=True)
        
        index = np.array(index)
        self.add_index("index", data=index)
    
        if color_array_index is None:
            color_array_index = np.repeat(np.arange(nprimitives), nsamples)
        color_array_index = np.array(color_array_index)
            
        ncolors = color.shape[0]
        ncomponents = color.shape[1]
        color = color.reshape((1, ncolors, ncomponents))
        
        dx = 1. / ncolors
        offset = dx / 2.
        
        self.add_texture('colormap', ncomponents=ncomponents, ndim=1, data=color)
        self.add_attribute('index', ndim=1, vartype='int', data=color_array_index)
        self.add_varying('vindex', vartype='int', ndim=1)
        self.add_uniform('nchannels', vartype='float', ndim=1, data=float(nprimitives))
        self.add_uniform('channel_height', vartype='float', ndim=1, data=channel_height)
        
        self.add_vertex_main("""
        vec2 position = position0;
        position.y = channel_height * position.y + .9 * (2 * index - (nchannels - 1)) / (nchannels - 1);
        vindex = index;
        """)
        
        self.add_fragment_main("""
        float coord = %.5f + vindex * %.5f;
        vec4 color = texture1D(colormap, coord);
        out_color = color;
        """ % (offset, dx))
        
        # add point size uniform (when it's not specified, there might be some
        # bugs where its value is obtained from other datasets...)
        self.add_uniform("point_size", data=point_size)
        self.add_vertex_main("""gl_PointSize = point_size;""")
        

# -----------------------------------------------------------------------------
# Interactivity
# -----------------------------------------------------------------------------
class RawDataInteractionManager(PlotInteractionManager):
    pass
    
class RawDataBindings(KlustaViewaBindings):
    pass

# -----------------------------------------------------------------------------
# Top-level widget
# -----------------------------------------------------------------------------
class RawDataView(KlustaView):
    
    # Initialization
    # --------------
    def initialize(self):
        
        self.set_bindings(RawDataBindings)
        self.set_companion_classes(
            paint_manager=RawDataPaintManager,
            interaction_manager=RawDataInteractionManager,
            data_manager=RawDataManager)
    
    def set_data(self, *args, **kwargs):
        self.data_manager.set_data(*args, **kwargs)

        # update?
        if self.initialized:
            self.paint_manager.update()
            self.updateGL()
      
        