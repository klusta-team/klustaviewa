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

from galry import (Manager, PlotPaintManager, EventProcessor, PlotInteractionManager, Visual,
    GalryWidget, QtGui, QtCore, show_window, enforce_dtype, NavigationEventProcessor, GridVisual, RectanglesVisual,
    TextVisual, DataNormalizer, process_coordinates, get_next_color, get_color)
from klustaviewa.views.common import KlustaViewaBindings, KlustaView
import klustaviewa.utils.logger as log
from klustaviewa.utils.settings import SETTINGS

__all__ = ['RawDataView']

# -----------------------------------------------------------------------------
# Data manager
# -----------------------------------------------------------------------------

class RawDataManager(Manager):
    info = {}
    
    # Initialization
    def set_data(self, rawdata=None, freq=None, channel_height=None):
        
        self.slice_ref = (0, 0)
        self.max_size = 500
        self.duration_initial = 5
        
        if channel_height is None:
            channel_height = 0.25
        self.channel_height = channel_height
        
        self.rawdata = rawdata
        self.freq = freq
        self.duration = (self.rawdata.shape[0] - 1) / self.freq
        self.nsamples, self.nchannels = self.rawdata.shape
        
        x = np.tile(np.linspace(0., self.duration, self.nsamples // self.max_size), (self.nchannels, 1))
        y = np.zeros_like(x)+ np.linspace(-.9, .9, self.nchannels).reshape((-1, 1))
        # 
        self.blobby, self.shape = process_coordinates(x=x, y=y)
        
        # load enough data for initial view
        # self.samples = self.rawdata[:(self.duration_initial*self.freq), :]
        # self.get_undersampled_data()
        
        # first, load initial slice(s) (from 0 to duration_initial)
        self.xlim = (0., self.duration_initial)
        # self.xlim = ((0 + 1) / 2. * (self.duration_initial), (0.5 + 1) / 2. * (self.duration_initial))
        print "oh my original xlim is ", self.xlim
        
        self.interaction_manager.get_processor('viewport').update_viewbox()
        self.load_correct_slices()
        self.interaction_manager.activate_grid()
        
    def load_correct_slices(self):
        
        # Find needed slice(s) of data
        
        xlim_ext, slice = self.get_view()
        print "xlim: ", self.xlim, " xlim_ext: ", xlim_ext
        
        dur = self.xlim[1] - self.xlim[0]
        index = int(np.floor(self.xlim[0] / dur))
        zoom_index = int(np.round(self.duration_initial / dur))
        i = (index, zoom_index)
        
        if i != self.slice_ref: # we need to load a new slice
            self.slice_ref = i
            
            print "timiojiojiojiojiojiojiois ", slice, " xlimex of ", xlim_ext
            
            self.samples, self.bounds, self.size = self.get_undersampled_data(xlim_ext, slice)
            self.nsamples = self.samples.shape[0]
            self.color_array_index = np.repeat(np.arange(self.nchannels), self.nsamples / self.nchannels)
            
            self.position = self.samples
            
            self.paint_manager.update()
    
    def get_view(self): 
        """Return the slice of the data.

        Arguments:

          * xlim: (x0, x1) of the window currently displayed.

        """
        # Viewport.
        x0, x1 = self.xlim
        d = x1 - x0
        dmax = self.duration
        zoom = max(dmax / d, 1)
        view_size = self.nsamples / zoom
        step = int(np.ceil(view_size / self.max_size))
        # Extended viewport for data.
        x0ex = np.clip(x0 - 3 * d, 0, dmax)
        x1ex = np.clip(x1 + 3 * d, 0, dmax)
        i0 = np.clip(int(np.round(x0ex * self.freq)), 0, self.nsamples)
        i1 = np.clip(int(np.round(x1ex * self.freq)), 0, self.nsamples)
        return (x0ex, x1ex), slice(i0, i1, step)
            
    def get_undersampled_data(self, xlim, slice):
        """
        Arguments:
    
          * data: a HDF5 dataset of size Nsamples x Nchannels.
          * xlim: (x0, x1) of the desired data view.
    
        """
        total_size = self.rawdata.shape[0]
        # Get the view slice.
        # x0ex, x1ex = xlim
        # x0d, x1d = x0ex / (duration_initial) * 2 - 1, x1ex / (duration_initial) * 2 - 1
        # Extract the samples from the data (HDD access).
        samples = self.rawdata[slice, :]
        # Convert the data into floating points.
        samples = np.array(samples, dtype=np.float32)
        # Normalize the data.
        samples *= (1. / 65535)
        # samples *= .25
        # Size of the slice.
        nsamples, nchannels = samples.shape
        # Create the data array for the plot visual.
        M = np.empty((nsamples * nchannels, 2))
        samples = samples.T# + np.linspace(-1., 1., nchannels).reshape((-1, 1))
        M[:, 1] = samples.ravel()
        # Generate the x coordinates.
        x = np.arange(slice.start, slice.stop, slice.step) / float(total_size - 1)
        # [0, 1] -> [-1, 2*duration.duration_initial - 1]
        x = x * 2 * self.duration / self.duration_initial - 1
        M[:, 0] = np.tile(x, nchannels)
        # Update the bounds.
        self.bounds = np.arange(nchannels + 1) * nsamples
        size = self.bounds[-1]
        return M, self.bounds, size
            
# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class RawDataPaintManager(PlotPaintManager):
    def initialize(self):
        self.add_visual(MultiChannelVisual,
            position=self.data_manager.position,
            name='rawdata_waveforms',
            shape=self.data_manager.shape,
            channel_height=self.data_manager.channel_height)
        
        self.add_visual(GridVisual, name='grid')

    def update(self):
        self.set_data(visual='rawdata_waveforms',
            channel_height=self.data_manager.channel_height,
            position0=self.data_manager.position,
            shape=self.data_manager.shape,
            size=self.data_manager.size,
            index=self.data_manager.color_array_index,
            bounds=self.data_manager.bounds)
        # print "shape is ", self.data_manager.shape
            

class MultiChannelVisual(Visual):
    def initialize(self, color=None, point_size=1.0,
            position=None, shape=None, nprimitives=None, index=None,
            color_array_index=None, channel_height=None,
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
        
        print "plotting with shape ", shape, "; color shape ", color.shape
        
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
# Grid
# -----------------------------------------------------------------------------
def nicenum(x, round=False):
    e = np.floor(np.log10(x))
    f = x / 10 ** e
    eps = 1e-6
    if round:
        if f < 1.5:
            nf = 1.
        elif f < 3:
            nf = 2.
        elif f < 7.:
            nf = 5.
        else:
            nf = 10.
    else:
        if f < 1 - eps:
            nf = 1.
        elif f < 2 - eps:
            nf = 2.
        elif f < 5 - eps:
            nf = 5.
        else:
            nf = 10.
    return nf * 10 ** e

def get_ticks(x0, x1):
    nticks = 5
    r = nicenum(x1 - x0, False)
    d = nicenum(r / (nticks - 1), True)
    g0 = np.floor(x0 / d) * d
    g1 = np.ceil(x1 / d) * d
    nfrac = int(max(-np.floor(np.log10(d)), 0))
    return np.arange(g0, g1 + .5 * d, d), nfrac

def format_number(x, nfrac=None):
    if nfrac is None:
        nfrac = 2

    if np.abs(x) < 1e-15:
        return "0"

    elif np.abs(x) > 100.001:
        return "%.3e" % x

    if nfrac <= 2:
        return "%.2f" % x
    else:
        nfrac = nfrac + int(np.log10(np.abs(x)))
        return ("%." + str(nfrac) + "e") % x

def get_ticks_text(x0, y0, x1, y1):
    ticksx, nfracx = get_ticks(x0, x1)
    ticksy, nfracy = get_ticks(y0, y1)
    n = len(ticksx)
    text = [format_number(x, nfracx) for x in ticksx]
    text += [format_number(x, nfracy) for x in ticksy]
    # position of the ticks
    coordinates = np.zeros((len(text), 2))
    coordinates[:n, 0] = ticksx
    coordinates[n:, 1] = ticksy
    return text, coordinates, n

class GridEventProcessor(EventProcessor):
    def update_axes(self, parameter):
        
        viewbox = self.interaction_manager.get_processor('viewport').viewbox
        
        text, coordinates, n = get_ticks_text(*viewbox)

        coordinates[:,0] = self.interaction_manager.get_processor('viewport').normalizer.normalize_x(coordinates[:,0])
        coordinates[:,1] = self.interaction_manager.get_processor('viewport').normalizer.normalize_y(coordinates[:,1])

        # here: coordinates contains positions centered on the static
        # xy=0 axes of the screen
        position = np.repeat(coordinates, 2, axis=0)
        position[:2 * n:2,1] = -1
        position[1:2 * n:2,1] = 1
        position[2 * n::2,0] = -1
        position[2 * n + 1::2,0] = 1

        axis = np.zeros(len(position))
        axis[2 * n:] = 1

        self.set_data(visual='grid_lines', position=position, axis=axis)

        coordinates[n:, 0] = -.95
        coordinates[:n, 1] = -.95

        t = "".join(text)
        n1 = len("".join(text[:n]))
        n2 = len("".join(text[n:]))

        axis = np.zeros(n1+n2)
        axis[n1:] = 1

        self.set_data(visual='grid_text', text=text,
            coordinates=coordinates,
            axis=axis)
            
class ViewportUpdateProcessor(EventProcessor):
    def initialize(self):
        self.register('Initialize', self.update_viewport)
        self.register('Pan', self.update_viewport)
        self.register('Zoom', self.update_viewport)
        self.register('Reset', self.update_viewport)
        self.register('Animate', self.update_viewport)
        self.register(None, self.update_viewport)
    
    def update_viewbox(self):
        # normalization viewbox
        self.normalizer = DataNormalizer()
        self.normalizer.normalize(
            (0, -1, self.parent.data_manager.duration, 1))
            
        nav = self.get_processor('navigation')
        if not nav:
            return

        self.viewbox = nav.get_viewbox()
        nav.constrain_navigation = True
        nav.xmin = -1
        nav.xmax = 2 * self.parent.data_manager.duration / self.parent.data_manager.duration_initial
        nav.sxmin = 1.
        
        x0, y0, x1, y1 = self.viewbox
        
        #nav.set_viewbox(self.normalizer.normalize_x(0.0), -1.0, self.normalizer.normalize_x(5.0), 1.0)
        x0 = self.normalizer.unnormalize_x(x0)
        y0 = self.normalizer.unnormalize_y(y0)
        x1 = self.normalizer.unnormalize_x(x1)
        y1 = self.normalizer.unnormalize_y(y1)
        
        self.parent.data_manager.xlim = ((self.viewbox[0] + 1) / 2. * (self.parent.data_manager.duration_initial),\
        (self.viewbox[2] + 1) / 2. * (self.parent.data_manager.duration_initial))
        
        print "viewbox is ", self.viewbox

        # now we know the viewport has been updated, update the grid 
        self.interaction_manager.get_processor('grid').update_axes(None)
        
        # check if we need to load/unload any slices
        self.parent.data_manager.load_correct_slices()
        
    def update_viewport(self, parameter):
        self.update_viewbox()
# -----------------------------------------------------------------------------
# Interactivity
# -----------------------------------------------------------------------------
class RawDataInteractionManager(PlotInteractionManager):
    def initialize(self):
        
        self.channel_height = 0.25
        self.register('ChangeChannelHeight', self.change_channel_height)
    
    def initialize_default(self, constrain_navigation=None,
        momentum=True,
        ):
        
        super(PlotInteractionManager, self).initialize_default()
        self.add_processor(NavigationEventProcessor,
            constrain_navigation=constrain_navigation, 
            momentum=momentum,
            name='navigation')
        
        self.add_processor(ViewportUpdateProcessor, name='viewport')
        self.add_processor(GridEventProcessor, name='grid')
        
    def activate_grid(self):
        self.paint_manager.set_data(visual='grid_lines', 
            visible=True)
        processor = self.get_processor('grid')
        if processor:
            processor.activate(True)
            processor.update_axes(None)
            
    def change_channel_height(self, parameter):
        self.data_manager.channel_height *= (1 + parameter)
        self.paint_manager.set_data(channel_height=self.data_manager.channel_height)
        self.paint_manager.update()
    
class RawDataBindings(KlustaViewaBindings):
    def set_channel_height(self):
        self.set('Wheel', 'ChangeChannelHeight', key_modifier='Control',
                   param_getter=lambda p: p['wheel'] * .001)
                   
    def initialize(self):
       self.set_channel_height()
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
      
        