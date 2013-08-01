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
    
    # initialization
    def set_data(self, rawdata=None, freq=None, channel_height=None, channel_names=None, dead_channels=None):

        # default settings
        self.max_size = 1000
        self.duration_initial = 10
        self.default_channel_height = 0.25
        self.channel_height_limits = (0.01, 2.)
        self.nticks = 10
        
        # load initial variables
        self.rawdata = rawdata
        self.freq = freq
        self.totalduration= (self.rawdata.shape[0] - 1) / self.freq
        self.totalsamples, self.nchannels = self.rawdata.shape
        
        if channel_height is None:
            channel_height = self.default_channel_height
        else:
            self.default_channel_height = channel_height
        self.channel_height = channel_height
        
        if channel_names is None:
            channel_names = ['ch{0:d}'.format(i) for i in xrange(self.nchannels)]
        self.channel_names = channel_names
        
        # these variables will be overwritten after initialization (used to check if init is complete)
        self.slice_ref = (0, 0)
        self.paintinitialized = False
        
        # write initial data to memory of the right length - this will be overwritten by the data updater, but this serves to give Galry a window size/ratio
        self.shape = (self.nchannels, self.duration_initial*self.freq)
        self.samples = self.rawdata[slice(0, (self.duration_initial*self.freq), 1), :]
        # Convert the data into floating points.
        self.samples = np.array(self.samples, dtype=np.float32)
        
        self.total_size = self.rawdata.shape[0]
        # Normalize the data.
        self.samples *= (1. / 65535)
        self.position = self.samples
        nsamples, nchannels = self.position.shape
        
        M = np.empty((nsamples * nchannels, 2))
        self.samples = self.samples.T
        M[:, 1] = self.samples.ravel()
        # Generate the x coordinates.
        x = np.arange(0, self.duration_initial*self.freq, 1) / float(self.total_size - 1)
        
        x = x * 2 * self.totalduration/ self.duration_initial - 1
        M[:, 0] = np.tile(x, nchannels)

        self.bounds = np.arange(nchannels + 1) * nsamples
        self.size = self.bounds[-1]
        
        self.position = self.samples = M
        self.color_array_index = np.repeat(np.arange(nchannels), nsamples / nchannels)
        
        self.interaction_manager.get_processor('viewport').update_viewbox()
        self.interaction_manager.activate_grid()
        
    def load_correct_slices(self):
        
        # dirty hack to make sure that we don't redraw the window until it's been drawn once, otherwise Galry automatically rescales
        if not self.paintinitialized:
             return
        
        dur = self.xlim[1] - self.xlim[0]
        index = int(np.floor(self.xlim[0] / dur))
        zoom_index = int(np.round(self.duration_initial / dur))
        i = (index, zoom_index)
        
        if i != self.slice_ref: # we need to load a new slice
            # Find needed slice(s) of data
            
            xlim_ext = self.get_buffered_viewlimits(self.xlim)
            slice = self.get_viewslice(xlim_ext)
            self.slice_ref = i
            
            self.samples, self.bounds, self.size = self.get_undersampled_data(xlim_ext, slice)
            self.color_array_index = np.repeat(np.arange(self.nchannels), self.samples.shape[0] / self.nchannels)
            
            self.position = self.samples
            
            self.paint_manager.update()
            
    def get_buffered_viewlimits(self, xlim):
        d = self.xlim[1] - self.xlim[0]
        x0_ext = np.clip(self.xlim[0] - 3 * d, 0, self.totalduration)
        x1_ext = np.clip(self.xlim[1] + 3 * d, 0, self.totalduration)
        return (x0_ext, x1_ext)
    
    def get_viewslice(self, xlim):
        d = self.xlim[1] - self.xlim[0]
        zoom = max(self.totalduration/ d, 1)
        view_size = self.totalsamples / zoom
        step = int(np.ceil(view_size / self.max_size))
        
        i0 = np.clip(int(np.round(xlim[0] * self.freq)), 0, self.totalsamples)
        i1 = np.clip(int(np.round(xlim[1] * self.freq)), 0, self.totalsamples)
        return slice(i0, i1, step)
            
    def get_undersampled_data(self, xlim, slice):
        """
        Arguments:
    
          * data: a HDF5 dataset of size Nsamples x Nchannels.
          * xlim: (x0, x1) of the desired data view.
    
        """
        total_size = self.rawdata.shape[0]
        
        samples = self.rawdata[slice, :]
        
        # Convert the data into floating points.
        samples = np.array(samples, dtype=np.float32)
        
        # Normalize the data.
        samples *= (1. / 65535)
        
        # Size of the slice.
        nsamples, nchannels = samples.shape
        # Create the data array for the plot visual.
        M = np.empty((nsamples * nchannels, 2))
        samples = samples.T# + np.linspace(-1., 1., nchannels).reshape((-1, 1))
        M[:, 1] = samples.ravel()
        # Generate the x coordinates.
        x = np.arange(slice.start, slice.stop, slice.step) / float(total_size - 1)
        
        x = x * 2 * self.totalduration/ self.duration_initial - 1
        M[:, 0] = np.tile(x, nchannels)

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
        self.data_manager.paintinitialized = True

    def update(self):
        self.set_data(visual='rawdata_waveforms',
            channel_height=self.data_manager.channel_height,
            position0=self.data_manager.position,
            shape=self.data_manager.shape,
            size=self.data_manager.size,
            index=self.data_manager.color_array_index,
            bounds=self.data_manager.bounds)

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

class GridEventProcessor(EventProcessor):
    def update_axes(self, parameter):
        
        viewbox = self.interaction_manager.get_processor('viewport').viewbox
        
        text, coordinates, n = self.get_ticks_text(*viewbox)
        
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
            
    def nicenum(self, x, round=False):
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

    def get_ticks(self, x0, x1):
        x0 = self.interaction_manager.get_processor('viewport').normalizer.unnormalize_x(x0)
        x1 = self.interaction_manager.get_processor('viewport').normalizer.unnormalize_x(x1)
        r = self.nicenum(x1 - x0 - 1e-6, False)
        d = self.nicenum(r / (self.parent.data_manager.nticks - 1), True)
        g0 = np.floor(x0 / d) * d
        g1 = np.ceil(x1 / d) * d
        nfrac = int(max(-np.floor(np.log10(d)), 0))
        return np.arange(g0, g1 + .5 * d, d), nfrac

    def format_number(self, x, nfrac=None):
        if nfrac is None:
            nfrac = 2

        if np.abs(x) < 1e-15:
            return "0"

        elif np.abs(x) > 1000.001:
            return "%.3e" % x

        # regular decimal notation (scientific notation for < 0.001s is not going to be used frequently if at all)
        return ("%." + str(nfrac) + "f") % x

    def get_ticks_text(self, x0, y0, x1, y1):
        
        ticksx, nfracx = self.get_ticks(x0, x1)
        ticksy = np.linspace(-0.9, 0.9, self.parent.data_manager.nchannels)
        
        n = len(ticksx)
        text = [self.format_number(x, nfracx) for x in ticksx]
        text += [str(self.parent.data_manager.channel_names[y]) for y in reversed(range(self.parent.data_manager.nchannels))]    
        
        # position of the ticks
        coordinates = np.zeros((len(text), 2))
        coordinates[:n, 0] = ticksx
        coordinates[n:, 1] = ticksy
        return text, coordinates, n
        
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
        self.normalizer.normalize((0, -1, self.parent.data_manager.duration_initial, 1))
            
        nav = self.get_processor('navigation')

        self.viewbox = nav.get_viewbox()
        
        nav.constrain_navigation = True
        nav.xmin = -1
        nav.xmax = 2 * self.parent.data_manager.totalduration / self.parent.data_manager.duration_initial
        nav.sxmin = 1.
        
        self.parent.data_manager.xlim = ((self.viewbox[0] + 1) / 2. * (self.parent.data_manager.duration_initial),\
        (self.viewbox[2] + 1) / 2. * (self.parent.data_manager.duration_initial))

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
        self.register('ChangeChannelHeight', self.change_channel_height)
        self.register('Reset', self.reset_channel_height)
    
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
        # get limits
        ll, ul = self.data_manager.channel_height_limits
        
        # increase/decrease channel height between limits
        if ll <= self.data_manager.channel_height <= ul:
            self.data_manager.channel_height *= (1 + parameter)
            self.paint_manager.set_data(channel_height=self.data_manager.channel_height)
            
        # restore limits to ensure it never exceeds them
        if self.data_manager.channel_height > ul:
            self.data_manager.channel_height = ul
        elif self.data_manager.channel_height < ll:
            self.data_manager.channel_height = ll
            
        self.paint_manager.update()
        
    def reset_channel_height(self, parameter):
        self.data_manager.channel_height = self.data_manager.default_channel_height
        self.paint_manager.update()
    
class RawDataBindings(KlustaViewaBindings):      
    def initialize(self):
        self.set('Wheel', 'ChangeChannelHeight', key_modifier='Control',
                   param_getter=lambda p: p['wheel'] * .001)
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
      
        