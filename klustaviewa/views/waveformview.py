"""Waveform View: show waveforms on all channels."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import numpy.random as rdn
from numpy.lib.stride_tricks import as_strided
import operator
import time

from qtools import QtGui, QtCore, show_window, get_application
from galry import (Manager, PlotPaintManager, PlotInteractionManager, Visual,
    GalryWidget, enforce_dtype, RectanglesVisual,
    TextVisual)
from kwiklib.dataio.tools import get_array
from kwiklib.dataio.selection import get_spikes_in_clusters, select, get_indices
from klustaviewa.views.common import HighlightManager, KlustaViewaBindings, KlustaView
from kwiklib.utils.colors import COLORMAP_TEXTURE, SHIFTLEN
from kwiklib.utils import logger as log
from klustaviewa import SETTINGS


__all__ = ['WaveformView']


# -----------------------------------------------------------------------------
# Shaders
# -----------------------------------------------------------------------------
VERTEX_SHADER = """
    // get channel position
    vec2 channel_position = channel_positions[int(channel)];
    
    // get the box position
    vec2 box_position = channel_position;
    
    // take probe scaling into account
    box_position *= probe_scale;
    
    // adjust box position in the separated case
    if (!superimposed)
    {
        box_position.x += box_size_margin.x * (0.5 + cluster - 0.5 * nclusters);
    }
    
    // compute the depth: put masked spikes on the background, unmasked ones
    // on the foreground on a different layer for each cluster
    float depth = 0.;
    if (mask == 1.)
        depth = -(cluster_depth + 1) / (nclusters + 10);
    
    // move the vertex to its position0
    vec3 position = vec3(position0 * 0.5 * box_size + box_position, depth);
    
    cmap_vindex = cmap_index;
    vhighlight = highlight;
    vmask = mask;
"""

FRAGMENT_SHADER = """
    float index = %CMAP_OFFSET% + cmap_vindex * %CMAP_STEP%;
    vec2 index2d = vec2(index, %SHIFT_OFFSET% + (1 + toggle_mask * (1 - vmask) * %SHIFTLEN%) * %SHIFT_STEP%);
    if (vhighlight > 0) {
        index2d.y = 0;
        out_color = texture2D(cmap, index2d);
        out_color.w = .5;
    }
    else {
        out_color = texture2D(cmap, index2d);
        out_color.w = .5;
    }
"""

FRAGMENT_SHADER_AVERAGE = """
    float index = %CMAP_OFFSET% + cmap_vindex * %CMAP_STEP%;
    vec2 index2d = vec2(index, %SHIFT_OFFSET% + (1 + toggle_mask * (1 - vmask) * %SHIFTLEN%) * %SHIFT_STEP%);
    out_color = texture2D(cmap, index2d);
    out_color.w = 1.;
"""


# -----------------------------------------------------------------------------
# Data manager
# -----------------------------------------------------------------------------
class WaveformPositionManager(Manager):
    def reset(self):
        # set parameters
        self.alpha = .02
        self.beta = .02
        self.box_size_min = .01
        self.probe_scale_min = .01
        self.probe_scale = (1., 1.)
        # for each spatial arrangement, the box sizes automatically computed,
        # or modified by the user
        self.box_sizes = dict()
        self.box_sizes.__setitem__('Linear', None)
        self.box_sizes.__setitem__('Geometrical', None)
        # self.T = None
        self.spatial_arrangement = 'Geometrical'
        self.superposition = 'Separated'
        
        # channel positions
        self.channel_positions = {}
        
    def normalize_channel_positions(self, spatial_arrangement, channel_positions):
        channel_positions = channel_positions.copy()
        
        # waveform data bounds
        xmin = channel_positions[:,0].min()
        xmax = channel_positions[:,0].max()
        ymin = channel_positions[:,1].min()
        ymax = channel_positions[:,1].max()
        
        # w, h = self.find_box_size(spatial_arrangement=spatial_arrangement)
    
        # w = .5
        h = .1
        
        size = self.load_box_size()
        if size is None:
            w = h = 0.
        else:
            w, _ = size
        
        # HACK: if the normalization depends on the number of clusters,
        # the positions will change whenever the cluster selection changes
        k = self.nclusters
        
        if xmin == xmax:
            ax = 0.
        else:
            # ax = (2 - k * w * (1 + 2 * self.alpha)) / (xmax - xmin)
            ax = (2 - w * (1 + 2 * self.alpha)) / (xmax - xmin)
            
        if ymin == ymax:
            ay = 0.
        else:
            ay = (2 - h * (1 + 2 * self.alpha)) / (ymax - ymin)
        
        # set bx and by to have symmetry
        bx = -.5 * ax * (xmax + xmin)
        by = -.5 * ay * (ymax + ymin)
        
        # transform the boxes positions so that everything fits on the screen
        channel_positions[:,0] = ax * channel_positions[:,0] + bx
        channel_positions[:,1] = ay * channel_positions[:,1] + by
        
        return enforce_dtype(channel_positions, np.float32)
    
    def get_channel_positions(self):
        return self.channel_positions[self.spatial_arrangement]
    
    def set_info(self, nchannels, nclusters, 
                       geometrical_positions=None,
                       spatial_arrangement=None, superposition=None,
                       box_size=None, probe_scale=None):
        """Specify the information needed to position the waveforms in the
        widget.
        
          * nchannels: number of channels
          * nclusters: number of clusters
          * coordinates of the electrodes
          
        """
        self.nchannels = nchannels
        self.nclusters = nclusters
        
        # linear position: indexing from top to bottom
        linear_positions = np.zeros((self.nchannels, 2), dtype=np.float32)
        linear_positions[:,1] = np.linspace(1., -1., self.nchannels)
        
        # check that the probe geometry is coherent with the number of channels
        if geometrical_positions is not None:
            if geometrical_positions.shape[0] != self.nchannels:
                geometrical_positions = None
            
        # default geometrical position
        if geometrical_positions is None:
            geometrical_positions = linear_positions.copy()
        
        # normalize and save channel position
        self.channel_positions['Linear'] = \
            self.normalize_channel_positions('Linear', linear_positions)
        self.channel_positions['Geometrical'] = \
            self.normalize_channel_positions('Geometrical', geometrical_positions)
        
        # set waveform positions
        self.update_arrangement(spatial_arrangement=spatial_arrangement,
                                superposition=superposition,
                                box_size=box_size,
                                probe_scale=probe_scale)
        
    def update_arrangement(self, spatial_arrangement=None, superposition=None,
                                 box_size=None, probe_scale=None):
        """Update the waveform arrangement (self.channel_positions).
        
          * spatial_arrangement: 'Linear' or 'Geometrical'
          * superposition: 'Superimposed' or 'Separated'
          
        """
        # save spatial arrangement
        if spatial_arrangement is not None:
            self.spatial_arrangement = spatial_arrangement
        if superposition is not None:
            self.superposition = superposition
        
        # save box size
        if box_size is not None:
            self.save_box_size(*box_size)
        
        # save probe scale
        if probe_scale is not None:
            self.probe_scale = probe_scale
        
        # retrieve info
        channel_positions = self.channel_positions[self.spatial_arrangement]
        
        size = self.load_box_size()
        if size is None:
            w = h = 0.
        else:
            w, h = size
            
        # update translation vector
        # order: cluster, channel
        T = np.repeat(channel_positions, self.nclusters, axis=0)
        Tx = np.reshape(T[:,0], (self.nchannels, self.nclusters))
        Ty = np.reshape(T[:,1], (self.nchannels, self.nclusters))
        
        # take probe scale into account
        psx, psy = self.probe_scale
        Tx *= psx
        Ty *= psy
        
        # shift in the separated case
        if self.superposition == 'Separated':
            clusters = np.tile(np.arange(self.nclusters), (self.nchannels, 1))
            Tx += w * (1 + 2 * self.alpha) * \
                                    (.5 + clusters - self.nclusters / 2.)

        # record box positions and size
        self.box_positions = Tx, Ty
        self.box_size = (w, h)
        
    def get_transformation(self):
        return self.box_positions, self.box_size

        
    # Geometry preferences
    # --------------------
    def set_geometry_preferences(self, pref=None):
        if pref is None:
            return
        self.spatial_arrangement = pref['spatial_arrangement']
        self.superposition = pref['superposition']
        self.box_sizes[self.spatial_arrangement] = pref['box_size']
        self.probe_scale = pref['probe_scale']
        
    def get_geometry_preferences(self):
        return {
            'spatial_arrangement': self.spatial_arrangement,
            'superposition': self.superposition,
            'box_size': self.load_box_size(effective=False),
            'probe_scale': self.probe_scale,
        }
        

    # Internal methods
    # ----------------
    def save_box_size(self, w, h, arrangement=None):
        if arrangement is None:
            arrangement = self.spatial_arrangement
        self.box_sizes[arrangement] = (w, h)

    def load_box_size(self, arrangement=None, effective=True):
        if arrangement is None:
            arrangement = self.spatial_arrangement
        size = self.box_sizes[arrangement]
        if size is None:
            size = .5, .1
        w, h = size
        # effective box width
        if effective and self.superposition == 'Separated' and self.nclusters >= 1:
            w = w / self.nclusters
        return w, h
    
        
    # Interactive update methods
    # --------------------------
    def change_box_scale(self, dsx, dsy):
        w, h = self.load_box_size(effective=False)
        # the w in box size has been divided by the number of clusters, so we
        # remultiply it to have the normalized value
        # w *= self.nclusters
        w = max(self.box_size_min, w * (1 + dsx))
        h = max(self.box_size_min, h * (1 + dsy))
        self.update_arrangement(box_size=(w, h))
        self.paint_manager.auto_update_uniforms("box_size", "box_size_margin")
        
    def change_probe_scale(self, dsx, dsy):
        # w, h = self.load_box_size()
        sx, sy = self.probe_scale
        sx = max(self.probe_scale_min, sx + dsx)
        sy = max(self.probe_scale_min, sy + dsy)
        self.update_arrangement(probe_scale=(sx, sy))
        self.paint_manager.auto_update_uniforms("probe_scale")
        
    def toggle_superposition(self):
        # switch superposition
        if self.superposition == 'Separated':
            self.superposition = 'Superimposed'
        else:
            self.superposition = 'Separated'
        # recompute the waveforms positions
        self.update_arrangement(superposition=self.superposition,
                                spatial_arrangement=self.spatial_arrangement)
        self.paint_manager.auto_update_uniforms("superimposed", "box_size", "box_size_margin")

    def toggle_spatial_arrangement(self):
        # switch spatial arrangement
        if self.spatial_arrangement == 'Linear':
            self.spatial_arrangement = 'Geometrical'
        else:
            self.spatial_arrangement = 'Linear'
        # recompute the waveforms positions
        self.update_arrangement(superposition=self.superposition,
                                spatial_arrangement=self.spatial_arrangement)
        self.paint_manager.auto_update_uniforms("channel_positions", "box_size", "box_size_margin")
        
        
    # Get methods
    # -----------
    def get_viewbox(self, channels):
        """Return the smallest viewbox such that the selected channels are
        visible.
        """
        w, h = self.load_box_size(effective=False)
        channels = np.array(channels)
        if channels.size == 0:
            return (-1., -1., 1., 1.)
        Tx, Ty = self.box_positions
        # find the box enclosing all channels center positions
        xmin, xmax = Tx[channels,:].min(), Tx[channels,:].max()
        ymin, ymax = Ty[channels,:].min(), Ty[channels,:].max()
        # take the size of the individual boxes into account
        # mx = w * (.5 + self.alpha)
        # my = h * (.5 + self.alpha)
        xmin -= w / 2
        xmax += w / 2
        ymin -= h / 2
        ymax += h / 2
        return xmin, ymin, xmax, ymax
        
    def find_box(self, xp, yp):
        if self.nclusters == 0:
            return 0, 0
        
        # transformation
        box_positions, box_size = self.get_transformation()
        Tx, Ty = box_positions
        w, h = box_size
        a, b = w / 2, h / 2
        
        # find the enclosed channels and clusters
        sx, sy = self.interaction_manager.get_processor('navigation').sx, self.interaction_manager.get_processor('navigation').sy
        dist = (np.abs(Tx - xp) * sx) ** 2 + (np.abs(Ty - yp) * sy) ** 2
        closest = np.argmin(dist.ravel())
        channel, cluster_rel = closest // self.nclusters, np.mod(closest, self.nclusters)
        
        return channel, cluster_rel
        
    def get_enclosed_channels(self, box):
        # channel positions: Nchannels x Nclusters
        Tx, Ty = self.box_positions
        
        w, h = self.box_size
        # print w, h
        # enclosed box
        x0, y0, x1, y1 = box
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        ind = ((x0 <= Tx + w/2) & (x1 >= Tx - w/2) &
               (y0 <= Ty + h/2) & (y1 >= Ty - h/2))
        return np.nonzero(ind)


class WaveformDataManager(Manager):
    # Initialization methods
    # ----------------------
    def set_data(self,
                 waveforms=None,
                 channels=None,
                 masks=None,
                 clusters=None,
                 # list of clusters that are selected, the order matters
                 clusters_selected=None,
                 cluster_colors=None,
                 geometrical_positions=None,
                 keep_order=None,
                 autozoom=None,
                 ):
                 
        self.autozoom = autozoom
        if clusters_selected is None:
            clusters_selected = []
        if waveforms is None:
            waveforms = np.zeros((0, 1, 1))
            masks = np.zeros((0, 1))
            clusters = np.zeros(0, dtype=np.int32)
            cluster_colors = np.zeros(0, dtype=np.int32)
            clusters_selected = []
            
        self.keep_order = keep_order
        
        # Not all waveforms have been selected, so select the appropriate 
        # samples in clusters and masks.
        self.waveform_indices = get_indices(waveforms)
        self.waveform_indices_array = get_array(self.waveform_indices)
        masks = select(masks, self.waveform_indices)
        clusters = select(clusters, self.waveform_indices)
        
        # Convert from Pandas into raw NumPy arrays.
        self.waveforms_array = get_array(waveforms)
        self.masks_array = get_array(masks)
        self.clusters_array = get_array(clusters)
        # Relative indexing.
        if len(clusters_selected) > 0 and len(self.waveform_indices) > 0:
            self.clusters_rel = np.array(np.digitize(self.clusters_array, 
                sorted(clusters_selected)) - 1, dtype=np.int32)
            self.clusters_rel_ordered = (np.argsort(clusters_selected)
                [self.clusters_rel]).astype(np.int32)
        else:
            self.clusters_rel = np.zeros(0, dtype=np.int32)
            self.clusters_rel_ordered = np.zeros(0, dtype=np.int32)
            
        if self.keep_order:
            self.clusters_rel_ordered2 = self.clusters_rel_ordered
            self.cluster_colors_array = get_array(cluster_colors, dosort=True)[np.argsort(clusters_selected)]
            self.clusters_selected2 = clusters_selected
        else:
            self.clusters_rel_ordered2 = self.clusters_rel
            self.cluster_colors_array = get_array(cluster_colors, dosort=True)
            self.clusters_selected2 = sorted(clusters_selected)
            
        self.nspikes, self.nsamples, self.nchannels = self.waveforms_array.shape
        if channels is None:
            channels = range(self.nchannels)
        self.channels = channels
        self.npoints = self.waveforms_array.size
        self.geometrical_positions = geometrical_positions
        self.clusters_selected = clusters_selected
        self.clusters_unique = sorted(clusters_selected)
        self.nclusters = len(clusters_selected)
        self.waveforms = waveforms
        self.clusters = clusters
        # self.cluster_colors = cluster_colors
        self.masks = masks
        
        # Prepare GPU data.
        self.data = self.prepare_waveform_data()
        self.masks_full = np.repeat(self.masks_array.T.ravel(), self.nsamples)
        self.clusters_full = np.tile(np.repeat(self.clusters_rel_ordered2, self.nsamples), self.nchannels)
        self.clusters_full_depth = np.tile(np.repeat(self.clusters_rel_ordered, self.nsamples), self.nchannels)
        self.channels_full = np.repeat(np.arange(self.nchannels, dtype=np.int32), self.nspikes * self.nsamples)
        
        # Compute average waveforms.
        self.data_avg = self.prepare_average_waveform_data()
        self.masks_full_avg = np.repeat(self.masks_avg.T.ravel(), self.nsamples_avg)
        self.clusters_full_avg = np.tile(np.repeat(self.clusters_rel_ordered_avg2, self.nsamples_avg), self.nchannels_avg)
        self.clusters_full_depth_avg = np.tile(np.repeat(self.clusters_rel_ordered_avg, self.nsamples_avg), self.nchannels_avg)
        self.channels_full_avg = np.repeat(np.arange(self.nchannels_avg, dtype=np.int32), self.nspikes_avg * self.nsamples_avg)
        
        # position waveforms
        self.position_manager.set_info(self.nchannels, self.nclusters, 
           geometrical_positions=self.geometrical_positions,)
        
        # update the highlight manager
        self.highlight_manager.initialize()
    
    
    # Internal methods
    # ----------------
    def prepare_waveform_data(self):
        """Define waveform data."""
        X = np.tile(np.linspace(-1., 1., self.nsamples),
                                (self.nchannels * self.nspikes, 1))
        
        # new: use strides to avoid unnecessary memory copy
        strides = self.waveforms_array.strides
        strides = (strides[2], strides[0], strides[1])

        shape = self.waveforms_array.shape
        shape = (shape[2], shape[0], shape[1])
        Y = as_strided(self.waveforms_array, strides=strides, shape=shape)
        
        # create a Nx2 array with all coordinates
        data = np.empty((X.size, 2), dtype=np.float32)
        data[:,0] = X.ravel()
        data[:,1] = Y.ravel()
        return data
    
    def prepare_average_waveform_data(self):
        waveforms_avg = np.zeros((self.nclusters, self.nsamples, self.nchannels))
        waveforms_std = np.zeros((self.nclusters, self.nsamples, self.nchannels))
        self.masks_avg = np.zeros((self.nclusters, self.nchannels))
        for i, cluster in enumerate(self.clusters_unique):
            spike_indices = get_spikes_in_clusters(cluster, self.clusters)
            w = select(self.waveforms, spike_indices)
            m = select(self.masks, spike_indices)
            waveforms_avg[i,...] = w.mean(axis=0)
            waveforms_std[i,...] = w.std(axis=0).mean()
            self.masks_avg[i,...] = m.mean(axis=0)
        
        # create X coordinates
        X = np.tile(np.linspace(-1., 1., self.nsamples),
                        (self.nchannels * self.nclusters, 1))
        # create Y coordinates
        if self.nclusters == 0:
            Y = np.array([], dtype=np.float32)
            thickness = np.array([], dtype=np.float32)
        else:
            Y = np.vstack(waveforms_avg)
            thickness = np.vstack(waveforms_std).T.ravel()
        
        # concatenate data
        data = np.empty((X.size, 2), dtype=np.float32)
        data[:,0] = X.ravel()
        data[:,1] = Y.T.ravel()
        
        if self.nclusters > 0:
            # thicken
            w = thickness.reshape((-1, 1))
            n = waveforms_avg.size
            Y = np.zeros((2 * n, 2))
            u = np.zeros((n, 2))
            u[1:,0] = -np.diff(data[:,1])
            u[1:,1] = data[1,0] - data[0,0]
            u[0,:] = u[1,:]
            r = (u[:,0] ** 2 + u[:,1] ** 2) ** .5
            r[r == 0.] = 1
            u[:,0] /= r
            u[:,1] /= r
            Y[::2,:] = data - w * u
            Y[1::2,:] = data + w * u
            data_thickened = Y
        else:
            n = 0
            data_thickened = data
        
        self.nsamples_avg = self.nsamples * 2
        self.npoints_avg = waveforms_avg.size * 2
        self.nspikes_avg = self.nclusters
        self.nclusters_avg = self.nclusters
        self.nchannels_avg = self.nchannels
        self.clusters_rel_avg = np.arange(self.nclusters, dtype=np.int32)
        self.clusters_rel_ordered_avg = np.argsort(self.clusters_selected)[self.clusters_rel_avg]
        if self.keep_order:
            self.clusters_rel_ordered_avg2 = self.clusters_rel_ordered_avg
        else:
            self.clusters_rel_ordered_avg2 = self.clusters_rel_avg
        
        return data_thickened


# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class WaveformVisual(Visual):
    @staticmethod
    def get_size_bounds(nsamples=None, npoints=None):
        size = npoints
        bounds = np.arange(0, npoints + 1, nsamples)
        return size, bounds

    def initialize(self, nclusters=None, nchannels=None, 
        nsamples=None, npoints=None, #nspikes=None,
        position0=None, mask=None, cluster=None, cluster_depth=None,
        cluster_colors=None, channel=None, highlight=None,
        average=None):

        self.size, self.bounds = WaveformVisual.get_size_bounds(nsamples, npoints)
        
        self.primitive_type = 'LINE_STRIP'
        
        # NEW: add depth
        # depth = np.zeros((self.size, 0))
        # position = np.hstack((position0, depth))
        position = position0
        
        self.add_attribute("position0", vartype="float", ndim=2, data=position)
        
        self.add_attribute("mask", vartype="float", ndim=1, data=mask)
        self.add_varying("vmask", vartype="float", ndim=1)
        
        self.add_attribute("cluster", vartype="int", ndim=1, data=cluster)
        self.add_attribute("cluster_depth", vartype="int", ndim=1, data=cluster_depth)
        
        self.add_attribute("channel", vartype="int", ndim=1, data=channel)
        
        self.add_attribute("highlight", vartype="int", ndim=1, data=highlight)
        self.add_varying("vhighlight", vartype="int", ndim=1)
        
        self.add_uniform("toggle_mask", vartype="int", ndim=1, data=1)
        
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        self.add_uniform("box_size", vartype="float", ndim=2)
        self.add_uniform("box_size_margin", vartype="float", ndim=2)
        self.add_uniform("probe_scale", vartype="float", ndim=2)
        self.add_uniform("superimposed", vartype="bool", ndim=1)
        
        # HACK: maximum number of channels
        self.add_uniform("channel_positions", vartype="float", ndim=2,
            size=1000)
        
        ncolors = COLORMAP_TEXTURE.shape[1]
        ncomponents = COLORMAP_TEXTURE.shape[2]
        
        global FRAGMENT_SHADER
        if average:
            FRAGMENT_SHADER = FRAGMENT_SHADER_AVERAGE
            
        cmap_index = cluster_colors[cluster]
        self.add_texture('cmap', ncomponents=ncomponents, ndim=2, data=COLORMAP_TEXTURE)
        self.add_attribute('cmap_index', ndim=1, vartype='int', data=cmap_index)
        self.add_varying('cmap_vindex', vartype='int', ndim=1)
        
        dx = 1. / ncolors
        offset = dx / 2.
        dx_shift = 1. / SHIFTLEN
        offset_shift = dx / 2.
        
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%CMAP_OFFSET%', "%.5f" % offset)
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%CMAP_STEP%', "%.5f" % dx)
        
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%SHIFT_OFFSET%', "%.5f" % offset_shift)
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%SHIFT_STEP%', "%.5f" % dx_shift)
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%SHIFTLEN%', "%d" % (SHIFTLEN - 1))
        
        
        # necessary so that the navigation shader code is updated
        self.is_position_3D = True
        
        self.add_vertex_main(VERTEX_SHADER)
        self.add_fragment_main(FRAGMENT_SHADER)


class AverageWaveformVisual(WaveformVisual):
    def initialize(self, *args, **kwargs):
        super(AverageWaveformVisual, self).initialize(*args, **kwargs)
        self.primitive_type = 'TRIANGLE_STRIP'


class WaveformPaintManager(PlotPaintManager):
    def get_uniform_value(self, name):
        if name == "box_size":
            size = self.position_manager.load_box_size()
            if size is None:
                w = h = 0.
            else:
                w, h = size
            return (w, h)
        if name == "box_size_margin":
            size = self.position_manager.load_box_size()
            if size is None:
                w = h = 0.
            else:
                w, h = size
            alpha, beta = self.position_manager.alpha, self.position_manager.beta
            return (w * (1 + 2 * alpha), h * (1 + 2 * beta))
        if name == "probe_scale":
            return self.position_manager.probe_scale
        if name == "superimposed":
            return self.position_manager.superposition == 'Superimposed'
        # if name == "cluster_colors":
            # return self.data_manager.cluster_colors
        if name == "channel_positions":
            pos = self.position_manager.get_channel_positions()
            return pos
    
    def auto_update_uniforms(self, *names):
        dic = dict([(name, self.get_uniform_value(name)) for name in names])
        self.set_data(visual='waveforms', **dic)
        self.set_data(visual='waveforms_avg', **dic)
        
    def initialize(self):
        if not hasattr(self.data_manager, 'npoints'):
            return
        
        self.add_visual(WaveformVisual, name='waveforms',
            npoints=self.data_manager.npoints,
            nchannels=self.data_manager.nchannels,
            nclusters=self.data_manager.nclusters,
            cluster_depth=self.data_manager.clusters_full_depth,
            nsamples=self.data_manager.nsamples,
            position0=self.data_manager.data,
            cluster_colors=self.data_manager.cluster_colors_array,
            mask=self.data_manager.masks_full,
            cluster=self.data_manager.clusters_full,
            channel=self.data_manager.channels_full,
            highlight=self.highlight_manager.highlight_mask)
            
        # average waveforms
        self.add_visual(AverageWaveformVisual, name='waveforms_avg',
            average=True,
            npoints=self.data_manager.npoints_avg,
            nchannels=self.data_manager.nchannels_avg,
            nclusters=self.data_manager.nclusters_avg,
            cluster_depth=self.data_manager.clusters_full_depth_avg,
            nsamples=self.data_manager.nsamples_avg,
            position0=self.data_manager.data_avg,
            cluster_colors=self.data_manager.cluster_colors_array,
            mask=self.data_manager.masks_full_avg,
            cluster=self.data_manager.clusters_full_avg,
            channel=self.data_manager.channels_full_avg,
            highlight=np.zeros(self.data_manager.npoints_avg, dtype=np.int32),
            visible=False)
        
        self.auto_update_uniforms("box_size", "box_size_margin", "probe_scale",
            "superimposed", "channel_positions",)
        
        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            # posoffset=(30., -30.),
            coordinates=(1., -1.),
            posoffset=(-180., 30.),
            is_static=True,
            color=(1., 1., 1., 1.),
            background_transparent=False,
            letter_spacing=350.,
            depth=-1,
            visible=False)
        
    def update(self):
        size, bounds = WaveformVisual.get_size_bounds(
            self.data_manager.nsamples, self.data_manager.npoints)
        cluster = self.data_manager.clusters_full
        cluster_colors = self.data_manager.cluster_colors_array
        cmap_index = cluster_colors[cluster]
        
        box_size = self.get_uniform_value('box_size')
        box_size_margin = self.get_uniform_value('box_size_margin')
        channel_positions = self.get_uniform_value('channel_positions')
    
        self.set_data(visual='waveforms', 
            size=size,
            bounds=bounds,
            nclusters=self.data_manager.nclusters,
            position0=self.data_manager.data,
            mask=self.data_manager.masks_full,
            cluster=self.data_manager.clusters_full,
            cluster_depth=self.data_manager.clusters_full_depth,
            cmap_index=cmap_index,
            channel=self.data_manager.channels_full,
            highlight=self.highlight_manager.highlight_mask,
            # auto update uniforms
            box_size=box_size,
            box_size_margin=box_size_margin,
            channel_positions=channel_positions,
            )
            
        # average waveforms
        size, bounds = WaveformVisual.get_size_bounds(self.data_manager.nsamples_avg, self.data_manager.npoints_avg)
        cluster = self.data_manager.clusters_full_avg
        cluster_colors = self.data_manager.cluster_colors_array
        cmap_index = cluster_colors[cluster]
        
        self.set_data(visual='waveforms_avg', 
            size=size,
            bounds=bounds,
            nclusters=self.data_manager.nclusters_avg,
            position0=self.data_manager.data_avg,
            mask=self.data_manager.masks_full_avg,
            cluster=self.data_manager.clusters_full_avg,
            cluster_depth=self.data_manager.clusters_full_depth_avg,
            cmap_index=cluster_colors[self.data_manager.clusters_full_avg],
            channel=self.data_manager.channels_full_avg,
            highlight=np.zeros(size, dtype=np.int32),
            # auto update uniforms
            box_size=box_size,
            box_size_margin=box_size_margin,
            channel_positions=channel_positions,
            )
        
        if self.data_manager.autozoom and size > 0:
            self.interaction_manager.autozoom()
        

# -----------------------------------------------------------------------------
# Interactivity
# -----------------------------------------------------------------------------
class WaveformHighlightManager(HighlightManager):
    def initialize(self):
        """Set info from the data manager."""
        super(WaveformHighlightManager, self).initialize()
        data_manager = self.data_manager
        # self.get_data_position = self.data_manager.get_data_position
        self.masks_full = self.data_manager.masks_full
        self.clusters_rel = self.data_manager.clusters_rel
        self.clusters_rel_ordered = self.data_manager.clusters_rel_ordered
        self.clusters_rel_ordered2 = self.data_manager.clusters_rel_ordered2
        self.cluster_colors = self.data_manager.cluster_colors_array
        self.nchannels = data_manager.nchannels
        self.nclusters = data_manager.nclusters
        self.nsamples = data_manager.nsamples
        self.nspikes = data_manager.nspikes
        self.npoints = data_manager.npoints
        self.waveforms = self.data_manager.waveforms
        self.waveforms_array = self.data_manager.waveforms_array
        self.waveform_indices = self.data_manager.waveform_indices
        self.highlighted_spikes = []
        self.highlight_mask = np.zeros(self.npoints, dtype=np.int32)
        self.highlighting = False

    def find_enclosed_spikes(self, enclosing_box):
        """Return relative indices."""
        if self.nspikes == 0:
            return np.array([])
        
        # first call
        if not self.highlighting:
            # create
            box_positions, box_size = self.position_manager.get_transformation()
            # Tx, Ty: Nchannels x Nclusters
            Tx, Ty = box_positions
            # to: Nspikes x Nsamples x Nchannels
            Px = np.tile(Tx[:,self.clusters_rel_ordered2].reshape((self.nchannels, self.nspikes, 1)), (1, 1, self.nsamples))
            self.Px = Px.transpose([1, 2, 0])
            Py = np.tile(Ty[:,self.clusters_rel_ordered2].reshape((self.nchannels, self.nspikes, 1)), (1, 1, self.nsamples))
            self.Py = Py.transpose([1, 2, 0])
            
            self.Wx = np.tile(np.linspace(-1., 1., self.nsamples).reshape((1, -1, 1)), (self.nspikes, 1, self.nchannels))
            self.Wy = self.waveforms_array
            
            self.highlighting = True
        
        x0, y0, x1, y1 = enclosing_box
        
        # press_position
        xp, yp = x0, y0

        # reorder
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        # transformation
        box_positions, box_size = self.position_manager.get_transformation()
        Tx, Ty = box_positions
        w, h = box_size
        a, b = w / 2, h / 2
        
        # find the enclosed channels and clusters
        channels, clusters = self.position_manager.get_enclosed_channels((x0, y0, x1, y1))
        channels = np.unique(channels)
        clusters = np.unique(clusters)
        
        if channels.size == 0:
            return np.array([])
        
        u, v = self.Px[:,:,channels], self.Py[:,:,channels]
        Wx, Wy = self.Wx[:,:,channels], self.Wy[:,:,channels]
        
        ind =  ((Wx >= (xmin-u)/a) & (Wx <= (xmax-u)/a) & \
                (Wy >= (ymin-v)/b) & (Wy <= (ymax-v)/b))
        
        spkindices = np.nonzero(ind.max(axis=1).max(axis=1))[0]

        return spkindices

    def find_indices_from_spikes(self, spikes):
        if spikes is None or len(spikes)==0:
            return None
        n = len(spikes)
        # find point indices in the data buffer corresponding to 
        # the selected spikes. In particular, waveforms of those spikes
        # across all channels should be selected as well.
        spikes = np.array(spikes, dtype=np.int32)
        ind = np.repeat(spikes * self.nsamples, self.nsamples)
        ind += np.tile(np.arange(self.nsamples), n)
        ind = np.tile(ind, self.nchannels)
        ind += np.repeat(np.arange(self.nchannels) * self.nsamples * self.nspikes,
                                self.nsamples * n)
        return ind

    def set_highlighted_spikes(self, spikes=[]):
        """Update spike colors to mark transiently selected spikes with
        a special color."""
        
        if len(spikes) == 0:
            # do update only if there were previously selected spikes
            do_update = len(self.highlighted_spikes) > 0
            self.highlight_mask[:] = 0
        else:
            do_update = True
            self.highlight_mask[:] = 0
            if len(spikes) > 0:
                ind = self.find_indices_from_spikes(spikes)
                self.highlight_mask[ind] = 1
        
        if do_update:
            self.paint_manager.set_data(
                highlight=self.highlight_mask,
                visual='waveforms')
        
        self.highlighted_spikes = spikes

    def highlighted(self, box):
        # Get selected spikes (relative indices).
        spikes = self.find_enclosed_spikes(box)
        # Set highlighted spikes.
        self.set_highlighted_spikes(spikes)
        # Emit the HighlightSpikes signal.
        self.emit(spikes)
        
    def highlight_spikes(self, spikes):
        """spikes in absolute indices."""
        spikes = np.intersect1d(self.data_manager.waveform_indices_array, 
            spikes)
        if len(spikes) > 0:
            spikes_rel = np.digitize(spikes, 
                self.data_manager.waveform_indices_array) - 1
            self.highlighting = True
            self.set_highlighted_spikes(spikes_rel)
        else:
            self.cancel_highlight()
            
    def cancel_highlight(self):
        super(WaveformHighlightManager, self).cancel_highlight()
        if self.highlighting:
            self.set_highlighted_spikes([])
            self.emit([])
            self.highlighting = False
    
    def emit(self, spikes):
        spikes = np.array(spikes, dtype=np.int32)
        spikes_abs = self.waveform_indices[spikes]
        self.parent.spikesHighlighted.emit(spikes_abs)


class WaveformInfoManager(Manager):
    def show_closest_cluster(self, xd, yd):
        channel, cluster_rel = self.position_manager.find_box(xd, yd)
        text = "{0:d} on channel {1:d}".format(
            self.data_manager.clusters_selected2[cluster_rel],
            self.data_manager.channels[channel],
            )
        
        self.paint_manager.set_data(
            text=text,
            visible=True,
            visual='clusterinfo')


class WaveformInteractionManager(PlotInteractionManager):
    def select_channel(self, coord, xp, yp):
        
        if self.data_manager.nclusters == 0:
            return
        
        # normalized coordinates
        xp, yp = self.get_processor('navigation').get_data_coordinates(xp, yp)
        # find closest channel
        channel, cluster_rel = self.position_manager.find_box(xp, yp)
        cluster = self.data_manager.clusters_unique[cluster_rel]
        # emit the boxClicked signal
        # log.debug("Select cluster {0:d}, channel {1:d} on axis {2:s}.".
            # format(cluster, channel, 'xy'[coord]))
        self.parent.boxClicked.emit(coord, cluster, channel)
    
    def initialize(self):
        self.toggle_mask_value = True
        
        self.register('ToggleSuperposition', self.toggle_superposition)
        self.register('ToggleSpatialArrangement', self.toggle_spatial_arrangement)
        self.register('ChangeBoxScale', self.change_box_scale)
        self.register('ChangeProbeScale', self.change_probe_scale)
        self.register('HighlightSpike', self.highlight_spikes)
        self.register('SelectChannel', self.select_channel_callback)
        self.register('ToggleAverage', self.toggle_average)
        self.register('ShowClosestCluster', self.show_closest_cluster)
        self.register('ToggleMask', self.toggle_mask)
        self.register(None, self.none_callback)
        self.average_toggled = False
  
    def toggle_average(self, parameter):
        self.average_toggled = not(self.average_toggled)
        self.paint_manager.set_data(visible=self.average_toggled,
            visual='waveforms_avg')
        self.paint_manager.set_data(visible=not(self.average_toggled),
            visual='waveforms')
  
    def toggle_superposition(self, parameter):
        self.position_manager.toggle_superposition()
        
    def toggle_spatial_arrangement(self, parameter):
        self.position_manager.toggle_spatial_arrangement()
        
    def change_box_scale(self, parameter):
        self.position_manager.change_box_scale(*parameter)
    
    def change_probe_scale(self, parameter):
        self.position_manager.change_probe_scale(*parameter)
        
    def highlight_spikes(self, parameter):
        self.highlight_manager.highlight(parameter)
        self.cursor = 'CrossCursor'
    
    def select_channel_callback(self, parameter):
        self.select_channel(*parameter)
        
    def none_callback(self, parameter):
        self.highlight_manager.cancel_highlight()
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        
    def show_closest_cluster(self, parameter):
        self.cursor = None
        
        nav = self.get_processor('navigation')
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        if self.data_manager.nspikes == 0:
            return
            
        self.info_manager.show_closest_cluster(xd, yd)
        
    def autozoom(self):
        channels = np.argsort(self.data_manager.masks_array.sum(axis=0))[::-1]
        channels = channels[:2]
        viewbox = self.position_manager.get_viewbox(channels)
        self.parent.process_interaction('SetViewbox', viewbox)
        
    # Misc
    # ----
    def toggle_mask(self, parameter=None):
        self.toggle_mask_value = 1 - self.toggle_mask_value
        self.paint_manager.set_data(visual='waveforms', 
            toggle_mask=self.toggle_mask_value)
        
    
class WaveformBindings(KlustaViewaBindings):
    def set_zoombox_keyboard(self):
        """Set zoombox bindings with the keyboard."""
        self.set('MiddleClickMove', 'ZoomBox',
                    # key_modifier='Shift',
                    param_getter=lambda p: (p["mouse_press_position"][0],
                                            p["mouse_press_position"][1],
                                            p["mouse_position"][0],
                                            p["mouse_position"][1]))
                                            
    def set_arrangement_toggling(self):
        # toggle superposition
        self.set('KeyPress',
                 'ToggleSuperposition',
                 key='O')
                 
        # toggle spatial arrangement
        self.set('KeyPress',
                 'ToggleSpatialArrangement',
                 key='P')
                               
    def set_average_toggling(self):
        # toggle average
        self.set('KeyPress',
                 'ToggleAverage',
                 key='M')
                 
    def set_box_scaling(self):
        ww = .002
        self.set('Wheel',
                 'ChangeBoxScale',
                 description='vertical',
                 key_modifier='Control',
                 param_getter=lambda p: (0, p["wheel"] * ww))
        self.set('Wheel',
                 'ChangeBoxScale',
                 description='horizontal',
                 key_modifier='Shift',
                 param_getter=lambda p: (p["wheel"] * ww, 0))
                 
        w = .1
        self.set('KeyPress',
                 'ChangeBoxScale',
                 description='vertical',
                 key='I',
                 param_getter=lambda p: (0, w))
        self.set('KeyPress',
                 'ChangeBoxScale',
                 description='vertical',
                 key='D',
                 param_getter=lambda p: (0, -w))
        self.set('KeyPress',
                 'ChangeBoxScale',
                 description='horizontal',
                 key='I', key_modifier='Control',
                 param_getter=lambda p: (w, 0))
        self.set('KeyPress',
                 'ChangeBoxScale',
                 description='horizontal',
                 key='D', key_modifier='Control',
                 param_getter=lambda p: (-w, 0))
                 
    def set_probe_scaling(self):
        self.set('RightClickMove',
                 'ChangeProbeScale',
                 key_modifier='Control',
                 param_getter=lambda p: (p["mouse_position_diff"][0] * 1,
                                         p["mouse_position_diff"][1] * 1))

    def set_highlight(self):
        # highlight
        # self.set('MiddleClickMove',
                 # 'HighlightSpike',
                 # param_getter=lambda p: (p["mouse_press_position"][0],
                                         # p["mouse_press_position"][1],
                                         # p["mouse_position"][0],
                                         # p["mouse_position"][1]))
        
        self.set('LeftClickMove',
                 'HighlightSpike',
                 key_modifier='Control',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],
                                         p["mouse_position"][0],
                                         p["mouse_position"][1]))
        
    def set_channel_selection(self):
        # CTRL + left click for selecting a channel for coordinate X in feature view
        self.set('LeftClick', 'SelectChannel',
                 key_modifier='Control',
                 param_getter=lambda p: (0, p["mouse_position"][0], p["mouse_position"][1]))
        # CTRL + right click for selecting a channel for coordinate Y in feature view
        self.set('RightClick', 'SelectChannel',
                 key_modifier='Control',
                 param_getter=lambda p: (1, p["mouse_position"][0], p["mouse_position"][1]))
        
    def set_clusterinfo(self):
        self.set('Move', 'ShowClosestCluster', #key_modifier='Shift',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
        
    def set_toggle_mask(self):
        self.set('KeyPress', 'ToggleMask', key='Y')
        
    def initialize(self):
        # super(WaveformBindings, self).initialize()
        self.set_arrangement_toggling()
        self.set_average_toggling()
        self.set_box_scaling()
        self.set_probe_scaling()
        self.set_highlight()
        self.set_channel_selection()
        self.set_clusterinfo()
        self.set_toggle_mask()


# -----------------------------------------------------------------------------
# Top-level widget
# -----------------------------------------------------------------------------
class WaveformView(KlustaView):
    # Signals
    # -------
    # Raise (cluster, channel) when a box is selected.
    boxClicked = QtCore.pyqtSignal(int, int, int)
    # Raise the list of highlighted spike absolute indices.
    spikesHighlighted = QtCore.pyqtSignal(np.ndarray)
    
    # Initialization
    # --------------
    def initialize(self):
        self.constrain_ratio = False
        self.activate3D = True
        self.set_bindings(WaveformBindings)
        self.set_companion_classes(
                data_manager=WaveformDataManager,
                position_manager=WaveformPositionManager,
                info_manager=WaveformInfoManager,
                interaction_manager=WaveformInteractionManager,
                paint_manager=WaveformPaintManager,
                highlight_manager=WaveformHighlightManager,
                )
        self.first = True

    def set_data(self, *args, **kwargs):
        if self.first:
            self.restore_geometry()
            
        self.data_manager.set_data(*args, **kwargs)
        
        # update?
        if self.initialized:
            self.paint_manager.update()
            self.updateGL()
            
        self.first = False

        
    # Public methods
    # --------------
    def highlight_spikes(self, spikes):
        self.highlight_manager.highlight_spikes(spikes)
        self.updateGL()

    def toggle_mask(self):
        self.interaction_manager.toggle_mask()
        self.updateGL()
        
        
    # Save and restore geometry
    # -------------------------
    def save_geometry(self):
        pref = self.position_manager.get_geometry_preferences()
        SETTINGS.set('waveform_view.geometry', pref)
        
    def restore_geometry(self):
        """Return a dictionary with the user preferences regarding geometry
        in the WaveformView."""
        pref = SETTINGS.get('waveform_view.geometry')
        self.position_manager.set_geometry_preferences(pref)
        
    def closeEvent(self, e):
        self.save_geometry()
        super(WaveformView, self).closeEvent(e)
        
        