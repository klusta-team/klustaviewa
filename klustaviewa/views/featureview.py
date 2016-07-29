"""Feature View: show spikes as 2D points in feature space."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import operator
import time

import numpy as np
import numpy.random as rdn

from qtools import QtGui, QtCore, show_window
from galry import (Manager, PlotPaintManager, PlotInteractionManager, Visual,
    GalryWidget, enforce_dtype, RectanglesVisual,
    TextVisual, PlotVisual, AxesVisual, GridVisual, NavigationEventProcessor,
    EventProcessor, DataNormalizer)

from kwiklib.dataio.selection import get_indices, select
from kwiklib.dataio.tools import get_array
from klustaviewa.views.common import HighlightManager, KlustaViewaBindings, KlustaView
from kwiklib.utils.colors import COLORMAP_TEXTURE, SHIFTLEN, COLORMAP
from klustaviewa import USERPREF
from kwiklib.utils import logger as log
import klustaviewa


# -----------------------------------------------------------------------------
# Shaders
# -----------------------------------------------------------------------------
VERTEX_SHADER = """
    // move the vertex to its position
    vec3 position = vec3(0, 0, 0);
    position.xy = position0;

    vhighlight = highlight;
    cmap_vindex = cmap_index;
    vmask = mask;
    vselection = selection;


    // compute the depth: put masked spikes on the background, unmasked ones
    // on the foreground on a different layer for each cluster
    float depth = 0.;
    //if (mask == 1.)
    depth = -(cluster_depth + 1) / (nclusters + 10);
    position.z = depth;

    if ((highlight > 0) || (selection > 0))
        gl_PointSize = 5.;
    else
        gl_PointSize = u_point_size;

    // DEBUG
    //gl_PointSize = 20;
"""

FRAGMENT_SHADER = """
    float index = %CMAP_OFFSET% + cmap_vindex * %CMAP_STEP%;
    vec2 index2d = vec2(index, %SHIFT_OFFSET% + (1 + toggle_mask * (1 - vmask) * %SHIFTLEN%) * %SHIFT_STEP%);
    if (vhighlight > 0) {{
        index2d.y = 0;
        out_color = texture2D(cmap, index2d);
        out_color.w = .85;
    }}
    else {{
        out_color = texture2D(cmap, index2d);
        out_color.w = {0:.3f};
    }}
"""

# Background spikes.
VERTEX_SHADER_BACKGROUND = """
    // move the vertex to its position
    vec3 position = vec3(0, 0, 0);
    position.xy = position0;

    position.z = 0.;

    gl_PointSize = u_point_size;
"""

FRAGMENT_SHADER_BACKGROUND = """
    out_color = vec4(.75, .75, .75, alpha);
"""


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def polygon_contains_points(polygon, points):
    """Returns the points within a polygon.

    Arguments:
      * polygon: a Nx2 array with the coordinates of the polygon vertices.
      * points: a Nx2 array with the coordinates of the points.

    Returns:
      * arr: a Nx2 array of booleans with the belonging of every point to
        the inside of the polygon.

    """
    try:
        from matplotlib.path import Path
        p = Path(polygon)
        return p.contains_points(points)
    except:
        import matplotlib.nxutils
        return matplotlib.nxutils.points_inside_poly(points, polygon)


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
    def initialize(self):
        self.register('Initialize', self.update_axes)
        self.register('Pan', self.update_axes)
        self.register('Zoom', self.update_axes)
        self.register('Reset', self.update_axes)
        self.register('Animate', self.update_axes)
        self.register(None, self.update_axes)

    def update_viewbox(self):
        # normalization viewbox
        self.normalizer = DataNormalizer()
        self.normalizer.normalize(
            (0, -1, self.parent.data_manager.duration, 1))

    def update_axes(self, parameter):
        nav = self.get_processor('navigation')
        if not nav:
            return

        if not self.parent.projection_manager.grid_visible:
            return

        viewbox = nav.get_viewbox()

        x0, y0, x1, y1 = viewbox
        x0 = self.normalizer.unnormalize_x(x0)
        y0 = self.normalizer.unnormalize_y(y0)
        x1 = self.normalizer.unnormalize_x(x1)
        y1 = self.normalizer.unnormalize_y(y1)
        viewbox = (x0, y0, x1, y1)

        text, coordinates, n = get_ticks_text(*viewbox)

        coordinates[:,0] = self.normalizer.normalize_x(coordinates[:,0])
        coordinates[:,1] = self.normalizer.normalize_y(coordinates[:,1])

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


# -----------------------------------------------------------------------------
# Data manager
# -----------------------------------------------------------------------------
class FeatureDataManager(Manager):
    # Initialization methods
    # ----------------------
    def set_data(self,
                 features=None,
                 features_background=None,
                 spiketimes=None,  # a subset of all spikes, disregarding cluster
                 masks=None,  # masks for all spikes in selected clusters
                 clusters=None,  # clusters for all spikes in selected clusters
                 clusters_selected=None,
                 cluster_colors=None,
                 fetdim=None,
                 nchannels=None,
                 channels=None,
                 nextrafet=None,
                 autozoom=None,  # None, or the target cluster
                 duration=None,
                 freq=None,
                 alpha_selected=.75,
                 alpha_background=.25,
                 time_unit=None,
                 ):

        if features is None:
            features = np.zeros((0, 2))
            features_background = np.zeros((0, 2))
            masks = np.zeros((0, 1))
            clusters = np.zeros(0, dtype=np.int32)
            clusters_selected = []
            cluster_colors = np.zeros(0, dtype=np.int32)
            fetdim = 2
            nchannels = 1
            nextrafet = 0

        if features.shape[1] == 1:
            features = np.tile(features, (1, 4))
        if features_background.shape[1] == 1:
            features_background = np.tile(features_background, (1, 4))

        assert fetdim is not None

        self.duration = duration
        self.spiketimes = spiketimes
        self.freq = freq
        self.interaction_manager.get_processor('grid').update_viewbox()

        # Feature background alpha value.
        self.alpha_selected = alpha_selected
        self.alpha_background = alpha_background

        # can be 'second' or 'samples'
        self.time_unit = time_unit

        # Extract the relevant spikes, but keep the other ones in features_full
        self.clusters = clusters

        # Contains all spikes, needed for splitting.
        self.features_full = features
        self.features_full_array = get_array(features)

        # Keep a subset of all spikes in the view.
        self.nspikes_full = len(features)
        # > features_nspikes_per_cluster_max spikes ==> take a selection
        nspikes_max = USERPREF.get('features_nspikes_per_cluster_max', 1000)
        k = self.nspikes_full // nspikes_max + 1
        # self.features = features[::k]
        subsel = slice(None, None, k)
        self.features = select(features, subsel)
        self.features_array = get_array(self.features)

        # self.features_background contains all non-selected spikes
        self.features_background = features_background
        self.features_background_array = get_array(self.features_background)

        # Background spikes are those which do not belong to the selected clusters
        self.npoints_background = self.features_background_array.shape[0]
        self.nspikes_background = self.npoints_background

        if channels is None:
            channels = range(nchannels)

        self.nspikes, self.ndim = self.features.shape
        self.fetdim = fetdim
        self.nchannels = nchannels
        self.channels = channels
        self.nextrafet = nextrafet
        self.npoints = self.features.shape[0]

        if masks is None:
            masks = np.ones_like(self.features, dtype=np.float32)
        self.masks = masks


        # Subselection
        self.masks = select(self.masks, subsel)
        self.masks_array = get_array(self.masks)
        if self.masks_array.ndim == 1:
            self.masks_array = self.masks_array[:, np.newaxis]
        if self.spiketimes is not None:
            self.spiketimes = select(self.spiketimes, subsel)
        self.clusters = select(self.clusters, subsel)
        self.clusters_array = get_array(self.clusters)


        self.feature_indices = get_indices(self.features)
        self.feature_full_indices = get_indices(self.features_full)
        self.feature_indices_array = get_array(self.feature_indices)

        self.cluster_colors = get_array(cluster_colors, dosort=True)

        # Relative indexing.
        if self.npoints > 0:
            self.clusters_rel = np.digitize(self.clusters_array, sorted(clusters_selected)) - 1
            self.clusters_rel_ordered = np.argsort(clusters_selected)[self.clusters_rel]
        else:
            self.clusters_rel = np.zeros(0, dtype=np.int32)
            self.clusters_rel_ordered = np.zeros(0, dtype=np.int32)

        self.clusters_unique = sorted(clusters_selected)
        self.nclusters = len(clusters_selected)
        self.masks_full = self.masks_array.T.ravel()
        self.clusters_full_depth = self.clusters_rel_ordered
        self.clusters_full = self.clusters_rel

        # prepare GPU data
        self.data = np.empty((self.nspikes, 2), dtype=np.float32)
        self.data_full = np.empty((self.nspikes_full, 2), dtype=np.float32)
        self.data_background = np.empty((self.nspikes_background, 2),
            dtype=np.float32)

        # set initial projection
        self.projection_manager.set_data()
        self.autozoom = autozoom
        if autozoom is None:
            self.projection_manager.reset_projection()
        else:
            self.projection_manager.auto_projection(autozoom)

        # update the highlight manager
        self.highlight_manager.initialize()
        self.selection_manager.initialize()
        self.selection_manager.cancel_selection()


# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class FeatureVisual(Visual):
    def initialize(self, npoints=None,
                    nclusters=None,
                    cluster_depth=None,
                    position0=None,
                    mask=None,
                    cluster=None,
                    highlight=None,
                    selection=None,
                    cluster_colors=None,
                    alpha=None,
                    ):

        self.primitive_type = 'POINTS'
        self.size = npoints

        self.add_attribute("position0", vartype="float", ndim=2, data=position0)

        self.add_attribute("mask", vartype="float", ndim=1, data=mask)
        self.add_varying("vmask", vartype="float", ndim=1)

        self.add_attribute("highlight", vartype="int", ndim=1, data=highlight)
        self.add_varying("vhighlight", vartype="int", ndim=1)

        self.add_uniform("toggle_mask", vartype="int", ndim=1, data=0)
        self.add_uniform("u_point_size", vartype="float", ndim=1, data=USERPREF['features_point_size'] or 3.)

        self.add_attribute("selection", vartype="int", ndim=1, data=selection)
        self.add_varying("vselection", vartype="int", ndim=1)

        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)

        self.add_attribute("cluster_depth", vartype="int", ndim=1, data=cluster_depth)

        # color map for cluster colors, each spike has an index of the color
        # in the color map
        # ncolors = COLORMAP.shape[0]
        # ncomponents = COLORMAP.shape[1]

        # associate the cluster color to each spike
        # give the correct shape to cmap

        ncolors = COLORMAP_TEXTURE.shape[1]
        ncomponents = COLORMAP_TEXTURE.shape[2]

        global FRAGMENT_SHADER
        fragment = FRAGMENT_SHADER.format(alpha)

        cmap_index = cluster_colors[cluster]
        self.add_texture('cmap', ncomponents=ncomponents, ndim=2, data=COLORMAP_TEXTURE)
        self.add_attribute('cmap_index', ndim=1, vartype='int', data=cmap_index)
        self.add_varying('cmap_vindex', vartype='int', ndim=1)

        dx = 1. / ncolors
        offset = dx / 2.
        dx_shift = 1. / SHIFTLEN
        offset_shift = dx / 2.

        fragment = fragment.replace('%CMAP_OFFSET%', "%.5f" % offset)
        fragment = fragment.replace('%CMAP_STEP%', "%.5f" % dx)

        fragment = fragment.replace('%SHIFT_OFFSET%', "%.5f" % offset_shift)
        fragment = fragment.replace('%SHIFT_STEP%', "%.5f" % dx_shift)
        fragment = fragment.replace('%SHIFTLEN%', "%d" % (SHIFTLEN - 1))



        # necessary so that the navigation shader code is updated
        self.is_position_3D = True

        self.add_vertex_main(VERTEX_SHADER)
        self.add_fragment_main(fragment)


class FeatureBackgroundVisual(Visual):
    def initialize(self, npoints=None,
                    position0=None,
                    alpha=None,
                    ):


        self.primitive_type = 'POINTS'
        self.size = npoints

        self.add_attribute("position0", vartype="float", ndim=2, data=position0)
        self.add_uniform("alpha", vartype="float", ndim=1, data=alpha)

        self.add_uniform("u_point_size", vartype="float", ndim=1, data=USERPREF['features_point_size'] or 3.)

        # necessary so that the navigation shader code is updated
        self.is_position_3D = True

        self.add_vertex_main(VERTEX_SHADER_BACKGROUND)
        self.add_fragment_main(FRAGMENT_SHADER_BACKGROUND)


class FeaturePaintManager(PlotPaintManager):
    def update_points(self):
        self.set_data(position0=self.data_manager.data,
            mask=self.data_manager.masks_full, visual='features')

        self.set_data(position0=self.data_manager.data_background,
            visual='features_background')

    def initialize(self):
        self.toggle_mask_value = False
        self.toggle_background_value = 1

        self.add_visual(FeatureVisual, name='features',
            npoints=self.data_manager.npoints,
            position0=self.data_manager.data,
            mask=self.data_manager.masks_full,
            cluster=self.data_manager.clusters_rel,
            highlight=self.highlight_manager.highlight_mask,
            selection=self.selection_manager.selection_mask,
            cluster_colors=self.data_manager.cluster_colors,
            nclusters=self.data_manager.nclusters,
            cluster_depth=self.data_manager.clusters_full_depth,
            alpha=self.data_manager.alpha_selected,
            )

        self.add_visual(AxesVisual, name='axes')
        self.add_visual(GridVisual, name='grid', visible=False)

        self.add_visual(FeatureBackgroundVisual, name='features_background',
            npoints=self.data_manager.npoints_background,
            position0=self.data_manager.data_background,
            alpha=self.data_manager.alpha_background,
            )

        # Projections.
        self.add_visual(TextVisual, name='projectioninfo_x',
            background_transparent=False,
            fontsize=16,
            is_static=True,
            coordinates=(-1., -1.),
            color=(1.,1.,1.,1.),
            posoffset=(100., 20.),
            text='0:A',
            letter_spacing=300.,
            depth=-1,
            visible=True)
        self.add_visual(TextVisual, name='projectioninfo_y',
            background_transparent=False,
            fontsize=16,
            is_static=True,
            coordinates=(-1., -1.),
            color=(1.,1.,1.,1.),
            posoffset=(50., 80.),
            text=' 0:B',
            letter_spacing=300.,
            depth=-1,
            visible=True)

        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            background_transparent=False,
            coordinates=(1., 1.),
            posoffset=(-120., -30.),
            is_static=True,
            color=(1., 1., 1., 1.),
            letter_spacing=350.,
            depth=-1,
            visible=False)

        # Wizard: target cluster
        self.add_visual(TextVisual, name='wizard_target',
            visible=False,
            background_transparent=False,
            fontsize=18,
            is_static=True,
            coordinates=(-1., 1.),
            color=(1.,) * 4,
            posoffset=(200., -30.),
            text='',
            letter_spacing=500.,
            depth=-1,)
        self.add_visual(TextVisual, name='wizard_candidate',
            visible=False,
            background_transparent=False,
            fontsize=18,
            is_static=True,
            coordinates=(-1., 1.),
            color=(1.,) * 4,
            posoffset=(200., -80.),
            text='',
            letter_spacing=500.,
            depth=-1,)

    def update(self):
        cluster = self.data_manager.clusters_rel
        cluster_colors = self.data_manager.cluster_colors
        cmap_index = cluster_colors[cluster]

        self.set_data(visual='features',
            size=self.data_manager.npoints,
            position0=self.data_manager.data,
            mask=self.data_manager.masks_full,
            highlight=self.highlight_manager.highlight_mask,
            selection=self.selection_manager.selection_mask,
            nclusters=self.data_manager.nclusters,
            cluster_depth=self.data_manager.clusters_full_depth,
            cmap_index=cmap_index,
            alpha=self.data_manager.alpha_selected,
            )

        self.set_data(visual='features_background',
            size=self.data_manager.npoints_background,
            position0=self.data_manager.data_background,
            alpha=self.toggle_background_value * self.data_manager.alpha_background,
            )

    def set_wizard_pair(self, target=None, candidate=None):
        # Display target.
        if target is None:
            self.set_data(visual='wizard_target',
                visible=False)
        else:
            text = 'best unsorted: {0:d}'.format(target[0])
            color = COLORMAP[target[1], :]
            color = np.hstack((color.squeeze(), [1.]))
            self.set_data(visual='wizard_target',
                visible=True,
                text=text,
                color=color)

        # Display candidate.
        if candidate is None or candidate[1] == 0:
            self.set_data(visual='wizard_candidate',
                visible=False)
        else:
            text = 'closest match: {0:d}'.format(candidate[0])
            color = COLORMAP[candidate[1], :]
            color = np.hstack((color.squeeze(), [1.]))
            self.set_data(visual='wizard_candidate',
                visible=True,
                text=text,
                color=color)

        self.updateGL()

    def toggle_mask(self):
        self.toggle_mask_value = 1 - self.toggle_mask_value
        self.set_data(visual='features', toggle_mask=self.toggle_mask_value)

    def toggle_background(self):
        self.toggle_background_value = 1 - self.toggle_background_value
        self.set_data(visual='features_background',
            alpha=self.toggle_background_value * self.data_manager.alpha_background)


# -----------------------------------------------------------------------------
# Highlight/Selection manager
# -----------------------------------------------------------------------------
class FeatureHighlightManager(HighlightManager):
    def initialize(self):
        super(FeatureHighlightManager, self).initialize()
        self.feature_indices = self.data_manager.feature_indices
        self.feature_indices_array = self.data_manager.feature_indices_array
        self.highlight_mask = np.zeros(self.data_manager.nspikes,
            dtype=np.int32)
        self.highlighted_spikes = []
        self.is_highlighting = False

    def find_enclosed_spikes(self, enclosing_box):
        x0, y0, x1, y1 = enclosing_box

        # press_position
        xp, yp = x0, y0

        # reorder
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        features = self.data_manager.data
        masks = self.data_manager.masks_full

        indices = (
                  # (masks > 0) & \
                  (features[:,0] >= xmin) & (features[:,0] <= xmax) & \
                  (features[:,1] >= ymin) & (features[:,1] <= ymax))
        spkindices = np.nonzero(indices)[0]
        spkindices = np.unique(spkindices)
        return spkindices

    def set_highlighted_spikes(self, spikes):
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
                self.highlight_mask[spikes] = 1

        if do_update:
            self.paint_manager.set_data(
                highlight=self.highlight_mask, visual='features')

        self.highlighted_spikes = spikes
        self.is_highlighting = True

    def highlighted(self, box):
        # Get selected spikes (relative indices).
        spikes = self.find_enclosed_spikes(box)
        # Set highlighted spikes.
        self.set_highlighted_spikes(spikes)
        # Emit the HighlightSpikes signal.
        self.emit(spikes)

    def highlight_spikes(self, spikes):
        """spikes in absolute indices."""
        spikes = np.intersect1d(self.data_manager.feature_indices_array,
            spikes)
        if len(spikes) > 0:
            spikes_rel = np.digitize(spikes,
                self.data_manager.feature_indices_array) - 1
        else:
            spikes_rel = []
        self.set_highlighted_spikes(spikes_rel)

    def cancel_highlight(self):
        super(FeatureHighlightManager, self).cancel_highlight()
        self.set_highlighted_spikes(np.array([]))
        self.is_highlighting = False
        self.emit([])

    def emit(self, spikes):
        spikes = np.array(spikes, dtype=np.int32)
        spikes_abs = self.feature_indices[spikes]
        # emit signal
        # log.debug("Highlight {0:d} spikes.".format(len(spikes_abs)))
        self.parent.spikesHighlighted.emit(spikes_abs)


class FeatureSelectionManager(Manager):
    projection = [None, None]

    def polygon(self):
        return self.points[:self.npoints + 2,:]

    def initialize(self):

        self.selection_polygon_color = (1., 1., 1., .5)
        self.points = np.zeros((100, 2))
        self.npoints = 0
        self.is_selection_pending = False
        self.projection = [None, None]

        if not self.paint_manager.get_visual('selection_polygon'):
            self.paint_manager.add_visual(PlotVisual,
                                    position=self.points,
                                    color=self.selection_polygon_color,
                                    primitive_type='LINE_LOOP',
                                    visible=False,
                                    name='selection_polygon')
        self.feature_indices = self.data_manager.feature_full_indices
        self.selection_mask = np.zeros(self.data_manager.nspikes, dtype=np.int32)
        self.selected_spikes = []

    def set_selected_spikes(self, spikes):
        """Update spike colors to mark transiently selected spikes with
        a special color."""
        if len(spikes) == 0:
            # do update only if there were previously selected spikes
            do_update = len(self.selected_spikes) > 0
            self.selection_mask[:] = 0
        else:
            do_update = True
            self.selection_mask[:] = 0
            self.selection_mask[spikes] = 1

        if do_update:
            self.paint_manager.set_data(
                selection=self.selection_mask, visual='features')

        self.selected_spikes = spikes

    def emit(self, spikes):
        spikes = np.array(spikes, dtype=np.int32)
        spikes_abs = self.feature_indices[spikes]
        # emit signal
        # log.debug("Select {0:d} spikes.".format(len(spikes_abs)))
        self.parent.spikesSelected.emit(spikes_abs)

    def find_enclosed_spikes(self, polygon=None):
        """Find the indices of the spikes inside the polygon (in
        transformed coordinates)."""
        if polygon is None:
            polygon = self.polygon()
        features = self.data_manager.data
        features_full = self.data_manager.data_full

        indices = polygon_contains_points(polygon, features)
        spkindices = np.nonzero(indices)[0]
        spkindices = np.unique(spkindices)

        indices_full = polygon_contains_points(polygon, features_full)
        spkindices_full = np.nonzero(indices_full)[0]
        spkindices_full = np.unique(spkindices_full)

        return spkindices, spkindices_full

    def select_spikes(self, polygon=None):
        """Select spikes enclosed in the selection polygon."""
        spikes, spikes_full = self.find_enclosed_spikes(polygon)
        self.set_selected_spikes(spikes)
        self.emit(spikes_full)

    def add_point(self, point):
        """Add a point in the selection polygon."""
        point = self.interaction_manager.get_processor('navigation').get_data_coordinates(*point)
        # Initialize selection.
        if not self.is_selection_pending:
            self.points = np.tile(point, (100, 1))
            self.paint_manager.set_data(
                    visible=True,
                    position=self.points,
                    visual='selection_polygon')
        self.is_selection_pending = True
        self.npoints += 1
        self.points[self.npoints,:] = point

    def point_pending(self, point):
        """A point is currently being positioned by the user. The polygon
        is updated in real time."""
        point = self.interaction_manager.get_processor('navigation').get_data_coordinates(*point)
        if self.is_selection_pending:
            self.points[self.npoints + 1,:] = point
            self.paint_manager.set_data(
                    position=self.points,
                    visual='selection_polygon')
            # select spikes
            self.select_spikes()

    def set_selection_polygon_visibility(self, visible):
        self.paint_manager.set_data(
                visible=visible,
                visual='selection_polygon')

    def end_point(self, point):
        """Terminate selection polygon."""
        # Right click = end selection, next right click = cancel selection.
        if self.is_selection_pending:
            point = self.interaction_manager.get_processor('navigation').get_data_coordinates(*point)
            # record the last point in the selection polygon
            self.points[self.npoints + 1,:] = point
            self.points[self.npoints + 2,:] = self.points[0,:]
            self.paint_manager.set_data(
                    position=self.points,
                    visual='selection_polygon')
            self.select_spikes()
            # record the projection axes corresponding to the current selection
            self.projection = list(self.projection_manager.projection)
            self.is_selection_pending = False
        else:
            self.cancel_selection()

    def cancel_selection(self):
        """Remove the selection polygon."""
        # hide the selection polygon
        # if self.paint_manager.get_visual('selection_polygon').get('visible', None):
        self.set_selected_spikes(np.array([]))
        self.is_selection_pending = False
        self.npoints = 0
        self.points[:] = 0
        self.paint_manager.set_data(visible=False,
            position=self.points,
            visual='selection_polygon')
        self.emit([])


# -----------------------------------------------------------------------------
# Projection
# -----------------------------------------------------------------------------
class FeatureProjectionManager(Manager):
    def initialize(self):
        self.grid_visible = False
        super(FeatureProjectionManager, self).initialize()

    def set_data(self):
        if not hasattr(self, 'projection'):
            self.projection = [None, None]
        self.nchannels = self.data_manager.nchannels
        self.fetdim = self.data_manager.fetdim
        self.nextrafet = self.data_manager.nextrafet
        self.nchannels = self.data_manager.nchannels

    def set_projection(self, coord, channel, feature, do_update=True):
        """Set the projection axes."""
        if channel < 0:
            channel += (self.data_manager.nchannels + self.data_manager.nextrafet)
        if channel < self.nchannels:
            i = channel * self.fetdim + feature
            self.data_manager.masks_full = self.data_manager.masks_array[:,channel]
            text = '{0:d}:{1:s}'.format(channel, 'ABCDEF'[feature])
        # handle extra feature, with channel being directly the feature index
        else:
            i = min(self.nchannels * self.fetdim + self.nextrafet - 1,
                    channel - self.nchannels + self.nchannels * self.fetdim)
            text = 'E{0:d}'.format(channel - self.nchannels)
        self.data_manager.data[:, coord] = self.data_manager.features_array[:, i].ravel()
        self.data_manager.data_full[:, coord] = self.data_manager.features_full_array[:, i].ravel()
        self.data_manager.data_background[:, coord] = \
            self.data_manager.features_background_array[:, i].ravel()

        if do_update:
            self.projection[coord] = (channel, feature)
            # show the selection polygon only if the projection axes correspond
            self.selection_manager.set_selection_polygon_visibility(
              (self.projection[0] == self.selection_manager.projection[0]) & \
              (self.projection[1] == self.selection_manager.projection[1]))

            # Update projection info.
            self.paint_manager.set_data(visual='projectioninfo_' + 'xy'[coord],
                text=text)

            # Show the grid only when time is on the x axis.
            # nav = self.interaction_manager.get_processor('navigation')
            self.grid_visible = (
                self.projection[0][0] == self.nchannels + self.nextrafet - 1)
            self.interaction_manager.activate_grid()
            self.paint_manager.set_data(visual='axes', visible=not(self.grid_visible))

    def reset_projection(self):
        if self.projection[0] is None or self.projection[1] is None:
            self.set_projection(0, 0, 0)#, False)
            self.set_projection(1, 0, 1)
        else:
            self.set_projection(0, self.projection[0][0], self.projection[0][1], False)
            self.set_projection(1, self.projection[1][0], self.projection[1][1])

    def auto_projection(self, target):
        fet = get_array(select(self.data_manager.features,
            self.data_manager.clusters == target))
        n = fet.shape[1]
        fet = np.abs(fet[:,0:n-self.nextrafet:self.fetdim]).mean(axis=0)
        channels_best = np.argsort(fet)[::-1]
        channel0 = channels_best[0]
        channel1 = channels_best[1]
        self.set_projection(0, channel0, 0)
        self.set_projection(1, channel1, 0)
        self.parent.projectionChanged.emit(0, channel0, 0)
        self.parent.projectionChanged.emit(1, channel1, 0)

    def select_neighbor_channel(self, coord, channel_dir, feature=None):
        # current channel and feature in the given coordinate
        proj = self.projection[coord]
        if proj is None:
            proj = (0, coord)
        channel, _ = proj
        # next or previous channel
        channel = np.mod(channel + channel_dir, self.data_manager.nchannels +
            self.data_manager.nextrafet)
        if feature is None:
            feature = self.get_smart_feature(coord, channel)
        self.set_projection(coord, channel, feature, do_update=True)

    def select_neighbor_projection(self, coord, channel_dir):
        # current channel and feature in the given coordinate
        proj = self.projection[coord]
        if proj is None:
            proj = (0, coord)
        channel, feature = proj
        lf = self.fetdim - 1
        if feature == 0 and channel_dir == -1:
            self.select_neighbor_channel(coord, channel_dir, feature=lf)
        elif feature == lf and channel_dir == 1:
            self.select_neighbor_channel(coord, channel_dir, feature=0)
        else:
            feature = feature + channel_dir
            self.set_projection(coord, channel, feature, do_update=True)

    def select_feature(self, coord, feature):
        # feature = np.clip(feature, 0, s - 1)
        # current channel and feature in the given coordinate
        proj = self.projection[coord]
        if proj is None:
            proj = (0, coord)
        channel, _ = proj
        self.set_projection(coord, channel, feature, do_update=True)

    def get_projection(self, coord):
        return self.projection[coord]

    def get_smart_feature(self, coord, channel):
        """Choose the best feature according to the current projections."""
        ch0, fet0 = self.projection[coord]
        ch1, fet1 = self.projection[1 - coord]
        if channel == ch1:
            return (1, 0, 0)[fet1]
        else:
            return 0


# -----------------------------------------------------------------------------
# Interaction
# -----------------------------------------------------------------------------
class FeatureInfoManager(Manager):
    def show_closest_cluster(self, xd, yd, zx=1, zy=1):
        # find closest spike
        dist = (np.abs(self.data_manager.data[:, 0] - xd) * zx +
                np.abs(self.data_manager.data[:, 1] - yd) * zy)
        ispk = dist.argmin()
        cluster = self.data_manager.clusters_rel[ispk]

        # Absolute spike index.
        ispk_abs = self.data_manager.feature_indices[ispk]
        # time = select(self.data_manager.features, ispk_abs)[-1]
        # time = (time + 1) * .5 * self.parent.data_manager.duration
        time = self.data_manager.spiketimes[ispk_abs]

        unit = self.data_manager.time_unit
        if unit == 'second':
            text = "{0:d}, {1:.5f}".format(
                self.data_manager.clusters_unique[cluster],
                time)
        else:
            text = "{0:d}, {1:d}".format(
                self.data_manager.clusters_unique[cluster],
                int(time * self.data_manager.freq))

        self.paint_manager.set_data(
            text=text,
            visible=True,
            visual='clusterinfo')


class FeatureInteractionManager(PlotInteractionManager):
    def initialize(self):
        self.constrain_navigation = False

        self.register(None, self.none_callback)
        self.register('HighlightSpike', self.highlight_spike)
        self.register('SelectionPointPending', self.selection_point_pending)
        self.register('AddSelectionPoint', self.selection_add_point)
        self.register('EndSelectionPoint', self.selection_end_point)
        self.register('SelectProjection', self.select_projection)
        self.register('ToggleMask', self.toggle_mask)
        self.register('ToggleBackground', self.toggle_background)
        self.register('SelectNeighborChannel', self.select_neighbor_channel)
        self.register('SelectNeighborProjection', self.select_neighbor_projection)
        self.register('SelectFeature', self.select_feature)

    def initialize_default(self, constrain_navigation=None,
        momentum=True,
        ):
        super(PlotInteractionManager, self).initialize_default()
        self.add_processor(NavigationEventProcessor,
            constrain_navigation=constrain_navigation,
            momentum=momentum,
            name='navigation')
        self.add_processor(GridEventProcessor, name='grid')


    # Grid
    # ----
    def activate_grid(self):
        visible = self.projection_manager.grid_visible
        self.paint_manager.set_data(visual='grid_lines',
            visible=visible)
        self.paint_manager.set_data(visual='grid_text', visible=visible)
        processor = self.get_processor('grid')
        if processor:
            processor.activate(visible)
            if visible:
                processor.update_axes(None)


    # Highlighting
    # ------------
    def none_callback(self, parameter):
        if self.highlight_manager.is_highlighting:
            self.highlight_manager.cancel_highlight()
        self.paint_manager.set_data(visible=False, visual='clusterinfo')

    def highlight_spike(self, parameter):
        self.highlight_manager.highlight(parameter)
        self.cursor = 'CrossCursor'


    # Selection
    # ---------
    def selection_point_pending(self, parameter):
        self.show_closest_cluster(parameter)
        self.selection_manager.point_pending(parameter)

    def selection_add_point(self, parameter):
        self.selection_manager.add_point(parameter)

    def selection_end_point(self, parameter):
        self.selection_manager.end_point(parameter)


    # Projection
    # ----------
    def select_neighbor_channel(self, parameter):
        coord, channel_dir = parameter

        self.projection_manager.select_neighbor_channel(coord, channel_dir)

        channel, feature = self.projection_manager.get_projection(coord)

        log.debug(("Projection changed to channel {0:d} and "
                   "feature {1:d} on axis {2:s}.").format(
                        channel, feature, 'xy'[coord]))
        self.parent.projectionChanged.emit(coord, channel, feature)

        self.paint_manager.update_points()
        self.paint_manager.updateGL()

    def select_neighbor_projection(self, parameter):
        coord, channel_dir = parameter

        self.projection_manager.select_neighbor_projection(coord, channel_dir)

        channel, feature = self.projection_manager.get_projection(coord)

        log.debug(("Projection changed to channel {0:d} and "
                   "feature {1:d} on axis {2:s}.").format(
                        channel, feature, 'xy'[coord]))
        self.parent.projectionChanged.emit(coord, channel, feature)

        self.paint_manager.update_points()
        self.paint_manager.updateGL()

    def select_feature(self, parameter):
        coord, feature = parameter

        if feature < 0 or feature >= self.data_manager.fetdim:
            return

        self.projection_manager.select_feature(coord, feature)
        channel, feature = self.projection_manager.get_projection(coord)

        log.debug(("Projection changed to channel {0:d} and "
                   "feature {1:d} on axis {2:s}.").format(
                        channel, feature, 'xy'[coord]))
        self.parent.projectionChanged.emit(coord, channel, feature)

        self.paint_manager.update_points()
        self.paint_manager.updateGL()

    def select_projection(self, parameter):
        """Select a projection for the given coordinate."""
        coord, _, _ = parameter
        self.projection_manager.set_projection(*parameter)  # coord, channel, feature
        channel, feature = self.projection_manager.get_projection(coord)
        self.parent.projectionChanged.emit(coord, channel, feature)
        self.paint_manager.update_points()
        self.paint_manager.updateGL()


    # Misc
    # ----
    def toggle_mask(self, parameter=None):
        self.paint_manager.toggle_mask()

    def toggle_background(self, parameter=None):
        self.paint_manager.toggle_background()

    def show_closest_cluster(self, parameter):

        self.cursor = None

        nav = self.get_processor('navigation')
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        zx, zy = nav.get_scaling()

        if self.data_manager.data.size == 0:
            return

        self.info_manager.show_closest_cluster(xd, yd, zx, zy)


class FeatureBindings(KlustaViewaBindings):
    def set_zoombox_keyboard(self):
        """Set zoombox bindings with the keyboard."""
        self.set('MiddleClickMove', 'ZoomBox',
                    # key_modifier='Shift',
                    param_getter=lambda p: (p["mouse_press_position"][0],
                                            p["mouse_press_position"][1],
                                            p["mouse_position"][0],
                                            p["mouse_position"][1]))

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

    def set_toggle_mask(self):
        self.set('KeyPress',
                 'ToggleMask',
                 key='T')

    def set_toggle_background(self):
        self.set('KeyPress',
                 'ToggleBackground',
                 key='B')

    def set_neighbor_channel(self):
        # select previous/next channel for coordinate 0
        self.set('Wheel', 'SelectNeighborChannel',
                 key_modifier='Control',
                 param_getter=lambda p: (0, -int(np.sign(p['wheel']))))

        # select previous/next channel for coordinate 1
        self.set('Wheel', 'SelectNeighborChannel',
                 key_modifier='Shift',
                 param_getter=lambda p: (1, -int(np.sign(p['wheel']))))

        self.set('KeyPress', 'SelectNeighborChannel',
                 key_modifier='Control',
                 key='Up',
                 param_getter=lambda p: (0, -1))

        self.set('KeyPress', 'SelectNeighborChannel',
                 key_modifier='Control',
                 key='Down',
                 param_getter=lambda p: (0, 1))

        self.set('KeyPress', 'SelectNeighborChannel',
                 key_modifier='Shift',
                 key='Up',
                 param_getter=lambda p: (1, -1))

        self.set('KeyPress', 'SelectNeighborChannel',
                 key_modifier='Shift',
                 key='Down',
                 param_getter=lambda p: (1, 1))

    def set_neighbor_projection(self):
        # select previous/next channel for coordinate 0
        self.set('Wheel', 'SelectNeighborProjection',
                 key_modifier='Alt',
                 param_getter=lambda p: (1, -int(np.sign(p['wheel']))))

    def set_time_channel(self):
        self.set('KeyPress', 'SelectProjection',
                    key='T', key_modifier='Control',
                    description='Time on X',
                    param_getter=(0, -1, 0))
        self.set('KeyPress', 'SelectProjection',
                    key='T', key_modifier='Shift',
                    description='Time on Y',
                    param_getter=(1, -1, 0))

    def set_feature(self):
        # select projection feature for coordinate 0
        for feature in xrange(4):
            self.set('KeyPress', 'SelectFeature',
                     key='F{0:d}'.format(feature+1), description='X',
                     key_modifier='Control',
                     param_getter=(0, feature))
            self.set('KeyPress', 'SelectFeature',
                     key='F{0:d}'.format(feature+1), description='Y',
                     key_modifier='Shift',
                     param_getter=(1, feature))

    def set_selection(self):
        # selection
        self.set('Move',
                 'SelectionPointPending',
                 # key_modifier='Control',
                 param_getter=lambda p: (p["mouse_position"][0],
                                         p["mouse_position"][1],))
        self.set('RightClick',
                 'AddSelectionPoint',
                 # key_modifier='Control',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
        self.set('LeftClick',
                 'EndSelectionPoint',
                 # key_modifier='Control',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))

    def initialize(self):
        self.set_highlight()
        self.set_toggle_mask()
        self.set_toggle_background()
        self.set_neighbor_channel()
        self.set_neighbor_projection()
        self.set_feature()
        self.set_time_channel()
        self.set_selection()


# -----------------------------------------------------------------------------
# Top-level widget
# -----------------------------------------------------------------------------
class FeatureView(KlustaView):
    # Raise the list of highlighted spike absolute indices.
    spikesHighlighted = QtCore.pyqtSignal(np.ndarray)
    spikesSelected = QtCore.pyqtSignal(np.ndarray)
    projectionChanged = QtCore.pyqtSignal(int, int, int)

    # Initialization
    # --------------
    def initialize(self):
        self.activate3D = True
        self.activate_grid = False
        self.set_bindings(FeatureBindings)
        self.set_companion_classes(
                paint_manager=FeaturePaintManager,
                data_manager=FeatureDataManager,
                projection_manager=FeatureProjectionManager,
                info_manager=FeatureInfoManager,
                highlight_manager=FeatureHighlightManager,
                selection_manager=FeatureSelectionManager,
                interaction_manager=FeatureInteractionManager)

    def set_data(self, *args, **kwargs):
        # if kwargs.get('clusters_selected', None) is None:
            # return
        self.data_manager.set_data(*args, **kwargs)

        # update?
        if self.initialized:
            self.paint_manager.update()
            self.updateGL()


    # Public methods
    # --------------
    def set_wizard_pair(self, target=None, color=None):
        self.paint_manager.set_wizard_pair(target, color)

    def highlight_spikes(self, spikes):
        self.highlight_manager.highlight_spikes(spikes)
        self.updateGL()

    def select_spikes(self, spikes):
        pass

    def toggle_mask(self):
        self.interaction_manager.toggle_mask()
        self.updateGL()

    def set_projection(self, coord, channel, feature, do_emit=True):
        if feature == -1:
            feature = self.projection_manager.get_smart_feature(coord, channel)
        log.debug(("Set projection on channel {0:d}, feature {1:d} "
                   "on coord {2:s}".format(channel, feature, 'xy'[coord])))
        self.projection_manager.set_projection(coord, channel, feature)
        if do_emit:
            self.projectionChanged.emit(coord, channel, feature)
        self.paint_manager.update_points()
        self.paint_manager.updateGL()

    def sizeHint(self):
        return QtCore.QSize(400, 2000)

    def maximumSize(self):
        return QtCore.QSize(2000, 2000)


