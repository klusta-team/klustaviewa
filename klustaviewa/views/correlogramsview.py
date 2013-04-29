"""Correlograms View: show auto- and cross- correlograms of the clusters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import numpy.random as rdn
from galry import (Manager, PlotPaintManager, PlotInteractionManager, Visual,
    GalryWidget, QtGui, QtCore, QtOpenGL, enforce_dtype, RectanglesVisual,
    TextVisual, PlotVisual, AxesVisual)
    
from klustaviewa.stats.cache import IndexedMatrix
from klustaviewa.io.tools import get_array
from klustaviewa.utils.colors import COLORMAP
from klustaviewa.views.common import HighlightManager, KlustaViewaBindings
from klustaviewa.views.widgets import VisualizationWidget


# -----------------------------------------------------------------------------
# Shaders
# -----------------------------------------------------------------------------
VERTEX_SHADER = """
    //vec3 color = vec3(1, 1, 1);

    float margin = 0.05;
    float a = 1.0 / (nclusters * (1 + 2 * margin));
    
    vec2 box_position = vec2(0, 0);
    box_position.x = -1 + a * (1 + 2 * margin) * (2 * cluster.x + 1);
    box_position.y = -1 + a * (1 + 2 * margin) * (2 * cluster.y + 1);
    
    vec2 transformed_position = position;
    transformed_position.y = 2 * transformed_position.y - 1;
    transformed_position = box_position + a * transformed_position;
"""


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_histogram_points(hist):
    """Tesselates correlograms.
    
    Arguments:
      * hist: a N x Nsamples array, where each line contains an histogram.
      
    Returns:
      * X, Y: two N x (5*Nsamples+1) arrays with the coordinates of the
        correlograms, a
      
    """
    if hist.size == 0:
        return np.array([[]]), np.array([[]])
    n, nsamples = hist.shape
    dx = 2. / nsamples
    
    x0 = -1 + dx * np.arange(nsamples)
    
    x = np.zeros((n, 5 * nsamples + 1))
    # y = -np.ones((n, 5 * nsamples + 1))
    y = np.zeros((n, 5 * nsamples + 1))
    
    x[:,0:-1:5] = x0
    x[:,1::5] = x0
    x[:,2::5] = x0 + dx
    x[:,3::5] = x0
    x[:,4::5] = x0 + dx
    x[:,-1] = 1
    
    y[:,1::5] = hist
    y[:,2::5] = hist
    
    return x, y

    
# -----------------------------------------------------------------------------
# Data manager
# -----------------------------------------------------------------------------
class CorrelogramsDataManager(Manager):
    def set_data(self, correlograms=None, cluster_colors=None, baselines=None,
        clusters_selected=None):
        
        if correlograms is None:
            correlograms = IndexedMatrix(shape=(0, 0, 0))
            cluster_colors = np.zeros(0)
            clusters_selected = []
            ncorrbins = 0            
        
        # self.correlograms_array = get_correlograms_array(correlograms,
            # clusters_selected=clusters_selected, ncorrbins=ncorrbins)
        self.correlograms = correlograms
        # self.indices = self.correlograms.indices
        self.correlograms_array = correlograms.to_array()
        nclusters, nclusters, self.nbins = self.correlograms_array.shape
        self.ncorrelograms = nclusters * nclusters
        self.clusters_selected = clusters_selected
        self.clusters_unique = sorted(clusters_selected)
        self.nclusters = len(clusters_selected)
        assert nclusters == self.nclusters
        self.cluster_colors_array = get_array(cluster_colors)
        
        # HACK: if correlograms is empty, ncorrelograms == 1 here!
        if self.correlograms_array.size == 0:
            self.ncorrelograms = 0
        
        # cluster i and j for each histogram in the view
        clusters = [(i,j) for i in xrange(self.nclusters) for j in xrange(self.nclusters)]
        self.clusters = np.array(clusters, dtype=np.int32)
        
        # normalization
        for i in xrange(self.nclusters):
            # # correlograms in a given row
            # ind = self.clusters[:,1] == j
            # # index of the (i,j) histogram
            # i0 = np.nonzero((self.clusters[:,0] == self.clusters[:,1]) & 
                # (self.clusters[:,0] == j))[0][0]
            correlogram_diagonal = self.correlograms_array[i, i, ...]
            # divide all correlograms in the row by the max of this histogram
            m = correlogram_diagonal.max()
            if m > 0:
                self.correlograms_array[i,:,:] /= m
            # normalize all correlograms in the row so that they all fit in the 
            # window
            m = self.correlograms_array[i,:,:].max()
            if m > 0:
                self.correlograms_array[i,:,:] /= m
        
        self.nprimitives = self.ncorrelograms
        # index 0 = heterogeneous clusters, index>0 ==> cluster index + 1
        self.cluster_colors = get_array(cluster_colors)
        
        # get the vertex positions
        X, Y = get_histogram_points(self.correlograms_array.reshape(
            (self.ncorrelograms, self.nbins)))
        n = X.size
        self.nsamples = X.shape[1]
        
        # fill the data array
        self.position = np.empty((n, 2), dtype=np.float32)
        self.position[:,0] = X.ravel()
        self.position[:,1] = Y.ravel()
    
        # baselines of the correlograms
        self.baselines = baselines
        
        # indices of correlograms on the diagonal
        if self.nclusters:
            identity = self.clusters[:,0] == self.clusters[:,1]
        else:
            identity = []
            
        color_array_index = np.zeros(self.ncorrelograms, dtype=np.int32)
        
        color_array_index[identity] = np.array(self.cluster_colors_array + 1, 
            dtype=np.int32)
        # very first color in color map = white (cross-correlograms)
        self.color = np.vstack((np.ones((1, 3)), COLORMAP))
        self.color_array_index = color_array_index
        
        self.clusters = np.repeat(self.clusters, self.nsamples, axis=0)
        self.color_array_index = np.repeat(self.color_array_index, self.nsamples, axis=0)
        
     
# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class CorrelogramsVisual(PlotVisual):
    def initialize(self, nclusters=None, ncorrelograms=None, #nsamples=None,
        position=None, color=None, color_array_index=None, clusters=None):
        
        self.position_attribute_name = "transformed_position"
        
        super(CorrelogramsVisual, self).initialize(
            position=position,
            nprimitives=ncorrelograms,
            color=color,
            color_array_index=color_array_index,
            autonormalizable=False,
            )
            
        self.primitive_type = 'TRIANGLE_STRIP'
        
        self.add_attribute("cluster", vartype="int", ndim=2, data=clusters)
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        
        self.add_vertex_main(VERTEX_SHADER)

        
class CorrelogramsBaselineVisual(PlotVisual):
    def initialize(self, nclusters=None, baselines=None, clusters=None):
        
        self.position_attribute_name = "transformed_position"
        
        # texture = np.ones((10, 10, 3))
        
        n = len(baselines)
        position = np.zeros((2 * n, 2))
        position[:,0] = np.tile(np.array([-1., 1.]), n)
        position[:,1] = np.repeat(baselines, 2)
        # position[n:,1] = baselines
        
        clusters = np.repeat(clusters, 2, axis=0)
        
        self.primitive_type = 'LINES'
        
        super(CorrelogramsBaselineVisual, self).initialize(
            position=position,
            # texture=texture
            )
            
        self.add_attribute("cluster", vartype="int", ndim=2, data=clusters)
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        
        self.add_vertex_main(VERTEX_SHADER)
        
        
class CorrelogramsPaintManager(PlotPaintManager):
    def initialize(self, **kwargs):
        self.add_visual(CorrelogramsVisual,
            nclusters=self.data_manager.nclusters,
            ncorrelograms=self.data_manager.ncorrelograms,
            position=self.data_manager.position,
            color=self.data_manager.color,
            color_array_index=self.data_manager.color_array_index,
            clusters=self.data_manager.clusters,
            name='correlograms')
            
        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            posoffset=(50., -50.),
            background_transparent=False,
            letter_spacing=350.,
            depth=-1,
            visible=False)
        
    def update(self):
        self.reinitialize_visual(
            size=self.data_manager.position.shape[0],
            nclusters=self.data_manager.nclusters,
            ncorrelograms=self.data_manager.ncorrelograms,
            position=self.data_manager.position,
            color=self.data_manager.color,
            color_array_index=self.data_manager.color_array_index,
            clusters=self.data_manager.clusters,
            visual='correlograms')
            

# -----------------------------------------------------------------------------
# Interaction
# -----------------------------------------------------------------------------
class CorrelogramsInfoManager(Manager):
    def initialize(self):
        pass
        
    def show_closest_cluster(self, xd, yd):
        
        margin = 0.05
        a = 1.0 / (self.data_manager.nclusters * (1 + 2 * margin))
        
        cx = int(((xd + 1) / (a * (1 + 2 * margin)) - 1) / 2 + .5)
        cy = int(((yd + 1) / (a * (1 + 2 * margin)) - 1) / 2 + .5)
        
        cx_rel = np.clip(cx, 0, self.data_manager.nclusters - 1)
        cy_rel = np.clip(cy, 0, self.data_manager.nclusters - 1)
        
        color1 = self.data_manager.cluster_colors[cy_rel]
        r, g, b = COLORMAP[color1,:]
        color1 = (r, g, b, .75)
        
        cx = self.data_manager.clusters_unique[cx_rel]
        cy = self.data_manager.clusters_unique[cy_rel]
        
        text = "%d / %d" % (cx, cy)
        
        self.paint_manager.set_data(coordinates=(xd, yd), #color=color1,
            text=text,
            visible=True,
            visual='clusterinfo')
        
    
class CorrelogramsInteractionManager(PlotInteractionManager):
    def initialize(self):
        self.register('ShowClosestCluster', self.show_closest_cluster)
        self.register(None, self.hide_closest_cluster)
            
    def hide_closest_cluster(self, parameter):
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        
    def show_closest_cluster(self, parameter):
        
        if self.data_manager.nclusters == 0:
            return
            
        self.cursor = None
        
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        self.info_manager.show_closest_cluster(xd, yd)
        
        
class CorrelogramsBindings(KlustaViewaBindings):
    def set_zoombox_keyboard(self):
        """Set zoombox bindings with the keyboard."""
        self.set('LeftClickMove', 'ZoomBox',
                    key_modifier='Shift',
                    param_getter=lambda p: (p["mouse_press_position"][0],
                                            p["mouse_press_position"][1],
                                            p["mouse_position"][0],
                                            p["mouse_position"][1]))

    def set_clusterinfo(self):
        self.set('Move', 'ShowClosestCluster', key_modifier='Shift',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
    
    def initialize(self):
        super(CorrelogramsBindings, self).initialize()
        self.set_clusterinfo()
    
    
# -----------------------------------------------------------------------------
# Top-level widget
# -----------------------------------------------------------------------------
class CorrelogramsView(GalryWidget):
    def __init__(self, *args, **kwargs):
        # Activate antialiasing.
        format = QtOpenGL.QGLFormat()
        format.setSampleBuffers(True)
        kwargs['format'] = format
        super(CorrelogramsView, self).__init__(**kwargs)

    def initialize(self):
        self.set_bindings(CorrelogramsBindings)
        self.set_companion_classes(paint_manager=CorrelogramsPaintManager,
            interaction_manager=CorrelogramsInteractionManager,
            info_manager=CorrelogramsInfoManager,
            data_manager=CorrelogramsDataManager,)
    
    def set_data(self, *args, **kwargs):
        # if kwargs.get('clusters_selected', None) is None:
            # return
        self.data_manager.set_data(*args, **kwargs)
        
        # update?
        if self.initialized:
            self.paint_manager.update()
            self.updateGL()

            
            
    def sizeHint(self):
        return QtCore.QSize(400, 400)
    