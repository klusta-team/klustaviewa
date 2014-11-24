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
from kwiklib.dataio.tools import get_array
from kwiklib.utils.colors import COLORMAP
from klustaviewa.views.common import HighlightManager, KlustaViewaBindings, KlustaView


# -----------------------------------------------------------------------------
# Shaders
# -----------------------------------------------------------------------------
VERTEX_SHADER = """
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
        clusters_selected=None, ncorrbins=None, corrbin=None,
        keep_order=None,
        normalization='row'):
        
        if correlograms is None:
            correlograms = IndexedMatrix(shape=(0, 0, 0))
            baselines = np.zeros(0)
            cluster_colors = np.zeros(0)
            clusters_selected = []
            ncorrbins = 0            
            corrbin = 0            
        
        self.keep_order = keep_order
        
        # self.correlograms_array = get_correlograms_array(correlograms,
            # clusters_selected=clusters_selected, ncorrbins=ncorrbins)
        self.correlograms = correlograms
        
        # Copy the original arrays for normalization.
        self.baselines = baselines
        self.baselines0 = baselines.copy()
        
        self.correlograms_array = correlograms.to_array()
        self.correlograms_array0 = self.correlograms_array.copy()
        
        nclusters, nclusters, self.nbins = self.correlograms_array.shape
        self.ncorrelograms = nclusters * nclusters
        self.nticks = (ncorrbins + 1) * self.ncorrelograms
        self.ncorrbins = ncorrbins
        self.corrbin = corrbin
        self.clusters_selected = np.array(clusters_selected, dtype=np.int32)
        self.clusters_unique = np.array(sorted(clusters_selected), dtype=np.int32)
        self.nclusters = len(clusters_selected)
        assert nclusters == self.nclusters
        self.cluster_colors = cluster_colors
        self.cluster_colors_array = get_array(cluster_colors, dosort=True)
        
        if keep_order:
            self.permutation = np.argsort(clusters_selected)
        else:
            self.permutation = np.arange(self.nclusters)
        self.cluster_colors_array_ordered = self.cluster_colors_array[self.permutation]
        
        # HACK: if correlograms is empty, ncorrelograms == 1 here!
        if self.correlograms_array.size == 0:
            self.ncorrelograms = 0
        
        # cluster i and j for each histogram in the view
        clusters = [(i,j) for j in self.permutation
                            for i in self.permutation]
        self.clusters = np.array(clusters, dtype=np.int32)
        self.clusters0 = self.clusters
        
        # baselines of the correlograms
        
        self.nprimitives = self.ncorrelograms
        # index 0 = heterogeneous clusters, index>0 ==> cluster index + 1
        # self.cluster_colors = get_array(cluster_colors)
        
        # normalize and update the data position
        self.normalize(normalization)
    
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
        self.color_array_index = np.repeat(self.color_array_index,
            self.nsamples, axis=0)
        
    def normalize(self, normalization='row'):
        self.correlograms_array = self.correlograms_array0.copy()
        self.baselines = self.baselines0.copy()
        if normalization == 'row':
            # normalization
            for i in range(self.nclusters):
                # divide all correlograms in the row by the max of this histogram
                correlogram_diagonal = self.correlograms_array[i, i, ...]
                m = correlogram_diagonal.max()
                if m > 0:
                    self.correlograms_array[i,:,:] /= m
                    self.baselines[i,:] /= m
                # normalize all correlograms in the row so that they all fit in the 
                # window
                m = self.correlograms_array[i,:,:].max()
                if m > 0:
                    self.correlograms_array[i,:,:] /= m
                    self.baselines[i,:] /= m
        elif normalization == 'uniform':
            M = self.correlograms_array.max(axis=2)
            self.correlograms_array /= M.reshape(
                (self.nclusters, self.nclusters, 1))
            self.baselines /= M
    
        # get the vertex positions
        X, Y = get_histogram_points(self.correlograms_array.reshape(
            (self.ncorrelograms, self.nbins)))
        n = X.size
        self.nsamples = X.shape[1]
        
        # fill the data array
        self.position = np.empty((n, 2), dtype=np.float32)
        self.position[:,0] = X.ravel()
        self.position[:,1] = Y.ravel()
        
     
# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class CorrelogramsVisual(PlotVisual):
    def initialize(self, nclusters=None, ncorrelograms=None, #nsamples=None,
        ncorrbins=None, corrbin=None,
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
    def initialize(self, nclusters=None, baselines=None, clusters=None,
        corrbin=None):
        
        self.position_attribute_name = "transformed_position"
        
        if baselines is None:
            baselines = np.zeros((nclusters, nclusters))
        
        baselines = baselines.ravel()
        
        n = len(baselines)
        position = np.zeros((2 * n, 2))
        position[:,0] = np.tile(np.array([-1., 1.]), n)
        position[:,1] = np.repeat(baselines, 2)
        position = np.array(position, dtype=np.float32)
        
        clusters = np.repeat(clusters, 2, axis=0)
        
        super(CorrelogramsBaselineVisual, self).initialize(
            position=position,
            nprimitives=n,
            color=(.25, .25, .25, 1.),
            autonormalizable=False,
            )
            
        self.primitive_type = 'LINES'
        
        self.add_attribute("cluster", vartype="int", ndim=2, data=clusters)
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        
        self.add_vertex_main(VERTEX_SHADER)
        
        
class CorrelogramsTicksVisual(PlotVisual):
    def initialize(self, ncorrbins=None, corrbin=None, ncorrelograms=None,
        clusters=None, nclusters=None):
        
        if ncorrbins is None:
            ncorrbins = 0
        if corrbin is None:
            corrbin = 0
        
        ncorrbins += 1
        
        self.position_attribute_name = "transformed_position"
        
        nticks = ncorrbins * ncorrelograms
        # n = 2 * nticks
        position = np.zeros((2 * ncorrbins, 2))
        position[:, 0] = np.repeat(np.linspace(-1., 1., ncorrbins), 2)
        position[1::2, 1] = 0.05
        position = np.array(position, dtype=np.float32)
        
        clusters = np.repeat(clusters, 2 * ncorrbins, axis=0)
        
        color = .25 * np.ones((ncorrbins, 4))
        if ncorrbins % 2 == 1:
            color[ncorrbins // 2, 3] = .85
            position[ncorrbins, 1] = 1
        else:
            color[ncorrbins // 2, 3] = .85
            color[ncorrbins // 2 - 1, 3] = .85
            position[ncorrbins, 1] = 1
            position[ncorrbins, 1] = 1
            
        color = np.repeat(color, 2, axis=0)
        color = np.tile(color, (ncorrelograms, 1))
        position = np.tile(position, (ncorrelograms, 1))
        
        super(CorrelogramsTicksVisual, self).initialize(
            position=position,
            nprimitives=nticks,
            color=color,
            autonormalizable=False,
            )
            
        self.primitive_type = 'LINES'
        
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
            ncorrbins=self.data_manager.ncorrbins,
            corrbin=self.data_manager.corrbin,
            name='correlograms')
            
        self.add_visual(CorrelogramsBaselineVisual,
            baselines=self.data_manager.baselines,
            nclusters=self.data_manager.nclusters,
            clusters=self.data_manager.clusters0,
            name='baselines',
            )
            
        self.add_visual(CorrelogramsTicksVisual,
            ncorrbins=self.data_manager.ncorrbins,
            corrbin=self.data_manager.corrbin,
            nclusters=self.data_manager.nclusters,
            ncorrelograms=self.data_manager.ncorrelograms,
            clusters=self.data_manager.clusters0,
            name='ticks',
            )
            
        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            # posoffset=(50., -50.),
            coordinates=(1., -1.),
            posoffset=(-80., 30.),
            is_static=True,
            color=(1., 1., 1., 1.),
            background_transparent=False,
            letter_spacing=350.,
            depth=-1,
            visible=False)
        
    def update(self):
        self.reinitialize_visual(
            # size=self.data_manager.position.shape[0],
            nclusters=self.data_manager.nclusters,
            ncorrelograms=self.data_manager.ncorrelograms,
            position=self.data_manager.position,
            color=self.data_manager.color,
            color_array_index=self.data_manager.color_array_index,
            clusters=self.data_manager.clusters,
            ncorrbins=self.data_manager.ncorrbins,
            corrbin=self.data_manager.corrbin,
            visual='correlograms')
            
        self.reinitialize_visual(
            # size=2 * self.data_manager.baselines.size,
            baselines=self.data_manager.baselines,
            nclusters=self.data_manager.nclusters,
            clusters=self.data_manager.clusters0,
            visual='baselines')
            
        self.reinitialize_visual(
            # size=2 * self.data_manager.nticks,
            ncorrbins=self.data_manager.ncorrbins,
            corrbin=self.data_manager.corrbin,
            nclusters=self.data_manager.nclusters,
            ncorrelograms=self.data_manager.ncorrelograms,
            clusters=self.data_manager.clusters0,
            visual='ticks')
            

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

        color1 = self.data_manager.cluster_colors_array_ordered[cy_rel]
        r, g, b = COLORMAP[color1,:]
        color1 = (r, g, b, .75)
        
        cx = self.data_manager.clusters_unique[self.data_manager.permutation][cx_rel]
        cy = self.data_manager.clusters_unique[self.data_manager.permutation][cy_rel]
        
        text = "%d / %d" % (cx, cy)
        
        self.paint_manager.set_data(#coordinates=(xd, yd), #color=color1,
            text=text,
            visible=True,
            visual='clusterinfo')
        
    
class CorrelogramsInteractionManager(PlotInteractionManager):
    def initialize(self):
        self.normalization_index = 0
        self.normalization_list = ['row', 'uniform']
        
        self.register('ShowClosestCluster', self.show_closest_cluster)
        self.register('ChangeNormalization', self.change_normalization)
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
        
    def change_normalization(self, normalization=None):
        if normalization is None:
            self.normalization_index = np.mod(self.normalization_index + 1,
                len(self.normalization_list))
            normalization = self.normalization_list[self.normalization_index]
        self.data_manager.normalize(normalization)
        self.paint_manager.update()
        self.parent.updateGL()
    
        
class CorrelogramsBindings(KlustaViewaBindings):
    def set_normalization(self):
        self.set('KeyPress', 'ChangeNormalization', key='N')

    def set_clusterinfo(self):
        self.set('Move', 'ShowClosestCluster', #key_modifier='Shift',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
    
    def initialize(self):
        super(CorrelogramsBindings, self).initialize()
        self.set_clusterinfo()
        self.set_normalization()
    
    
# -----------------------------------------------------------------------------
# Top-level widget
# -----------------------------------------------------------------------------
class CorrelogramsView(KlustaView):
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
        kwargs['normalization'] = self.interaction_manager.normalization_list[
            self.interaction_manager.normalization_index]
        self.data_manager.set_data(*args, **kwargs)
        
        # update?
        if self.initialized:
            self.paint_manager.set_data(visible=True, visual='correlograms')
            self.paint_manager.set_data(visible=True, visual='baselines')
            self.paint_manager.set_data(visible=True, visual='ticks')
            self.paint_manager.update()
            self.updateGL()

    def clear(self):
        self.paint_manager.set_data(visible=False, visual='correlograms')
        self.paint_manager.set_data(visible=False, visual='baselines')
        self.paint_manager.set_data(visible=False, visual='ticks')
            
    def change_normalization(self, normalization=None):
        self.interaction_manager.change_normalization(normalization)
            
