"""Correlation matrix View: show correlation matrix."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import numpy.random as rdn
from galry import (Manager, DefaultPaintManager, PlotInteractionManager,
    Visual, PlotPaintManager,
    GalryWidget, QtGui, QtCore, QtOpenGL, enforce_dtype, RectanglesVisual,
    TextVisual, TextureVisual)
from galry.tools import hsv_to_rgb
    
from kwiklib.dataio.selection import get_indices
from kwiklib.dataio.tools import get_array
from kwiklib.utils.colors import COLORMAP
from kwiklib.utils import logger as log
from klustaviewa.views.common import HighlightManager, KlustaViewaBindings, KlustaView


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def colormap(x, col0=None, col1=None):
    """Colorize a 2D grayscale array.
    
    Arguments: 
      * x:an NxM array with values in [0,1].
      * col0=None: a tuple (H, S, V) corresponding to color 0. By default, a
        rainbow color gradient is used.
      * col1=None: a tuple (H, S, V) corresponding to color 1.
    
    Returns:
      * y: an NxMx3 array with a rainbow color palette.
    
    """
    # record values to be removed
    removed = x == -1
    
    x[np.isnan(x)] = 0.
    # x -= x.min()
    # x *= (1. / x.max())
    # Set the maximum values above which the max color should be used.
    max = .1
    
    x = np.clip(x / max, 0., 1.)
    
    # Gamma correction. Doesn't look very good.
    # x = x ** .2

    shape = x.shape
    
    if col0 is None:
        col0 = (.67, .91, .65)
    if col1 is None:
        col1 = (0., 1., 1.)
    
    col0 = np.array(col0).reshape((1, 1, -1))
    col1 = np.array(col1).reshape((1, 1, -1))
    
    col0 = np.tile(col0, x.shape + (1,))
    col1 = np.tile(col1, x.shape + (1,))
    
    x = np.tile(x.reshape(shape + (1,)), (1, 1, 3))
    
    y = hsv_to_rgb(col0 + (col1 - col0) * x)
    
    # value of -1 = black
    y[removed,:] = 0
    # Remove diagonal.
    n = y.shape[0]
    y[xrange(n), xrange(n), :] = 0
    
    return y
    

# -----------------------------------------------------------------------------
# Data manager
# -----------------------------------------------------------------------------
class SimilarityMatrixDataManager(Manager):
    def set_data(self, similarity_matrix=None,
        cluster_colors_full=None,
        clusters_hidden=[],  # WARNING: relative indexing
        ):
        if similarity_matrix is None:
            similarity_matrix = np.zeros(0)
            cluster_colors_full = np.zeros(0)
        
        if similarity_matrix.size == 0:
            similarity_matrix = -np.ones((2, 2))
        elif similarity_matrix.shape[0] == 1:
            similarity_matrix = -np.ones((2, 2))
           
        self.similarity_matrix = similarity_matrix
        n = similarity_matrix.shape[0]
        
        self.texture = colormap(self.similarity_matrix.copy())
        # similarity_matrix axes are originally (x, y) from the lower left corner
        # but when displayed, they are (i, j) from the upper left corner
        # so we need to transpose
        self.texture = np.swapaxes(self.texture, 0, 1)
        
        self.clusters_unique = get_indices(cluster_colors_full)
        
        self.cluster_colors = cluster_colors_full
        self.nclusters = len(self.clusters_unique)
    
        # Remove hidden clusters.
        indices = np.array(sorted(set(range(self.nclusters)) - set(clusters_hidden)),
                                dtype=np.int32)
        self.indices = indices
        
        if len(indices) >= 2:
            tex0 = self.texture.copy()
            self.texture = tex0[[[_] for _ in (indices)],indices,:]
        else:
            indices = range(self.nclusters)
        
        self.clusters_displayed = self.clusters_unique[indices]
        self.nclusters_displayed = len(indices)
        
        self.texture = self.texture[::-1,:,:]
        
    
# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class SimilarityMatrixPaintManager(PlotPaintManager):
    def initialize(self):
        self.add_visual(TextureVisual,
            texture=self.data_manager.texture, 
            name='similarity_matrix')

        self.add_visual(RectanglesVisual, coordinates=np.zeros((2, 4)),
            color=(1., 1., 1., .5), autonormalizable=False, name='square')
        
        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            background_transparent=False,
            posoffset=(150., -60.),
            color=(1., 1., 1., 1.),
            letter_spacing=350.,
            depth=-1,
            visible=False)
            
    def update(self):
        self.set_data(
            texture=self.data_manager.texture, visual='similarity_matrix')
        

# -----------------------------------------------------------------------------
# Interaction
# -----------------------------------------------------------------------------
class SimilarityMatrixInfoManager(Manager):
    def initialize(self):
        pass
        
    def get_closest_cluster(self, xd, yd):
        nclu = self.data_manager.nclusters_displayed
        
        cx = int((xd + 1) / 2. * nclu)
        cy = int((yd + 1) / 2. * nclu)
        
        cx_rel = np.clip(cx, 0, nclu - 1)
        cy_rel = np.clip(cy, 0, nclu - 1)
        
        return cx_rel, cy_rel
        
    def show_closest_cluster(self, xd, yd):
        
        if len(self.data_manager.clusters_displayed) == 0:
            return
        
        cx_rel, cy_rel = self.get_closest_cluster(xd, yd)
        
        cx = self.data_manager.clusters_displayed[cx_rel]
        cy = self.data_manager.clusters_displayed[cy_rel]
        
        ind = self.data_manager.indices
        matrix = self.data_manager.similarity_matrix
        matrix = matrix[ind,:][:,ind]
        
        if ((cx_rel >= matrix.shape[0]) or
            (cy_rel >= matrix.shape[1])):
            return
            
        val = matrix[cx_rel, cy_rel]
        
        text = "%d/%d:%.3f" % (cx, cy, val)
        
        x, y = self.interaction_manager.get_processor('navigation').get_window_coordinates(xd, yd)
        
        posoffset=(-np.sign(x) * 140., -np.sign(y) * 40.)
        
        self.paint_manager.set_data(coordinates=(xd, yd), 
            text=text,
            visible=True,
            posoffset=posoffset,
            visual='clusterinfo')
        
    
class SimilarityMatrixInteractionManager(PlotInteractionManager):
    def initialize(self):
        # self.register('ShowClosestCluster', self.show_closest_cluster)
        self.register('SelectPair', self.select_pair)
        self.register('AddPair', self.add_pair)
        self.register('MoveSquare', self.move_square)
        self.register(None, self.hide_closest_cluster)
        
        self.square_coordinates = np.zeros((2, 4))
        self.clusters_selected = np.array([])
            
    def hide_closest_cluster(self, parameter):
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        self.paint_manager.set_data(visible=False, visual='square')
        
    def select_pair(self, parameter, add=False):
        if self.data_manager.nclusters_displayed == 0:
            return
            
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        cx_rel, cy_rel = self.info_manager.get_closest_cluster(xd, yd)
        cx = self.data_manager.clusters_displayed[cx_rel]
        cy = self.data_manager.clusters_displayed[cy_rel]
        if cx != cy:
            clusters = np.array([cx, cy])
        else:
            clusters = np.array([cx])
        
        if add:
            clusters = np.array(sorted(set(self.clusters_selected).union(
                clusters)))
            # clusters_new = clusters
            # clusters = self.clusters_selected.extend([cluster 
                # for cluster in clusters_new 
                    # if cluster not in self.clusters_selected])
        
        self.clusters_selected = clusters
        
        # Emit signal.
        log.debug("Selected clusters {0:s}.".format(str(clusters)))
        self.parent.clustersSelected.emit(clusters)
        
    def add_pair(self, parameter):
        self.select_pair(parameter, True)
        
    def show_cluster(self, clu0, clu1):
        if self.data_manager.nclusters_displayed <= 1:
            return
            
        n = self.data_manager.texture.shape[0]
        dx = 1 / float(n)
        i, j = np.digitize([clu0, clu1], self.data_manager.clusters_displayed) - 1
        x0, y0 = i * dx * 2 - 1, j * dx * 2 - 1,
        x1, y1 = (i + 1) * dx * 2 - 1, (j + 1) * dx * 2 - 1
        self.info_manager.show_closest_cluster((x0 + x1) / 2, (y0 + y1) / 2)
        
    def show_closest_cluster(self, parameter):
        if self.data_manager.nclusters_displayed <= 1:
            return
            
        nclu = self.data_manager.nclusters_displayed
        
        if nclu == 0:
            return
            
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        self.info_manager.show_closest_cluster(xd, yd)
        
    def show_square(self, clu0, clu1):
        if self.data_manager.nclusters_displayed <= 1:
            return
            
        n = self.data_manager.texture.shape[0]
        dx = 1 / float(n)
        i, j = np.digitize([clu0, clu1], self.data_manager.clusters_displayed) - 1
        x0, y0 = i * dx * 2 - 1, j * dx * 2 - 1,
        x1, y1 = (i + 1) * dx * 2 - 1, (j + 1) * dx * 2 - 1
        self.square_coordinates[0,:] = (x0, -1, x1, 1)
        self.square_coordinates[1,:] = (-1, y0, 1, y1)
        self.paint_manager.set_data(coordinates=self.square_coordinates, 
            visible=True,
            visual='square')
        
    def move_square(self, parameter):
        if self.data_manager.nclusters_displayed <= 1:
            return
        
        self.show_closest_cluster(parameter)
        
        # data coordinates
        x, y = parameter
        nav = self.get_processor('navigation')
        x, y = nav.get_data_coordinates(x, y)
        
        n = self.data_manager.texture.shape[0]
        dx = 1 / float(n)
        i = np.clip(int((x + 1) / 2. * n), 0, n - 1)
        j = np.clip(int((y + 1) / 2. * n), 0, n - 1)

        clu0 = self.data_manager.clusters_displayed[i]
        clu1 = self.data_manager.clusters_displayed[j]
        self.show_square(clu0, clu1)
        
        
class SimilarityMatrixBindings(KlustaViewaBindings):
    def get_base_cursor(self):
        return 'ArrowCursor'
    
    def set_selectcluster(self):
        self.set('RightClick', 'SelectPair', #key_modifier='Shift',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
        self.set('RightClick', 'AddPair', key_modifier='Control',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
    
    def set_move(self):
        self.set('Move', 'MoveSquare',
            param_getter=lambda p: p['mouse_position'])
        self.set('Move', 'MoveSquare', key_modifier='Control',
            param_getter=lambda p: p['mouse_position'])
    
    def initialize(self):
        # super(SimilarityMatrixBindings, self).initialize()
        # self.set_clusterinfo()
        self.set_selectcluster()
        self.set_move()
    

# -----------------------------------------------------------------------------
# Top-level module
# -----------------------------------------------------------------------------
class SimilarityMatrixView(KlustaView):
    
    # Raise the list of highlighted spike absolute indices.
    clustersSelected = QtCore.pyqtSignal(np.ndarray)
    
    def initialize(self):
        self.set_bindings(SimilarityMatrixBindings)
        self.set_companion_classes(
            paint_manager=SimilarityMatrixPaintManager,
            info_manager=SimilarityMatrixInfoManager,
            interaction_manager=SimilarityMatrixInteractionManager,
            data_manager=SimilarityMatrixDataManager,)
    
    def set_data(self, *args, **kwargs):
        # if kwargs.get('similarity_matrix', None) is None:
            # return
        self.data_manager.set_data(*args, **kwargs)
        
        # update?
        if self.initialized:
            self.paint_manager.update()
            self.updateGL()

    def clear(self):
        self.set_data()
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        self.paint_manager.set_data(visible=False, visual='square')
            
    def show_selection(self, clu0, clu1):
        self.interaction_manager.show_square(clu0, clu1)
        self.interaction_manager.show_cluster(clu0, clu1)
        self.updateGL()
    
    
    # def sizeHint(self):
        # return QtCore.QSize(300, 400)
        