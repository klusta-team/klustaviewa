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
from matplotlib.colors import hsv_to_rgb
    
from klustaviewa.io.selection import get_indices
from klustaviewa.io.tools import get_array
from klustaviewa.utils.colors import COLORMAP
import klustaviewa.utils.logger as log
from klustaviewa.views.common import HighlightManager, KlustaViewaBindings


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
    x -= x.min()
    x *= (1. / x.max())
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
        clusters_hidden=[],
        ):
        
        if similarity_matrix is None:
            similarity_matrix = np.zeros(0)
            cluster_colors_full = np.zeros(0)
        
        if similarity_matrix.size == 0:
            similarity_matrix = -np.ones((2, 2))
        elif similarity_matrix.shape[0] == 1:
            similarity_matrix = -np.ones((2, 2))
        # else:
            # # Normalize the correlation matrix.
            # s = similarity_matrix.sum(axis=1)
            # similarity_matrix[s == 0, 0] = 1e-9
            # s = similarity_matrix.sum(axis=1)
            # similarity_matrix *= (1. / s.reshape((-1, 1)))
        n = similarity_matrix.shape[0]
        
        self.texture = colormap(similarity_matrix)[::-1, :, :]
        self.similarity_matrix = similarity_matrix
        
        # Hide some clusters.
        tex0 = self.texture.copy()
        for clu in clusters_hidden:
            # Inversion of axes in the y axis
            self.texture[- clu - 1, :, :] = tex0[- clu - 1, :, :] * .5
            self.texture[:, clu, :] = tex0[:, clu, :] * .5
        
        self.clusters_unique = get_indices(cluster_colors_full)
        self.cluster_colors = cluster_colors_full
        self.nclusters = len(self.clusters_unique)
    
    
# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class SimilarityMatrixPaintManager(PlotPaintManager):
    def initialize(self):
        self.add_visual(TextureVisual,
            texture=self.data_manager.texture, 
            name='similarity_matrix')

        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            background_transparent=False,
            posoffset=(50., -60.),
            color=(1., 1., 1., 1.),
            letter_spacing=350.,
            depth=-1,
            visible=False)
            
        self.add_visual(RectanglesVisual, coordinates=(0., 0., 0., 0.),
            color=(1., 1., 1., .75), autonormalizable=False, name='square')
        
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
        nclu = self.data_manager.nclusters
        
        cy = int((xd + 1) / 2. * nclu)
        cx = int((yd + 1) / 2. * nclu)
        
        cx_rel = np.clip(cx, 0, nclu - 1)
        cy_rel = np.clip(cy, 0, nclu - 1)
        
        return cx_rel, cy_rel
        
    def show_closest_cluster(self, xd, yd):
        
        cx_rel, cy_rel = self.get_closest_cluster(xd, yd)
        
        cx = self.data_manager.clusters_unique[cx_rel]
        cy = self.data_manager.clusters_unique[cy_rel]
        
        if ((cx_rel >= self.data_manager.similarity_matrix.shape[0]) or
            (cy_rel >= self.data_manager.similarity_matrix.shape[1])):
            return
            
        val = self.data_manager.similarity_matrix[cx_rel, cy_rel]
        
        text = "%d/%d:%.3f" % (cy, cx, val)
        
        self.paint_manager.set_data(coordinates=(xd, yd), 
            text=text,
            visible=True,
            visual='clusterinfo')
        
    
class SimilarityMatrixInteractionManager(PlotInteractionManager):
    def initialize(self):
        # self.register('ShowClosestCluster', self.show_closest_cluster)
        self.register('SelectPair', self.select_pair)
        self.register('AddPair', self.add_pair)
        self.register('MoveSquare', self.move_square)
        self.register(None, self.hide_closest_cluster)
        
        self.clusters_selected = np.array([])
            
    def hide_closest_cluster(self, parameter):
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        self.paint_manager.set_data(visible=False, visual='square')
        
    def select_pair(self, parameter, add=False):
        if self.data_manager.nclusters == 0:
            return
            
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        cx_rel, cy_rel = self.info_manager.get_closest_cluster(xd, yd)
        
        cx = self.data_manager.clusters_unique[cx_rel]
        cy = self.data_manager.clusters_unique[cy_rel]
        clusters = np.unique([cx, cy])
        
        if add:
            clusters = np.array(sorted(set(self.clusters_selected).union(
                clusters)))
        
        self.clusters_selected = clusters
        
        # Emit signal.
        log.debug("Selected clusters {0:s}.".format(str(clusters)))
        self.parent.clustersSelected.emit(clusters)
        
    def add_pair(self, parameter):
        self.select_pair(parameter, True)
        
    def show_closest_cluster(self, parameter):
        nclu = self.data_manager.nclusters
        
        if nclu == 0:
            return
            
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        self.info_manager.show_closest_cluster(xd, yd)
        
    def move_square(self, parameter):
        if self.data_manager.nclusters == 0:
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
        coordinates = (
            i * dx * 2 - 1, 
            j * dx * 2 - 1, 
            (i + 1) * dx * 2 - 1, 
            (j + 1) * dx * 2 - 1)
        self.paint_manager.set_data(coordinates=coordinates, visible=True,
            visual='square')
        
        
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
class SimilarityMatrixView(GalryWidget):
    
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

    
    
    def sizeHint(self):
        return QtCore.QSize(300, 400)
        