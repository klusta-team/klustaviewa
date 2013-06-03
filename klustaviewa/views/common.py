import numpy as np
import operator
import collections

from galry import *


__all__ = [
           'HighlightManager', 'KlustaViewaBindings',
           ]


class HighlightManager(Manager):
    
    highlight_rectangle_color = (0.75, 0.75, 1., .25)
    
    def initialize(self):
        self.highlight_box = None
        # self.paint_manager.ds_highlight_rectangle = \
        if not self.paint_manager.get_visual('highlight_rectangle'):
            self.paint_manager.add_visual(RectanglesVisual,
                                    coordinates=(0., 0., 0., 0.),
                                    color=self.highlight_rectangle_color,
                                    is_static=True,
                                    visible=False,
                                    name='highlight_rectangle')
    
    def highlight(self, enclosing_box):
        # get the enclosing box in the window relative coordinates
        x0, y0, x1, y1 = enclosing_box
        
        # set the highlight box, in window relative coordinates, used
        # for displaying the selection rectangle on the screen
        self.highlight_box = (x0, y0, x1, y1)
        
        # paint highlight box
        self.paint_manager.set_data(visible=True,
            coordinates=self.highlight_box,
            visual='highlight_rectangle')
        
        # convert the box coordinates in the data coordinate system
        x0, y0 = self.interaction_manager.get_processor('navigation').get_data_coordinates(x0, y0)
        x1, y1 = self.interaction_manager.get_processor('navigation').get_data_coordinates(x1, y1)
        
        self.highlighted((x0, y0, x1, y1))
        
    def highlighted(self, box):
        pass

    def cancel_highlight(self):
        # self.set_highlighted_spikes([])
        if self.highlight_box is not None:
            self.paint_manager.set_data(visible=False,
                visual='highlight_rectangle')
            self.highlight_box = None
    

class KlustaViewaBindings(PlotBindings):
    def set_panning_keyboard(self):
        pass
        
    def set_zooming_keyboard(self):
        pass
        
    def set_grid(self):
        pass
        
    def set_fullscreen(self):
        pass
    
    
class KlustaView(GalryWidget):
    pass
    