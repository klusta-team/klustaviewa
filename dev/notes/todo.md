KlustaViewa: graphical interface for semi-automatic spike sorting
===========================================================

Message for alpha pre-release version:

    This is an alpha version of a software that is still in development.
    Some features are missing, bugs and crashes may appear, and the software
    may be slow with large data sets. 
    These issues will be fixed in later releases.
    Make sure you backup your data before loading it (although the software
    won't modify any of your files by default, proposing you to save in 
    a different CLU file).
  
  
Minor features
--------------

  * in feature view, add non-selected spikes in gray in the background
  * CTRL + click in correlation matrix = add pair to the selection
  
  * put scale and grid in all views
  * option to change width/bin of correlograms
  * option to toggle normalization in correlograms view
  * when selecting a group, only show the corresponding clusters in the
    correlation matrix
  * most recent selected cluster on top in both views
    
    
  * buttons or menu for all commands (reset view for any view)
  * store visualization-related information in the XML file (cluster colors,
    probe scaling, etc) (klustaviewa table)
  * feature view: switch X/Y
  
  * option to toggle showing masks as gray waveforms in waveform view (T)
  * masks & transparency: shading, continuity, hsv space
  
  * robot: function for automatic zoom in waveformview as a function of
    channels and clusters
  * fix dock issue with feature view
  * undock maximize button
  * interaction mode buttons: deactivate in gray to highlight the current mode

  * initial launch: default window geometry config  
  * make settings independent from pyqt/pyside
  
Ideas
-----
  
  * ISI widget
  * multiple widgets of the same type
  * feature view: when masks toggled (features gray) not possible to select
    them. when no masks, everything can be selected.
  * trace view (neuroscope)
  * test: fetdim variable in controller widget (1 to ?)
  * (Measure of cluster quality: ratio of mask/unmask on each channel)


Fixes
-----

  * make sure the GUI work in IPython
  * force cleaning up of widgets upon closing
  * bug fix focus issue with floating docks
  
  
Refactoring
-----------

  * in gui.py, put all actions in a separate class, and avoid communicating 
    directly from mainwindow to widgets, rather, use signals
  * use pandas for keeping track of spike and cluster absolute indices
  * refactor interactions in waveformview/featureview with different
    processors...
  * refactoring correlograms: put the actual computations outside dataio
  * put COLORMAP in the data holder and remove dependencies 
  * move data manager into templates, so that templates contain everything


Multiplatform notes
-------------------

  * pyside: crash when running the software = erase the .INI file
  * PySide: bug when runnin python test.py, one needs to do ipython then run...
  * windows/pyside cluster view: cluster selection not blue but gray (style not working?)

  