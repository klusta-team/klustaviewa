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

  * store visualization-related information in the XML file (cluster colors,
    probe scaling, etc) (klustaviewa table)
    
  * masks & transparency: shading, continuity, hsv space
  
  
Ideas
-----
  
  * ISI widget
  * feature view: when masks toggled (features gray) not possible to select
    them. when no masks, everything can be selected.
  * trace view (neuroscope)
  * (Measure of cluster quality: ratio of mask/unmask on each channel)


Fixes
-----

  * make sure the GUI work in IPython
  * bug fix focus issue with floating docks
  
  
Multiplatform notes
-------------------

  * PySide: bug when running python test.py, one needs to do ipython then run...
  * windows/pyside cluster view: cluster selection not blue but gray (style not working?)

  