KlustaViewa: graphical interface for semi-automatic spike sorting
===========================================================
  
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

  