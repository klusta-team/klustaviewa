Development notes
=================

This document describes the architecture of KlustaViewa. It is intended to those interested in extending the software. It is merely a guide for reading the code, the best way to understand the software's architecture is to look at the code directly (and the unit tests). This document is up-to-date at the time of writing (mid-July 2013).


High-level overview
-------------------

KlustaViewa is a software for the manual stage of spike sorting, which comes after the automatic clustering stage. It includes a semi-automatic assistant that guides the user through ambiguous clusters, letting him or her refine the data. It is specifically adapted to recordings made with silicon probes containing tens of channels per shank.

The software is written in Python, and depends on NumPy and Pandas for high-performance vectorized computing, PyQt for the graphical user interface, matplotlib and PyOpenGL for efficient hardware-accelerated interactive visualization. It works on any platform where Python and those dependencies are available: Windows, Mac OS X, and Linux at least.

Other home-made dependencies include Galry and QTools. Galry is a high-performance interactive data visualization library based on PyOpenGL, and QTools includes some Qt-related tools, notably an easy-to-use multithread and multiprocess task queue. As Galry is still an experimental project at the time of writing, it is automatically bundled in the package during the build procedure (as well as QTools). Users don't have to install Galry and QTools by themselves.

Some small portions of the code (notably the computation of all pairwise correlograms) have a Cython version for achieving higher performance, so that a C compiler is necessary when installing or building the software from source.

The architecture is modular: there are several sub-packages handling various aspects of the software functionality. Each sub-package includes a testing suite, based on the nose package. The unit tests ensure that every aspect of every sub-package works as expected. Looking at these tests is the best way to understand the internals of the software.


Sub-packages
------------

Currently, the sub-packages are:

  * control: implement the actions performed on the data, like merging, splitting, etc.
  * dataio: implement loading/saving functions.
  * gui: implement the Qt graphical user interface, including multithreading and multiprocessing capabilities for keeping the interface responsive, and taking advantage of multicore processors.
  * stats: implement statistical functions, like cross-correlations and the similarity matrix. It also defines specific data structures for handling matrices of values with non-contiguous indices.
  * utils: implement various utility functions.
  * views: implement the different views, notably the OpenGL-based widgets for visualization of the data.
  * wizard: implement the wizard.

The two most important sub-packages are: dataio and views. These two sub-packages implement the core of the software, allowing to load data from files and visualize them.


Data I/O
--------

Broadly speaking, this module implements two things: loading features, and selection features. Loading features enable the software to load the data from files stored in the hard drive, to the system memory. These features depend on the specific file formats, and different loaders can correspond to different file formats. Managing memory is a tricky issue because the files may be too large for residing in system memory, so some sort of memory mapping needs to be implemented. One of the easiest possibility is to use the HDF5 file format, and that is currently work in progress.

The loaders also implement an interface for accessing the data: all the data, cluster-specific data, or spike-specific data. This interface is based on the selection features. These are based on Pandas, which allows to keep track of absolute indices in data arrays, which is especially useful when selecting portions of the data. The selection features may need to be aware of the file formats, particularly when memory mapping is involved.


Views
-----

The views are essentially based on Galry, except non-visualization views like the ClusterView and the IPythonView. Each view is a standalone Qt widget deriving from GalryWidget (and QGLWidget). This fact is important: it means that the views can be tested independently from the rest of the software. Every view implements a testing suite which launches the view with randomly generated data.

The views also implement public methods allowing to make minor changes to the views (e.g. changing the projections, the scales, etc.). They also implement user actions using Galry's infrastructure: user-triggered actions may emit widget-level signals, that may be bound by the top-level widget (essentially the main window). In short, it means that views communicate with the rest of the software differently according to the direction of commmunication:

  * widget to outside: Qt signals.
  * outside to widget: widget public methods.

Each view implements several specific classes (that's what Galry requires), called managers:

  * a DataManager, handling the transformation from data passed by the loader/selector in the dataio sub-package, to a format suited to Galry/OpenGL (typically NumPy arrays with the adequate structure for vertex buffer objects or textures).
  * Visuals, which contain shader code for GPU-accelerated visualization, and are typically based on built-in visuals in Galry.
  * PaintManager, which handles the different visuals. Each visual is one piece of visual information with an homogeneous type: a set of lines, a set of triangles, a set of textured rectangles, etc.
  * InteractionManager, which implements user actions that have an influence on the visualization.
  * Bindings, that bind actions (mouse, keyboard, etc.) to higher-level events. This is where keyboard shortcuts are implemented.
  * the View itself: the standalone visualization widget, which can also implement public methods as well as Qt signals.

Specific views can also implement more specialized managers, like a HighlightManager for transient highlighting feature, a InfoManager for tool tips, etc.


Control
-------

The actions that make changes on the data are implemented in this sub-package. There are basically three elements: 

  * Stack
  * Controller
  * Processor
  
The stack implements and undo stack, which keeps track of all performed actions on the data so that the actions can be undone if necessary. It is a generic class.

The controller offers high-level actions, to merge, split, change the cluster colors, etc. It also offers an undo and a redo methods.

The processor actually implements the actions. The processor acts on the underlying data through the loader object. Every method has a corresponding undo method. The arguments of the processor's methods are important, because they are deeply copied in the stack, so that these arguments must contain all necessary information to undo the action. For example, when merging two clusters, it is not sufficient to keep track of the two clusters, we also need to backup the previous assignment to either of the two clusters so that we can undo the merge action. When calling a method of the controller, the arguments are saved. The undo method must have the exact same arguments as its direct counterpart. 


GUI
---

The main window is implemented in the GUI sub-package. There is also a specific mechanism implementing the chain of events following each user action. For example, when a merge is requested, the controller's merge method need to be called, then the similarity matrix needs to be recomputed in an external process, the different views need to be updated. When the similarity matrix has been computed, the view needs to be updated, the quality is also updated in the ClusterView, and the wizard is also updated. This chain of events is implemented in the task graph. It contains a list of methods, each method returning a list of other methods to be called upon method completion.

The computation of the correlograms and the similarity matrix happens in external processes, so that they don't block the UI, and can execute on different CPUs. This is implemented in `threads.py`.

A buffer allows to handle impatient users who select a lot of clusters in a short amount of time. This could make the software crash as these operations may take some time (especially if they require hard-disk access). The buffer only accepts user actions after a short delay if no other action has been requested in the meantime. It makes the software much more responsive.


Stats
-----

This sub-package implements the computation of the correlograms and similarity matrix. The indexed matrix structure implements a matrix with values indexed by arbitrary numbers, which is useful for manipulating the similarity matrix. There's also a cache system so that correlograms that have already been computed are not computed again. However, if the corresponding clusters have changed, the cache is invalidated for these specific clusters.


Utils
-----

Various utility features are implemented in this sub-package.

User settings, which are parameters automatically set while the user uses the software, like the last opened file, the position of the different views, etc.

User preferences are defined in a Python file located in the user's home folder. There's a default file in `preferences_default.py`, which is automatically copied in `~/.klustaviewa/preferences.py` when running the software if this file does not already exist.

There's also a logger that saves all actions performed by the user in a local `.kvwlg` file and in a global logger file in `~/.klustaviewa/`.

The color modules implements color-related routines and the generation of the cluster colors in the HSV space.


Wizard
------

This sub-package implements the wizard. The wizard class can compute the best and candidate clusters, and offers public methods to navigate in the different pairs.
