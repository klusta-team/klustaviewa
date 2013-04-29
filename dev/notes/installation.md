Installation
------------

Notes about how to simplify the installation of KlustaViewa.

Manual installation: install successively:

  * Python 2.7.3
  * Distribute
  * Numpy 
  * Matplotlib
  * PyQt4
  * PyOpenGL
  * h5py
  * galry
  * klustaviewa
  
Binary installers for all external dependencies are available for recent
versions of Windows (Chris Gohlke's webpage), Ubuntu/Debian (official
packages), OS X (MacPorts, etc.).

Automatic installation:
  
  * target users: those who don't care about Python and just want the GUI
  * goal: single-click installer which installs everything: Python, the 
    dependencies, klustaviewa
    
Ubuntu: it should be possible to create a package with all the dependencies
(which are other ubuntu packages), and even to automatize the package build
with a Python script. Search "build debian installer" in Google...

Windows: several ways to automate the installation of all the dependencies:

  * py2exe: bundle everything in a single executable
  * http://hg.python.org/cpython/file/2.7/Tools/msi/msi.py
  * create a new installer with eg NSIS, and use Autoit to automate the 
    installation of the dependencies (which are windows installers which
    require the user to press Next: autoit does that automatically)
    http://nsis.sourceforge.net/Embedding_other_installers
  * check out kivy's developers solution:
    https://github.com/kivy/kivy/blob/master/kivy/tools/packaging/win32/build.py
  