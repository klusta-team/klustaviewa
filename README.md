KlustaViewa
===========

*KlustaViewa* (previously *Spiky*) is a software for semi-automatic spike 
sorting with high-channel count silicon probes.
It is meant to be used after the automatic clustering stage.
This interface automatically guides the user through the clustered data and 
lets him or her refine the data. 
The goal is to make the manual stage more reliable, quicker, and less
error-prone.

This software was developed by Cyrille Rossant in the [Cortical Processing Laboratory](http://www.ucl.ac.uk/cortexlab) at UCL.

Installation
------------

The software is still in development, but you can download an experimental
version here.

### Step 1

You need Python 2.7 and a bunch of other common scientific packages.

  * Install [Python 2.7](http://python.org/download/) (available on Windows, OS X, or Linux),
    64 bits (recommended) or 32 bits.

For the external packages, the procedure depends on your system.

#### Windows

[Go on this webpage](http://www.lfd.uci.edu/~gohlke/pythonlibs/) and 
download:
    
  * Numpy >= 1.7
  * Pandas >= 0.10
  * Matplotlib >= 1.1.1
  * PyOpenGL >= 3.0.1
  * PyQt4
  * Distribute
  * (optional) Pip (for installing Python packages easily)
  * (optional) PyTables (for a future version which will support HDF5)
  * (optional) IPython 1.0.dev (for interactive computing)
  * (optional) Nose (for unit tests)
  * (optional) Cython
  
For each package, be sure to choose the appropriate version (it should be
`*.win-amd64-py2.7.exe` for Python 64 bits, or `*.win32-py2.7.exe`
for Python 32 bits).

#### OS X

Go on [HomeBrew](http://mxcl.github.io/homebrew/) and install the packages
listed above.

#### Ubuntu >= 13.x

(Untested) Type in a shell:

    $ sudo apt-get install python2.7 python-numpy python-pandas python-matplotlib python-opengl python-qt4 python-qt4-gl python-distribute python-pip python-nose

#### Ubuntu < 13.x

(Untested) Type in a shell:

    $ sudo apt-get install python2.7 python-matplotlib python-opengl python-qt4 python-qt4-gl python-distribute python-pip python-nose
    
Then, you need to install NumPy >= 1.7 and Pandas >= 0.10, and they are only available for Ubuntu >= 13.x. To install them on Ubuntu < 13.x, you can do the following:

  * Go [here](http://packages.ubuntu.com/raring/python/) and download:
  
      * python-numpy
      * python-pandas-lib
      * python-pandas
      
  * Install them, in that order, with `dpgk -i python-numpy_1.7.1-1ubuntu1_amd64.deb` and so on, replacing the file name with the appropriate ones (`amd64` for Python 64 bits, `i386` for Python 32 bits).


#### IPython 1.0.dev

This step is optional and only useful to experienced users. If IPython 1.0.dev is installed, then you will have the possibility to open an IPython terminal in the context of the GUI. This allows you to access all elements and data variables programatically. To install this version of IPython, do the following:

  * Ensure that any version of IPython is uninstalled.
  * Install pygments with `pip install pygments`
  * Execute the following commands:
  
        git clone https://github.com/ipython/ipython.git
        cd ipython
        python setupegg.py develop
  
Then, in the software, you will be able to open an interactive IPython shell (with pylab mode activated) in the Views menu. Type `%who` to see the list of available variables.

  
### Step 2

  * Windows & Python 64 bits: [download and install KlustaViewa.](http://klustaviewa.rossant.net/klustaviewa-0.1.0.dev.win-amd64-py2.7.exe)
  Skip steps 3 and 4 and go directly to step 5.
  * Other configurations: [download KlustaViewa.](http://klustaviewa.rossant.net/klustaviewa-0.1.0.dev.zip)

Some portions of the code are accelerated using Cython, so you need a C compiler like gcc if you want to use the accelerated version. If you don't have one, a slower version will be automatically used.

### Step 3

Extract this ZIP file in a temporary folder.


### Step 4

  * On Windows, double-click on `windows_install.bat`.
  * On other systems, open a system shell in the temporary directory where
    you extracted the package, and execute the following command:
    `python setup.py install`.

### Step 5

To run the software, execute `klustaviewa` in a system shell.

On Windows, you may need to add `C:\Python27\Scripts` in the PATH variable,
as [described here](http://geekswithblogs.net/renso/archive/2009/10/21/how-to-set-the-windows-path-in-windows-7.aspx).


Screenshots
-----------

[![Screenshot 1](images/img0.png)](images/img0.png)
[![Screenshot 2](images/img1.png)](images/img1.png)


Details
-------

### Dependencies
  
The following libraries are required:
  
  * Python 2.7
  * Numpy >= 1.7
  * Pandas >= 0.10
  * Matplotlib >= 1.1.1
  * PyOpenGL >= 3.0.1
  * either PyQt4 or PySide

More generally, we also recommend [pip](https://pypi.python.org/pypi/pip) and 
[IPython](http://ipython.org/) as they are commonly used for 
scientific computing with Python.
  
### OpenGL
  
For KlustaView, you need OpenGL >= 2.1. To find out which version of OpenGL 
you have:

  * Use [OpenGL Extensions Viewer](http://www.realtech-vr.com/glview/)
  * Alternatively, on Linux, run `glxinfo`.

KlustaViewa works better with a good graphics card as it uses
hardware-accelerated visualization. With a lower end graphics card, the
software will work but somewhat slower.


### Development version

Use this if you want to be able to update with `git pull` (you need git).

  * Clone the repository:
  
        git clone https://github.com/rossant/klustaviewa.git
  
  * Install KlustaViewa with `pip` so that external packages are automatically
    updated (like `qtools` which contains some Qt-related utility functions):
  
        pip install -r requirements.txt

  