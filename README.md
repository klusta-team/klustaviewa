KlustaViewa
===========

*KlustaViewa* is a software for semi-automatic spike sorting with high-channel count silicon probes. It streamlines the manual stage occurring after the automatic clustering stage by guiding the user through the clustered data, letting him or her refine the clusters. The goal is to make the manual stage more reliable, quicker, and less error-prone.

This software was developed by Cyrille Rossant in the [Cortical Processing Laboratory](http://www.ucl.ac.uk/cortexlab) at UCL.


User guide
----------

  * [See the user guide here](https://github.com/klusta-team/klustaviewa/blob/master/docs/manual.md).


Screenshots
-----------

[![Screenshot 1](images/thumbnails/img0.png)](images/img0.PNG)
[![Screenshot 2](images/thumbnails/img1.png)](images/img1.PNG)


Installation
------------

  * The software is in beta version at this time.
  * It can run on Windows, Mac OS X and Linux.
  * This document contains all the instructions to install the software on all systems.

### Windows 7 & 8 (64 bits)

#### Standard method

  * Step 1: [download the KlustaViewa Full Installer for Windows 64 bits (>200MB)](http://klustaviewa.rossant.net/klustaviewa-setup.exe).
  * Step 2: execute the installer.
  * Step 3: click on the *KlustaViewa* icon on your desktop or in the Start menu to launch the software.
  * Step 4: to update the software at any time, execute *Update KlustaViewa* in the Start menu (you need an active Internet connection).
  
#### Alternative method (don't use it unless you're a Python developer)

Use this only if you know what you're doing and you don't want to use the first method.

  * Step 1: if you don't already have a Python installation, [download this ZIP file (72MB)](http://klustaviewa.rossant.net/klustaviewa-dependencies.zip) containing a full Python distribution (required by KlustaViewa).
  * Step 2: unzip the archive. You'll find several installers: execute them all one by one, in the right order.
  * Step 3: download the [KlustaViewa Light Installer for Windows 64 bits (<1MB)](http://klustaviewa.rossant.net/klustaviewa-0.1.0.dev.win-amd64-py2.7.exe) and execute it.
  * Step 4: you can launch the software with the shortcut named *KlustaViewa* on your desktop and in the start menu.

Once the software is installed, you can update it when a new version is available by doing again **Step 3** only.


### Mac OS X and Linux

  * Step 1: [install Anaconda](http://continuum.io/downloads). Download the adequate version (64 bits if possible). If you're asked confirmation at some steps during installation, answer *yes*.
  
  * Step 2: execute the following commands:

        $ pip install pyopengl
        $ conda install -c http://conda.binstar.org/asmeurer pyside
        $ pip install klustaviewa

  * Step 3: to run KlustaViewa, type the following command in a system shell:
  
        klustaviewa


Details
-------

### Dependencies
  
The following libraries are required:
  
  * Python 2.7
  * Numpy
  * Pandas
  * Matplotlib
  * PyOpenGL
  * either PyQt4 or PySide


### OpenGL
  
KlustaViewa requires OpenGL >= 2.1. To find out which version of OpenGL is available on your system:

  * Use [OpenGL Extensions Viewer](http://www.realtech-vr.com/glview/)
  * Alternatively, on Linux, run `glxinfo`.

KlustaViewa works better with a good graphics card as it uses hardware-accelerated visualization. With a lower end graphics card, the software will work but somewhat slower.


### Development version

Use this if you want to be able to update with `git pull` (you need git).

  * Clone the repository:
  
        git clone https://github.com/rossant/klustaviewa.git
  
  * Install KlustaViewa with `pip` so that external packages are automatically updated (like `qtools` which contains some Qt-related utility functions):
  
        pip install -r requirements.txt
        

### IPython 1.0.dev

If IPython 1.0.dev is installed, then you will have the possibility to open an IPython terminal in the context of the GUI. This lets you access all elements and data variables programatically. To install this version of IPython, do the following:

  * Ensure that any version of IPython is uninstalled.
  * Install pygments with `pip install pygments`
  * Execute the following commands:
  
        git clone https://github.com/ipython/ipython.git
        cd ipython
        python setupegg.py develop
  
Then, in the software, you will be able to open an interactive IPython shell (with pylab mode activated) in the Views menu. Type `%who` to see the list of available variables.


Contact
-------

If you have any trouble, bug, comment or suggestion:
  
  * You can [send a message on the Google group](https://groups.google.com/forum/?fromgroups#!forum/klustaviewas).
  * You can send an e-mail to the author of the software: cyrille *dot* rossant *at* gmail *dot* com.


