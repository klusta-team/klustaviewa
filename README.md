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
  * Installing the software is easy on Windows, but more complicated *for now* on Mac OS X and Linux. **We're trying to make the installation easier on these systems.**
  * This document contains all the instructions to install the software on all systems.

### Windows 7/8 64 bits

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


### Ubuntu >= 12.10

  * Step 1: type in a shell:

        $ sudo apt-get install python2.7 python-numpy python-pandas python-matplotlib python-opengl python-qt4 python-qt4-gl python-distribute python-pip python-nose

  * Step 2: [download KlustaViewa](http://klustaviewa.rossant.net/klustaviewa-0.1.0.dev.zip) and extract the package.
  
  * Step 3: open a system shell in the directory where you extracted the package, and execute the following command:
  
        sudo python setup.py install

  * Step 4: to run KlustaViewa, type the following command in a system shell:
  
        klustaviewa


### Ubuntu < 12.10

  * Step 1: install ActivePython-2.7.
  
      * Step 1.1: [download the package](http://www.activestate.com/activepython/downloads/thank-you?dl=http://downloads.activestate.com/ActivePython/releases/2.7.2.5/ActivePython-2.7.2.5-linux-x86_64.tar.gz). [Here is the link to the main website for your reference](http://www.activestate.com/activepython/downloads). 
      
      * Step 1.2: install it with:

            tar xzf ActivePython-2.7.2.5-linux-x86_64.tar.gz
            cd ActivePython-2.7.2.5-linux-x86_64
            sudo ./install.sh
            
      * Step 1.3: put the following line in your `~/.bashrc`:
      
            export PATH=/opt/ActivePython-2.7/bin:$PATH
  
  * Step 2: to install the required Python packages, type in a shell:
  
        pypm install distribute numpy pandas matplotlib pyopengl

  * Step 3: Install Qt 4.8.4 and the Python headers:
  
        $ sudo apt-get install qt-sdk python-dev
  
  * Step 4: Install SIP 4.14.7:
  
      * Step 4.1: [download the source](http://sourceforge.net/projects/pyqt/files/sip/sip-4.14.7/sip-4.14.7.tar.gz).
      * Step 4.2: extract the package.
      
            $ tar -xvf sip-4.14.7.tar.gz
            
      * Step 4.3: type in the extracted directory:
      
            $ cd sip-4.14.7
            $ sudo /opt/ActivePython-2.7/bin/python configure.py
            $ sudo make
            $ sudo make install
            
  * Step 5: install PyQt 4.10.2:
  
      * Step 5.1: [download the source](http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.10.2/PyQt-x11-gpl-4.10.2.tar.gz/download).
      * Step 5.2: extract the package:
      
            $ tar -xvf PyQt-x11-gpl-4.10.2.tar.gz
      
      * Step 5.3: in the extracted directory, type the following commands thay may take a while (grab a coffee). Note that you need to specify the path to SIP, which is generally `/opt/ActivePython-2.7/bin/sip`. You need to look further up the end of the output of the last command in step 4.3.
      
            $ sudo /opt/ActivePython-2.7/bin/python configure-ng.py --sip=/opt/ActivePython-2.7/bin/sip
            $   yes
            $ sudo make
            $ sudo make install
            
  * Step 6: [download KlustaViewa](http://klustaviewa.rossant.net/klustaviewa-0.1.0.dev.zip) and extract the package.
  
  * Step 7: open a system shell in the directory where you extracted the package, and execute the following command:
  
        python setup.py install
  
  * Step 8: to run KlustaViewa, type the following command in a system shell:
  
        klustaviewa


### Mac OS X
    
  * Step 0 (only for OS X 10.8): allow unsigned applications installation in your system preferences so that you can install the packages. [Here are the instructions for OS X 10.8](https://www.my-private-network.co.uk/knowledge-base/apple-related-questions/osx-unsigned-apps.html).

  * Step 1: [install ActivePython](http://downloads.activestate.com/ActivePython/releases/2.7.2.5/ActivePython-2.7.2.5-macosx10.5-i386-x86_64.dmg). [Here is the link to the main website for your reference](http://www.activestate.com/activepython/downloads).
  
  * Step 2: to install the required Python packages, type in a shell:
  
        sudo pypm install numpy
        sudo pypm install pandas
        sudo pypm install matplotlib
        sudo pypm install pyopengl
        
  * Step 3: [install Apple Xcode](http://itunes.apple.com/us/app/xcode/id497799835?ls=1&mt=12).

  * Step 4: [install the Xcode command line tools](http://stackoverflow.com/questions/9329243/xcode-4-4-command-line-tools?answertab=votes#tab-top).
  
  * Step 5: [install Qt 4.8.4](http://download.qt-project.org/official_releases/qt/4.8/4.8.4/qt-mac-opensource-4.8.4.dmg).
  
  * Step 6: Install SIP 4.14.7:
  
      * Step 6.1: [download the source](http://sourceforge.net/projects/pyqt/files/sip/sip-4.14.7/sip-4.14.7.tar.gz).
      * Step 6.2: extract the package.
      * Step 6.3: type in the extracted directory:
      
            $ sudo python configure.py
            $ sudo make
            $ sudo make install
            
  * Step 7: install PyQt 4.10.2:
  
      * Step 7.1: [download the source](http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.10.2/PyQt-mac-gpl-4.10.2.tar.gz).
      * Step 7.2: extract the package.
      * Step 7.3: in the extracted directory, type the following commands thay may take a while (grab a coffee). Note that you need to specify the path to SIP, which is generally either `/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/sip` or `/System/Library/Frameworks/Python.framework/Versions/2.7/bin/sip`. You need to look further up the end of the output of the last command in step 6.3.
      
            $ sudo python configure-ng.py --sip=[SIP_PATH_THAT_YOU_NEED_TO_COPY_FROM_THE_OUTPUT_OF_STEP_6]
            $ sudo make
            $ sudo make install
            
  * Step 8: [download KlustaViewa](http://klustaviewa.rossant.net/klustaviewa-0.1.0.dev.zip) and extract the package.
  
  * Step 9: open a system shell in the directory where you extracted the package, and execute the following command:
  
        sudo python setup.py install
  
  * Step 10: to run KlustaViewa, type the following command in a system shell:
  
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

  
Troubleshooting
---------------

### Common errors in Windows

  * If you can't run `python` or `klustaviewa` in a console, you may need to add `C:\Python27\Scripts` in the PATH variable, as [described here](http://geekswithblogs.net/renso/archive/2009/10/21/how-to-set-the-windows-path-in-windows-7.aspx).


### Contact

If you have any trouble, bug, comment or suggestion:
  
  * You can [send a message on the Google group](https://groups.google.com/forum/?fromgroups#!forum/klustaviewas).
  * You can send an e-mail to the author of the software: cyrille *dot* rossant *at* gmail *dot* com.


