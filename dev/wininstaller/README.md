This document describes how to create a full standalone installer for KlustaViewa on Windows 64 bits.

The idea is to create a standalone folder with a full Python distribution and all external packages, and with KlustaViewa as well. This folder is created using [WinPython](https://code.google.com/p/winpython/) which is a portable scientific Python distribution. Then, we use [Inno Setup](http://www.jrsoftware.org/isinfo.php) to create a Windows installer which copies this directory as is on the user's computer. Finally, we create some scripts and shortcuts to make it easy for the user to launch the software, and to update it.

To create the installer:

  * In this folder `klustaviewa/dev/wininstaller`, create a sub-folder named `KlustaViewa` with the following hierarchy:
      * `downloads/`: an empty folder.
      * `tools/`: a folder with `update.py`.
      * `WinPython-64bit-2.7.5.0/`: a folder with the full WinPython installation. KlustaViewa should be installed using the WinPython installer which accepts a Windows executable installer generated with distutils.
      * `favicon.ico`: the icon.

  * Run Inno Setup and create the installer using `klustaviewa.iss`. The installer will be created in `klustaviewa-setup.exe`.

Now, end-users can just download this installer, execute it, and they will be able to launch the software using the icons created on the desktop. The main icon executes `pythonw.exe` with the script path as an argument.

The update icon calls the `update.py` script, which does the following:

  * Connect to `[URL]/filename.txt` to retrieve the filename of the latest version of the software.
  * Download `[URL]/[FILENAME]` and save it locally in the `downloads` folder.
  * Use the `winpython` module to install this Windows installer created with distutils.

As an example, here is the Inno Setup line that creates the shortcut:

    Name: "{group}\KlustaViewa"; Filename: "{app}\WinPython-64bit-2.7.5.0\python-2.7.5.amd64\pythonw.exe"; WorkingDir: "{app}"; Parameters: """{app}\WinPython-64bit-2.7.5.0\python-2.7.5.amd64\Lib\site-packages\klustaviewa\scripts\runklustaviewa.py"""; IconFilename: "{app}\favicon.ico"
    


