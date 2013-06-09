import os
import sys
import shutil

import klustaviewa    

DESKTOP_FOLDER = get_special_folder_path("CSIDL_DESKTOPDIRECTORY")
STARTMENU_FOLDER = get_special_folder_path("CSIDL_STARTMENU")
NAME = 'KlustaViewa.lnk'

if sys.argv[1] == '-install':
    create_shortcut(
        os.path.join(sys.prefix, 'pythonw.exe'), # program
        'KlustaViewa: graphical user interface for semi-automatic spike sorting',
        NAME, # filename
        os.path.join(os.path.dirname(klustaviewa.__file__), 'scripts/runklustaviewa.py'),
        '', # workdir
        # to create ICO from PNG: http://www.icoconverter.com/
        os.path.join(os.path.dirname(klustaviewa.__file__), 'icons/favicon.ico'), # iconpath
    )
    # move shortcut from current directory to folders
    shutil.copyfile(os.path.join(os.getcwd(), NAME),
                os.path.join(DESKTOP_FOLDER, NAME))
    shutil.move(os.path.join(os.getcwd(), NAME),
                os.path.join(STARTMENU_FOLDER, NAME))
    # tell windows installer that we created another
    # file which should be deleted on uninstallation
    file_created(os.path.join(DESKTOP_FOLDER, NAME))
    file_created(os.path.join(STARTMENU_FOLDER, NAME))

# This will be run on uninstallation. Nothing to do.
if sys.argv[1] == '-remove':
    pass
    
    
