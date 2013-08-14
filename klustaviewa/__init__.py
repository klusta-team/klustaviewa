# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
__version__ = '0.2.0dev'

APPNAME = 'klustaviewa'

ABOUT = """KlustaViewa is a software for semi-automatic spike sorting with high-channel count silicon probes. It is meant to be used after the automatic clustering stage. This interface automatically guides the user through the clustered data and lets him or her refine the data. The goal is to make the manual stage more reliable, quicker, and less error-prone.

This software was developed by Cyrille Rossant in the Cortical Processing Laboratory at UCL (http://www.ucl.ac.uk/cortexlab)."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys


# -----------------------------------------------------------------------------
# Folder-related functions
# -----------------------------------------------------------------------------
def get_app_folder(appname=None):
    if appname is None:
        appname = APPNAME
    return os.path.expanduser(os.path.join('~', '.' + appname))

def get_global_path(filename, folder=None, appname=None):
    if appname is None:
        appname = APPNAME
    if folder is None:
        folder = get_app_folder(appname)
    return os.path.join(folder, filename)


# -----------------------------------------------------------------------------
# Default logger
# -----------------------------------------------------------------------------
import klustaviewa.utils.logger as log
import klustaviewa.utils.userpref as pref


# -----------------------------------------------------------------------------
# User preferences
# -----------------------------------------------------------------------------
PREFERENCES_DEFAULT_PATH = pref.get_default_preferences_path()
FILENAME = 'preferences.py'
FOLDER = get_app_folder()
FILEPATH = get_global_path(FILENAME)
USERPREF = pref.UserPreferences(appname=APPNAME, folder=FOLDER, filepath=FILEPATH)


# -----------------------------------------------------------------------------
# Loggers
# -----------------------------------------------------------------------------
LOGGERS = {}
log.LOGGERS = LOGGERS
# Console logger.
LOGGER = log.ConsoleLogger(name='{0:s}.console'.format(APPNAME))
log.register(LOGGER)

sys.excepthook = log.handle_exception

# Set the logging level specified in the user preferences.
loglevel = USERPREF['loglevel']
if loglevel:
    log.set_level(loglevel)

