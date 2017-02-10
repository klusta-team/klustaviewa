# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------

__version__ = '0.4.7'

APPNAME = 'klustaviewa'

ABOUT = """KlustaViewa is a software for semi-automatic spike sorting with high-channel count silicon probes. It is meant to be used after the automatic clustering stage. This interface automatically guides the user through the clustered data and lets him or her refine the data. The goal is to make the manual stage more reliable, quicker, and less error-prone.

This software was developed by Cyrille Rossant in the Cortical Processing Laboratory at UCL (http://www.ucl.ac.uk/cortexlab).

Version {0:s}.

""".format(__version__)


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
from kwiklib.utils import logger as log
from kwiklib.utils import userpref as pref
from kwiklib.utils import settings


# -----------------------------------------------------------------------------
# User preferences
# -----------------------------------------------------------------------------
USERAPP_FOLDER = get_app_folder()
PREFERENCES_DEFAULT_PATH = pref.get_default_preferences_path()
PREFERENCES_FILENAME = 'preferences.py'
PREFERENCES_FILEPATH = get_global_path(PREFERENCES_FILENAME)
USERPREF = pref.UserPreferences(appname=APPNAME, folder=USERAPP_FOLDER,
    filepath=PREFERENCES_FILEPATH)


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
PREFERENCES_FILENAME = 'settings'
PREFERENCES_FILEPATH = get_global_path(PREFERENCES_FILENAME)
SETTINGS = settings.Settings(appname=APPNAME, folder=USERAPP_FOLDER,
    filepath=PREFERENCES_FILEPATH)


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
