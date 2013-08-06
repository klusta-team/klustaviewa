"""Manager read-only user preferences stored in a user-editable Python file."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import cPickle
import os

from klustaviewa.utils.globalpaths import (get_global_path, get_app_folder, APPNAME)
import klustaviewa.utils.logger as log
from klustaviewa.utils.settings import ensure_folder_exists


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load(filepath, appname=''):
    """Load the settings from the file, and creates it if it does not exist."""
    if not os.path.exists(filepath):
        save(filepath, appname=appname)
    with open(filepath, 'r') as f:
        preferences_string = f.read()
    # Parse the preferences string.
    preferences = {}
    try:
        exec(preferences_string, {}, preferences)
    except Exception as e:
        log.exception("An exception occurred in the user preferences file.")
    return preferences
    
def save(filepath, preferences=None, appname=''):
    """Save the preferences in the file, only for the default file."""
    if preferences is None:
        default = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'preferences_default.py')
        with open(default, 'r') as f:
            preferences = f.read()
    with open(filepath, 'w') as f:
        f.write(preferences)
        # f.write("# User preferences for {0:s}\n{1:s}".format(appname,
            # preferences))
    return preferences


# -----------------------------------------------------------------------------
# User preferences
# -----------------------------------------------------------------------------
class UserPreferences(object):
    """Manage user preferences.
    
    They are stored in a user-editable Python file in the user home folder.
    
    Preferences are only loaded once from disk as soon as an user preference field
    is explicitely requested.
    
    """
    def __init__(self, appname=None, folder=None, filepath=None):
        """The preferences file is not loaded here, but only once when a field is
        first accessed."""
        self.appname = appname
        self.folder = folder
        self.filepath = filepath
        self.preferences = None
    
    
    # I/O methods
    # -----------
    def _load_once(self):
        """Load or create the preferences file, unless it has already been
        loaded."""
        if self.preferences is None:
            # Create the folder if it does not exist.
            ensure_folder_exists(self.folder)
            # Load or create the preferences file.
            self.preferences = load(self.filepath, appname=self.appname)

    def refresh(self):
        self.preferences = None
        self._load_once()
    
    
    # Getter methods
    # --------------
    def get(self, key, default=None):
        self._load_once()
        return self.preferences.get(key, default)
        
    def __getitem__(self, key):
        return self.get(key)
        
    def __setitem__(self, key, value):
        self.preferences[key] = value


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
FILENAME = 'preferences.py'
FOLDER = get_app_folder()
FILEPATH = get_global_path(FILENAME)
USERPREF = UserPreferences(appname=APPNAME, folder=FOLDER, filepath=FILEPATH)


