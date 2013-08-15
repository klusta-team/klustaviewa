"""Internal persistent settings store with cPickle."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import cPickle
import os

from kwiklib.utils.globalpaths import ensure_folder_exists


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load(filepath):
    """Load the settings from the file, and creates it if it does not exist."""
    if not os.path.exists(filepath):
        save(filepath)
    with open(filepath, 'rb') as f:
        settings = cPickle.load(f)
    return settings
    
def save(filepath, settings={}):
    """Save the settings in the file."""
    with open(filepath, 'wb') as f:
        cPickle.dump(settings, f)
    return settings


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
class Settings(object):
    """Manage internal settings.
    
    They are stored in a binary file in the user home folder.
    
    Settings are only loaded once from disk as soon as an user preference field
    is explicitely requested.
    
    """
    def __init__(self, appname=None, folder=None, filepath=None,
        autosave=True):
        """The settings file is not loaded here, but only once when a field is
        first accessed."""
        self.appname = appname
        self.folder = folder
        self.filepath = filepath
        self.settings = {}
        self.settings = None
        self.autosave = autosave
    
    # I/O methods
    # -----------
    def _load_once(self):
        """Load or create the settings file, unless it has already been
        loaded."""
        if self.settings is None:
            # Create the folder if it does not exist.
            ensure_folder_exists(self.folder)
            # Load or create the settings file.
            self.settings = load(self.filepath)
    
    def save(self):
        save(self.filepath, self.settings)
    
    
    # Getter and setter methods
    # -------------------------
    def set(self, key, value):
        self._load_once()
        self.settings[key] = value
        if self.autosave:
            self.save()
    
    def get(self, key, default=None):
        self._load_once()
        return self.settings.get(key, default)
        
    def __setitem__(self, key, value):
        self.set(key, value)
        
    def __getitem__(self, key):
        return self.get(key)
        
        