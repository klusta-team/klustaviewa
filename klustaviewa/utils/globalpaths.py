"""Define global variables with useful paths."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def ensure_folder_exists(folder):
    """Create the settings folder if it does not exist."""
    if not os.path.exists(folder):
        os.mkdir(folder)

def delete_file(filepath):
    """Delete a file."""
    if os.path.exists(filepath):
        os.remove(filepath)

def delete_folder(folderpath):
    """Delete a folder."""
    if os.path.exists(folderpath):
        if os.path.isdir(folderpath):
            # shutil.rmtree(folderpath)
            os.rmdir(folderpath)
