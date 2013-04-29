"""Unit tests for settings module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys


import klustaviewa
import klustaviewa.utils.globalpaths as paths

APPNAME_ORIGINAL = paths.APPNAME

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    # HACK: monkey patch
    paths.APPNAME = APPNAME_ORIGINAL + '_test'
    reload(klustaviewa.utils.userpref)
    import klustaviewa.utils.userpref as pref
    
    userpref = """field1 = 123"""
    paths.ensure_folder_exists(pref.FOLDER)
    pref.save(pref.FILEPATH, userpref, appname=pref.APPNAME)
    
def teardown():
    import klustaviewa.utils.userpref as pref
    
    paths.delete_file(pref.FILEPATH)
    paths.delete_folder(pref.FOLDER)
    
    # HACK: cancel monkey patch
    paths.APPNAME = APPNAME_ORIGINAL
    
    reload(klustaviewa.utils.userpref)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_userpref():
    import klustaviewa.utils.userpref as pref
    
    pref.USERPREF._load_once()
    assert pref.USERPREF['field1'] == 123    
    