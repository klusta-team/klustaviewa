"""This Python script located in {app}/tools/ updates automatically the 
software."""
from __future__ import print_function
import sys
import os
import urllib2
import time
import tempfile

from winpython import wppm, utils

# Find the directory of the application.
APPDIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
BASEURL = 'http://klustaviewa.rossant.net/'

# Find the filename of the installer by looking at the value in
# BASEURL/filename.txt
try:
    filename = urllib2.urlopen(os.path.join(BASEURL, 'filename.txt')).read()
except:
    raise Exception(("Unable to retrieve on the server the filename "
        "of the installer."))

# Generate the local path of the downloaded file.
# TEMPDIR = tempfile.mkdtemp()
# localpath = os.path.join(TEMPDIR, "{0:s}".format(filename))
localpath = os.path.join(APPDIR, "downloads/{0:s}".format(filename))

# Download the file and save it.
url = os.path.join(BASEURL, filename)
print("Downloading {0:s}...".format(url), end=' ')
try:
    with open(localpath, 'wb') as f:
        f.write(urllib2.urlopen(url).read())
except:
    raise Exception("Unable to download the file to {0:s}.".format(localpath))
print("Done!")

# Install the package.
print("Installing the distribution...", end=' ')
try:
    dist = wppm.Distribution(sys.prefix)
    package = wppm.Package(localpath)
    dist.install(package)
except:
    raise Exception("Unable to install the package.")
print("Done!")

time.sleep(1)
