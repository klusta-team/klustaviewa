
import os
# from distutils.core import setup
from setuptools import *

# Try importing Cython.
from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True
    
import numpy as np

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("klustaviewa.stats.correlograms_cython", 
            ["klustaviewa/stats/correlograms_cython.pyx"]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("klustaviewa.stats.correlograms_cython", 
            ["klustaviewa/stats/correlograms_cython.c"]),
    ]


LONG_DESCRIPTION = """Spike sorting graphical interface."""

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

if __name__ == '__main__':

    setup(
        zip_safe=False,
        name='klustaviewa',
        version='0.2.0',
        author='Cyrille Rossant',
        author_email='rossant@github',
        packages=['klustaviewa',
                  'klustaviewa.control',
                  'klustaviewa.dataio',
                  'klustaviewa.gui',
                  'klustaviewa.scripts',
                  'klustaviewa.stats',
                  'klustaviewa.utils',
                  'klustaviewa.views',
                  'klustaviewa.wizard',

                  #<
                  # 'qtools',
                  # 'qtools.qtpy',
                  # 'qtools.tests',
                  
                  # 'galry',
                  # 'galry.managers',
                  # 'galry.processors',
                  # 'galry.test',
                  # 'galry.visuals',
                  # 'galry.visuals.fontmaps',
                  #>
                  
                  ],
        
        # Scripts.
        entry_points={
            'gui_scripts': [
                'klustaviewa = klustaviewa.scripts.runklustaviewa:main' ]
        },
        
        scripts=['postinstall.py'],
        
        package_data={
            'klustaviewa': ['icons/*.png', 'icons/*.ico', 'gui/*.css'],
            
            # INCLUDE GALRY
            'galry': ['cursors/*.png', 'icons/*.png'],
            'galry.visuals': ['fontmaps/*.*'],
            'galry.test': ['autosave/*REF.png'],
            
        },
        
        # Cython stuff.
        cmdclass = cmdclass,
        ext_modules=ext_modules,
        
        include_dirs=np.get_include(),
        
        url='https://github.com/klusta-team/klustaviewa',
        license='LICENSE.md',
        description='Spike sorting graphical interface.',
        long_description=LONG_DESCRIPTION,
    )