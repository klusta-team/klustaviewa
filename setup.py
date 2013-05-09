
import os
# from distutils.core import setup
# from setuptools import *

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
        name='klustaviewa',
        version='0.1.0.dev',  # alpha pre-release
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
        entry_points = {
            'console_scripts': [
                'klustaviewa = klustaviewa.scripts.runklustaviewa:main' ]
        },
        
        package_data={
            'klustaviewa': ['icons/*.png', 'gui/*.css'],
            
            # INCLUDE GALRY
            'galry': ['cursors/*.png', 'icons/*.png'],
            'galry.visuals': ['fontmaps/*.*'],
            'galry.test': ['autosave/*REF.png'],
            
        },
        
        # Cython stuff.
        cmdclass = cmdclass,
        ext_modules=ext_modules,
        
        include_dirs=np.get_include(),
        
        url='https://github.com/rossant/klustavieway',
        license='LICENSE.md',
        description='Spike sorting graphical interface.',
        long_description=LONG_DESCRIPTION,
        install_requires=[
            "numpy >= 1.7",
            "matplotlib >= 1.1.1",
        ],
    )