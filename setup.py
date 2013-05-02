import os
# from distutils.core import setup
from setuptools import *

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
                  'klustaviewa.gui',
                  'klustaviewa.io',
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
        
        # scripts=['scripts/runklustaviewa.py'],
        
        url='https://github.com/rossant/klustavieway',
        license='LICENSE.md',
        description='Spike sorting graphical interface.',
        long_description=LONG_DESCRIPTION,
        install_requires=[
            "numpy >= 1.6",
            "PyOpenGL >= 3.0",
        ],
    )