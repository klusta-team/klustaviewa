import os
from setuptools import setup

import numpy as np

cmdclass = { }
ext_modules = [ ]

LONG_DESCRIPTION = """Spike sorting graphical interface."""

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

if __name__ == '__main__':

    setup(
        zip_safe=False,
        name='klustaviewa',
        version='0.4.0',
        author='Cyrille Rossant',
        author_email='rossant@github',
        packages=['klustaviewa',
                  'klustaviewa.control',
                  'klustaviewa.control.tests',
                  'klustaviewa.gui',
                  'klustaviewa.gui.tests',
                  'klustaviewa.scripts',
                  'klustaviewa.stats',
                  'klustaviewa.stats.tests',
                  'klustaviewa.views',
                  'klustaviewa.views.tests',
                  'klustaviewa.wizard',
                  'klustaviewa.wizard.tests',

                  ],

        # Scripts.
        entry_points={
            'gui_scripts': [
                'klustaviewa = klustaviewa.scripts.runklustaviewa:main',
                ],
        },

        # app=['klustaviewa/scripts/runklustaviewa.pyw',
        #     ],

        package_data={
            'klustaviewa': ['icons/*.png', 'icons/*.ico', 'gui/*.css'],

        },

        # Cython stuff.
        cmdclass = cmdclass,
        ext_modules=ext_modules,

        include_dirs=np.get_include(),

        url='https://klusta-team.github.io',
        license='LICENSE.md',
        description='Spike sorting software suite.',
        long_description=LONG_DESCRIPTION,
    )
