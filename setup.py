import os
import os.path as op
import re
from setuptools import setup

import numpy as np

cmdclass = { }
ext_modules = [ ]


# Find the version.
curdir = op.dirname(op.realpath(__file__))
filename = op.join(curdir, 'klustaviewa/__init__.py')
with open(filename, 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


LONG_DESCRIPTION = """Spike sorting graphical interface."""

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

if __name__ == '__main__':

    setup(
        zip_safe=False,
        name='klustaviewa',
        version=version,
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
