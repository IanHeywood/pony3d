import re
from setuptools import setup, find_packages

# Single source for the version number: pony3d/__init__.py
with open('pony3d/__init__.py') as f:
    version = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read()).group(1)

setup(
    name='pony3d',
    version=version,
    author='Ian Heywood',
    author_email='ian.heywood@physics.ox.ac.uk',
    description='Parallelised 3D mask making for spectral line deconvolution',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/IanHeywood/pony3d',
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'regions',
    ],
    extras_require={
        # pony3d-movie (and progress bars in pony3d-makecube)
        'tools': ['tqdm', 'matplotlib', 'opencv-python', 'Pillow'],
        # pony3d-makecube --compress
        'compress': ['fitsio'],
    },
    entry_points={
        'console_scripts': [
            'pony3d=pony3d.main:main',
            'pony3d-movie=pony3d.tools.movie:main',
            'pony3d-makecube=pony3d.tools.makecube:main',
            'pony3d-breakcube=pony3d.tools.breakcube:main',
            'pony3d-contsub=pony3d.tools.contsub:main',
        ],
    },
    license='GPLv3',
)
