from setuptools import setup, find_packages

setup(
    name='pony3d',
    version='0.2.0',
    author='Ian Heywood',
    author_email='ian.heywood@physics.ox.ac.uk',
    description='Parallelised 3D mask making for spectral line deconvolution',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/IanHeywood/pony3d',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
    ],
    entry_points={
        'console_scripts': [
            'pony3d=pony3d.main:main',
        ],
    },
    license='GPLv3',
)
