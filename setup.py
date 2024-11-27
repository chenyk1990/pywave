#!/usr/bin/env python
# -*- encoding: utf8 -*-
import io
import os

#from setuptools import find_packages
from setuptools import setup
from distutils.core import Extension
import numpy


long_description = """
Source code: https://github.com/chenyk1990/pywave""".strip() 


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")).read()

from distutils.core import Extension

ftfa_module = Extension('ftfacfun', sources=['pywave/src/tf.c',
                                                'pywave/src/wave_alloc.c',
                                                'pywave/src/wave_kissfft.c',
                                                'pywave/src/wave_komplex.c',
                                                'pywave/src/wave_conjgrad.c',
                                                'pywave/src/wave_cdivn.c',
                                                'pywave/src/wave_triangle.c',
                                                'pywave/src/wave_trianglen.c',
                                                'pywave/src/wave_ntriangle.c',
                                                'pywave/src/wave_ntrianglen.c',
                                                'pywave/src/wave_decart.c',
                                                'pywave/src/wave_win.c',
                                                'pywave/src/wave_memcpy.c',
                                                'pywave/src/wave_fft1.c'],
                                                include_dirs=[numpy.get_include()])

aps_module = Extension('apscfun', sources=['pywave/src/aps.c',
                                                'pywave/src/wave_psp.c',
                                                'pywave/src/wave_ricker.c',
                                                'pywave/src/wave_abc.c',
                                                'pywave/src/wave_fft2.c',
                                                'pywave/src/wave_fft3.c',
                                                'pywave/src/wave_freqfilt.c',
                                                'pywave/src/wave_alloc.c',
                                                'pywave/src/wave_kissfft.c',
                                                'pywave/src/wave_komplex.c',
                                                'pywave/src/wave_conjgrad.c',
                                                'pywave/src/wave_cdivn.c',
                                                'pywave/src/wave_triangle.c',
                                                'pywave/src/wave_trianglen.c',
                                                'pywave/src/wave_ntriangle.c',
                                                'pywave/src/wave_ntrianglen.c',
                                                'pywave/src/wave_decart.c',
                                                'pywave/src/wave_win.c',
                                                'pywave/src/wave_memcpy.c',
                                                'pywave/src/wave_fft1.c'],
                                                include_dirs=[numpy.get_include()])
                                                                                       
setup(
    name="pywave",
    version="0.0.2",
    license='GNU General Public License, Version 3 (GPLv3)',
    description="A python package of non-stationary predictive filtering for denoising and interpolation of multi-dimensional multi-channel seismic data",
    long_description=long_description,
    author="pywave developing team",
    author_email="chenyk2016@gmail.com",
    url="https://github.com/chenyk1990/pywave",
    ext_modules=[ftfa_module,aps_module],
    packages=['pywave'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    keywords=[
        "seismology", "earthquake seismology", "exploration seismology", "array seismology", "denoising", "science", "engineering", "structure", "local slope", "filtering"
    ],
    install_requires=[
        "numpy", "scipy", "matplotlib"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
