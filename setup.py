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
                                                'pywave/src/wave_ctriangle.c',
                                                'pywave/src/wave_ctrianglen.c',
                                                'pywave/src/wave_cntriangle.c',
                                                'pywave/src/wave_cntrianglen.c',
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
                                                'pywave/src/wave_ctriangle.c',
                                                'pywave/src/wave_ctrianglen.c',
                                                'pywave/src/wave_cntriangle.c',
                                                'pywave/src/wave_cntrianglen.c',
                                                'pywave/src/wave_decart.c',
                                                'pywave/src/wave_win.c',
                                                'pywave/src/wave_memcpy.c',
                                                'pywave/src/wave_fft1.c'],
                                                include_dirs=[numpy.get_include()])

afd_module = Extension('afdcfun', sources=['pywave/src/afd.c',
                                                'pywave/src/wave_fdm.c',
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
                                                'pywave/src/wave_ctriangle.c',
                                                'pywave/src/wave_ctrianglen.c',
                                                'pywave/src/wave_cntriangle.c',
                                                'pywave/src/wave_cntrianglen.c',
                                                'pywave/src/wave_decart.c',
                                                'pywave/src/wave_win.c',
                                                'pywave/src/wave_memcpy.c',
                                                'pywave/src/wave_fft1.c'],
                                                include_dirs=[numpy.get_include()])

pfwi_module = Extension('pfwicfun', sources=['pywave/src/pfwi.c',
                                                'pywave/src/wave_fwi.c',
                                                'pywave/src/wave_fwiutil.c',
                                                'pywave/src/wave_fwigradient.c',
                                                'pywave/src/wave_fwilbfgs.c',
                                                'pywave/src/wave_fwimodeling.c',
                                                'pywave/src/wave_triutil.c',
                                                'pywave/src/wave_bigsolver.c',
                                                'pywave/src/wave_cgstep.c',
                                                'pywave/src/wave_butter.c',
                                                'pywave/src/wave_chain.c',
                                                'pywave/src/wave_fdm.c',
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
                                                'pywave/src/wave_blas.c',
                                                'pywave/src/wave_blasc.c',
                                                'pywave/src/wave_decart.c',
                                                'pywave/src/wave_win.c',
                                                'pywave/src/wave_memcpy.c',
                                                'pywave/src/wave_fft1.c'],
                                                include_dirs=[numpy.get_include()])
                                                                                                                     
setup(
    name="pywave",
    version="0.0.3",
    license='MIT License',
    description="An open-source Python package for solving wave equations using various methods for educational purposes",
    long_description=long_description,
    author="pywave developing team",
    author_email="chenyk2016@gmail.com",
    url="https://github.com/chenyk1990/pywave",
    ext_modules=[ftfa_module,aps_module,afd_module,pfwi_module],
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
        "License :: OSI Approved :: MIT License "
    ],
    keywords=[
        "seismology", "seismic waves", "wave propagation", "earthquake seismology", "exploration seismology", "array seismology", "denoising", "science", "engineering"
    ],
    install_requires=[
        "numpy", "scipy", "matplotlib"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
