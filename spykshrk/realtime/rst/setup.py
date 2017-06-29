from setuptools import setup, find_packages
from setuptools.extension import Extension 
from Cython.Build import cythonize

import numpy


sourcefiles = ['RSTPython.pyx','RStarTree.c','RSTInterUtil.c','RSTInOut.c','RSTInstDel.c','RSTUtil.c','RSTQuery.c','RSTJoin.c','RSTFunctions.c']
		
extensions = [Extension('RSTPython', sourcefiles, include_dirs=[numpy.get_include()])]

setup (name = 'RSTPython',
       ext_modules = cythonize(extensions),
       packages=['spykshrk','spykshrk.franklab',
       package_dir={'':'.'})
