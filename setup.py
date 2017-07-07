
from setuptools import setup, find_packages
from setuptools.extension import Extension 
from Cython.Build import cythonize

import numpy


sourcefiles = ['RSTPython.pyx', 
               'RStarTree.c','RSTInterUtil.c','RSTInOut.c','RSTInstDel.c','RSTUtil.c','RSTQuery.c','RSTJoin.c','RSTFunctions.c']
		
sourcefiles = ['./src/spykshrk/realtime/rst/' + src for src in sourcefiles]

extensions = [Extension('spykshrk.realtime.rst.RSTPython', sourcefiles, include_dirs=[numpy.get_include()])]

setup (name = 'Spykshrk Realtime',
       ext_modules = cythonize(extensions),
       packages=find_packages('./src/'),
       package_dir={'':'./src/'})
