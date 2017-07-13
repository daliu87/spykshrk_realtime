
from setuptools import setup, find_packages
from setuptools.extension import Extension 
from Cython.Build import cythonize

import numpy


rst_sourcefiles = ['RSTPython.pyx', 
                   'RStarTree.c','RSTInterUtil.c','RSTInOut.c','RSTInstDel.c','RSTUtil.c','RSTQuery.c','RSTJoin.c','RSTFunctions.c']

rst_sourcefiles = ['./src/spykshrk/realtime/rst/' + src for src in rst_sourcefiles]

binary_rec_sourcefiles = ['./src/spykshrk/realtime/binary_record_cy.pyx']

extensions = [Extension('spykshrk.realtime.rst.RSTPython', rst_sourcefiles, include_dirs=[numpy.get_include()]),
              Extension('spykshrk.realtime.binary_record_cy', binary_rec_sourcefiles, include_dirs=[numpy.get_include()])]

setup (name = 'Spykshrk Realtime',
       ext_modules = cythonize(extensions),
       packages=find_packages('./src/'),
       package_dir={'':'./src/'})
