from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize(Extension("linear_assignment", ["spg/linear_assignment.pyx"],
        include_dirs = [np.get_include()]))    
)
