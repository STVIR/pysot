from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        '_mask',
        sources=['common/maskApi.c', '_mask.pyx'],
        include_dirs = [np.get_include(), 'common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(name='pycocotools',
      packages=['pycocotools'],
      package_dir = {'pycocotools': '.'},
      version='2.0',
      ext_modules=
          cythonize(ext_modules)
      )
