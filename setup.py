from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name='toolkit.utils.region',
        sources=[
            'toolkit/utils/region.pyx',
            'toolkit/utils/src/region.c',
        ],
        include_dirs=[
            'toolkit/utils/src'
        ]
    )
]

setup(
    name='toolkit',
    packages=['toolkit'],
    ext_modules=cythonize(ext_modules)
)
