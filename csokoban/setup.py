#!/usr/bin/python3

import os
from setuptools import setup, find_packages, Extension
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                ext = ".cpp"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


extensions = [
    Extension("gym_csokoban.envs.csokoban", ["gym_csokoban/envs/csokoban.pyx"]),
]

#CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None
CYTHONIZE = True

if CYTHONIZE:
    #compiler_directives = {"language_level": 3, "embedsignature": True}
    compiler_directives = {}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    name="gym_csokoban",
    version="0.0.1",
    packages=["gym_csokoban"],
    ext_modules=extensions,
    install_requires=install_requires,
    include_dirs=[".", numpy.get_include()]
)