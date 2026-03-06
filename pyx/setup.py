# setup.py
# -----------------------------------------------------------------------
#  Builds the sqg_cy Cython extension that wraps sqg.c / sqg.h.
#
#  Usage
#  -----
#  # Build in-place (development):
#      pip install cython numpy
#      python setup.py build_ext --inplace
#
#  # Install into the active environment:
#      pip install .
#
#  # Or use the modern pyproject.toml path (see pyproject.toml):
#      pip install --no-build-isolation -e .
#
#  The compiled extension will be named sqg_cy.<platform>.so (Linux/macOS)
#  or sqg_cy.<platform>.pyd (Windows) and can be imported as:
#      from sqg_cy import SQG
# -----------------------------------------------------------------------

from setuptools import setup, Extension
import numpy as np
import os

# ---- locate FFTW3 headers / libraries --------------------------------
# Override by setting FFTW_DIR in the environment, e.g.:
#   FFTW_DIR=/opt/fftw3 python setup.py build_ext --inplace

fftw_dir   = os.environ.get("FFTW_DIR", "")
fftw_inc   = os.path.join(fftw_dir, "include") if fftw_dir else "/usr/include"
fftw_lib   = os.path.join(fftw_dir, "lib")     if fftw_dir else "/usr/lib"

# ---- Cython .pyx -> .c pre-compilation (optional) --------------------
# If Cython is installed we re-generate sqg_cy.c from sqg_cy.pyx.
# If only the pre-generated .c is present (distributed tarballs) we fall
# back to using it directly.
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        [Extension(
            name="sqg_cy",
            sources=["sqg_cy.pyx", "sqg.c"],
            include_dirs=[
                np.get_include(),
                fftw_inc,
                ".",          # for sqg.h
            ],
            library_dirs=[fftw_lib],
            libraries=["fftw3f", "m"],
            extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        )],
        compiler_directives={
            "language_level": "3",
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
        },
    )
except ImportError:
    # Fall back to pre-generated sqg_cy.c
    ext_modules = [
        Extension(
            name="sqg_cy",
            sources=["sqg_cy.c", "sqg.c"],
            include_dirs=[
                np.get_include(),
                fftw_inc,
                ".",
            ],
            library_dirs=[fftw_lib],
            libraries=["fftw3f", "m"],
            extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        )
    ]

setup(
    name="sqg_cy",
    version="1.0.0",
    description="Cython wrapper for the C SQG turbulence model",
    ext_modules=ext_modules,
    python_requires=">=3.8",
    install_requires=["numpy"],
)
