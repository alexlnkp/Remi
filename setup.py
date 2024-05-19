try:
    from setuptools import setup  # type: ignore
except ImportError:
    from distutils.core import setup  # type: ignore

from Cython.Distutils import build_ext
from setuptools import find_packages

setup(
    name="Remi",
    version="0.3.0",
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    # If you need to include package data files:
    package_data={
        "Remi": ["Remi/infer/*", "Remi/ft/*"],
    },
)
