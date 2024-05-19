try:
    from setuptools import Extension, setup  # type: ignore
except ImportError:
    from distutils.core import setup  # type: ignore
    from distutils.extension import Extension  # type: ignore

from Cython.Distutils import build_ext

ext_modules = [
    # Extension("Remi.ft", ["Remi/ft/__init__.py", "Remi/ft/lib.py"]),
    # Extension("Remi.infer", ["Remi/infer/__init__.py", "Remi/infer/utils.py"]),
    Extension("Remi", ["Remi/__init__.py", "Remi/infer/__init__.py", "Remi/ft/__init__.py"]),
]

setup(
    name="Remi",
    version="0.3.0",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=["Remi", "Remi.ft", "Remi.infer"],
)
