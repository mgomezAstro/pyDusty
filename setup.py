from setuptools import setup
from pydusty.__version__ import __version__

setup(
    name="pydusty",
    version=__version__,
    packages=["pydusty"],
    description="Python wrapper for the radiative transfer code Dusty (v4) originally created by Maia Nenkova (2000ASPC..196...77N).",
    author="M. A. Gomez-Munoz",
    author_email="mgomez_astro@outlook.com",
    keywords="astronomy radiative transfer dusty",
)
