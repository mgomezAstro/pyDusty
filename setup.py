from setuptools import setup, find_packages

setup(
    name="pydusty",
    version="0.2.0",
    packages=find_packages(),
    install_requires=["miepython"],
    description="Python wrapper for the radiative transfer code Dusty (v4) originally created by Maia Nenkova (2000ASPC..196...77N).",
    author="M. A. Gomez-Munoz",
    author_email="mgomez_astro@outlook.com",
    keywords="astronomy radiative transfer dusty",
)
