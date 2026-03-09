import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


class InstallDUSTYv4(build_py):

    def run(self):

        project_root = os.path.abspath(os.path.dirname(__file__))
        datadir = os.path.join(project_root, "pydusty/fortran/dustyV4")
        include = os.path.join(project_root, "pydusty/fortran/dustyV4")
        bindir = os.path.join(project_root, "pydusty/bin")

        exe = os.path.join(bindir, "dusty")
        src = "pydusty/fortran/dustyV4/dusty.f90"

        subprocess.check_call(
            [
                "gfortran",
                "-O3",
                "-lgomp",
                "-fopenmp",
                "-cpp",
                f'-DDATADIR_MACRO="{datadir}/"',
                f"-I{include}",
                src,
                "-o",
                exe,
            ]
        )

        super().run()


setup(
    name="pydusty",
    version="1.10.10",
    packages=find_packages(),
    package_data={"pydusty": ["data/*.txt", "bin/dusty", "fortran/dustyV4/**/*"]},
    include_package_data=True,
    install_requires=["miepython >= 3.0"],
    description="Python wrapper for the radiative transfer code Dusty (v4) originally created by Maia Nenkova (2000ASPC..196...77N).",
    author="M. A. Gomez-Munoz",
    author_email="mgomez_astro@outlook.com",
    keywords="astronomy radiative transfer dusty",
    cmdclass={"build_py": InstallDUSTYv4},
)
