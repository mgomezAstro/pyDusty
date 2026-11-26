import os
import re
import shutil
import subprocess
import tempfile

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DUSTY_SRC_DIR = os.path.join(PROJECT_ROOT, "pydusty", "fortran", "dustyV4")


def read_version():
    """Read ``__version__`` from pydusty/__version__.py without importing
    the package (so building does not need numpy etc. installed)."""
    path = os.path.join(PROJECT_ROOT, "pydusty", "__version__.py")
    with open(path, encoding="utf-8") as fh:
        match = re.search(
            r"""^__version__\s*=\s*['"]([^'"]+)['"]""", fh.read(), re.M
        )
    if match is None:
        raise RuntimeError(f"Unable to find __version__ in {path}")
    return match.group(1)


def compile_dusty(exe):
    """Compile the DUSTY (v4) Fortran code into the executable ``exe``."""
    fc = os.environ.get("FC") or "gfortran"
    fc_path = shutil.which(fc)
    if fc_path is None:
        raise RuntimeError(
            f"Fortran compiler '{fc}' not found. pydusty needs gfortran to "
            "build the DUSTY executable: install it (e.g. 'apt install "
            "gfortran', 'brew install gcc' or 'conda install -c conda-forge "
            "gfortran') and reinstall pydusty, or point the FC environment "
            "variable to a gfortran-compatible compiler."
        )

    os.makedirs(os.path.dirname(exe), exist_ok=True)
    src = os.path.join(DUSTY_SRC_DIR, "dusty.f90")
    # Keep the compiler's .mod files out of the source tree.
    with tempfile.TemporaryDirectory() as moddir:
        subprocess.check_call(
            [
                fc_path,
                "-O3",
                "-fopenmp",
                f"-I{DUSTY_SRC_DIR}",
                f"-J{moddir}",
                src,
                "-o",
                exe,
            ]
        )
    os.chmod(exe, 0o755)


class BuildPyWithDusty(build_py):
    """build_py that also compiles the DUSTY executable into pydusty/bin."""

    def run(self):
        super().run()
        if getattr(self, "editable_mode", False):
            # Editable installs import pydusty from the source tree.
            exe = os.path.join(PROJECT_ROOT, "pydusty", "bin", "dusty")
        else:
            exe = os.path.join(self.build_lib, "pydusty", "bin", "dusty")
        compile_dusty(exe)


class BinaryDistribution(Distribution):
    """pydusty ships a compiled executable, so its wheels are not pure."""

    def has_ext_modules(self):
        return True


class BdistWheelPlatform(bdist_wheel):
    """Platform-specific but Python-version-independent wheel tag."""

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(
    name="pydusty",
    version=read_version(),
    packages=find_packages(),
    # Runtime data only (bin/dusty is compiled by BuildPyWithDusty). The
    # Fortran sources are shipped in the sdist via MANIFEST.in, not here.
    package_data={
        "pydusty": [
            "data/*.txt",
            "fortran/dustyV4/data/*",
            "fortran/dustyV4/data/**/*",
        ]
    },
    include_package_data=False,
    python_requires=">=3.10",
    install_requires=["numpy", "scipy", "miepython >= 3.0", "emcee", "h5py"],
    description="Python wrapper for the radiative transfer code Dusty (v4) originally created by Maia Nenkova (2000ASPC..196...77N).",
    author="M. A. Gomez-Munoz",
    author_email="mgomez_astro@outlook.com",
    license="BSD-3-Clause",
    license_files=["LICENSE", "LICENSE-DUSTY"],
    keywords="astronomy radiative transfer dusty",
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPyWithDusty, "bdist_wheel": BdistWheelPlatform},
)
