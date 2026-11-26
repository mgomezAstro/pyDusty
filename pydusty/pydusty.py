import glob
import re
import numpy as np
import subprocess
import importlib.resources as pkg
import time
from pathlib import Path
from dataclasses import dataclass
import logging
import warnings
from typing import Union, List, Optional


logger = logging.getLogger(__name__)

# Column of each quantity in the DUSTY spectral files (.s###).
_SPECTRA_COLUMNS = {
    "fTot": 1,
    "xAtt": 2,
    "xDs": 3,
    "xDe": 4,
    "fInp": 5,
    "TauTot": 6,
    "albedo": 7,
}

# Keys always present in DustyReader.get_output_data(), in order.
_OUT_BASE_KEYS = (
    "tau0", "Ps1/Ps0", "Fi(W/m2)", "R1", "R1/Rc", "theta1",
    "T1(K)", "Td(K)", "RPr", "error%",
)

# RESULTS header names written by DUSTY -> output_data keys. Unknown header
# names are used as keys unchanged.
_OUT_COLUMN_KEYS = {
    "tau0": "tau0",
    "Psi/Psi0": "Ps1/Ps0",
    "Fi(W/m2)": "Fi(W/m2)",
    "r1(cm)": "R1",
    "r1/rc": "R1/Rc",
    "theta1": "theta1",
    "T1(K)": "T1(K)",
    "Td(K)": "Td(K)",
    "RPr(1)": "RPr",
    "e(%)": "error%",
    "Mdot": "Mdot",
    "Ve": "Ve",
    "M>": "M>",
}

# One numeric field of a Fortran-formatted table row. Handles fields glued
# together (e.g. a 9-char negative number in an e9.2 column), Fortran
# D exponents, exponents without "E" ("1.00-100") and overflow asterisks.
_OUT_TOKEN_RE = re.compile(
    r"\*+"
    r"|[-+]?(?:nan|inf(?:inity)?)"
    r"|[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+|[-+]\d+)?",
    re.IGNORECASE,
)

_MISSING_E_RE = re.compile(r"(?<=[\d.])([-+]\d+)$")


def _ffloat(s) -> float:
    """Convert a Fortran-formatted number to float.

    Handles D exponents ("1.0D+02"), 3-digit exponents written without the
    "E" ("1.0-100", "2.5+101") and overflow fields of asterisks ("****",
    returned as ``np.nan``).
    """
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    if s and set(s) == {"*"}:
        return np.nan
    t = s.upper().replace("D", "E")
    if "E" not in t:
        t = _MISSING_E_RE.sub(r"E\1", t)
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"could not convert Fortran number {s!r} to float")


class DustyError(RuntimeError):
    """Raised when a DUSTY run fails or does not produce its outputs."""


@dataclass
class DustyInp:
    model_name: str
    exe_path: Union[str, Path, None] = None
    project_dir: str | Path | None = None

    def __post_init__(self) -> None:

        if self.exe_path is None:
            self.exe_path = Path(__file__).parent / "bin/dusty"

        self._dusty_logger = logging.getLogger(__name__)
        self.geometry: str = ""
        self.geometry_params: dict = {}
        self.external_radiation: dict = {"central": "ON", "external": "OFF"}
        self.spectral_shape: dict = {}
        self.scale: dict = {}
        self.grains_abund = {
            "Sil-Ow": 0,
            "Sil-Oc": 0,
            "Sil-DL": 0,
            "grf-DL": 0,
            "amC-Hn": 0,
            "SiC-Pg": 0,
        }
        self.abund_user: dict = {"nk_n": 0}
        self.grains_params: dict = {}
        self.tau_params: dict = {}
        self.output_text: str = ">\n"
        self.comments: List[str] = ["*---------------"]
        self.options: dict = {
            "flux conservation": 0.1,
            "s": 2,
            "i": 0,
            "j": 0,
            "r": 2,
            "m": 2,
        }

        # Resolved here (not as a class-level default) so that the current
        # working directory is taken when the object is created, not at import.
        if self.project_dir is None:
            self.project_dir = Path.cwd()
        self.project_dir = Path(self.project_dir)

        self.output_path = self.project_dir
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.full_model_name = self.output_path / self.model_name

    def set_sphere(self, set_matrix: bool = False) -> None:
        self.geometry = {"GEOMETRY": "SPHERE"}
        if set_matrix:
            self.geometry = {"GEOMETRY": "SPHERE_MATRIX"}

    def set_density_profile(
        self,
        density_type: str,
        n_pwd: Optional[int] = 1,
        thickness: Optional[Union[float, List[float]]] = 1e4,
        p: Optional[Union[float, List[float]]] = 0,
        sigma: Optional[float] = None,
        profile_filename: Optional[Union[str, Path]] = None,
        profile_filenae: Optional[Union[str, Path]] = None,
    ) -> None:
        """Set the density distribution of the shell.

        density_type is one of POWD, EXPD, RDW, RDWA or USER_SUPPLIED
        ("USR_SUPPLD" is accepted as an alias of USER_SUPPLIED). For
        USER_SUPPLIED, ``profile_filename`` is the file with the density
        profile (three header lines followed by two columns: radius, density);
        a relative path is resolved from ``project_dir``, where DUSTY runs.
        ``profile_filenae`` is a deprecated misspelled alias.
        """
        if profile_filenae is not None:
            warnings.warn(
                "'profile_filenae' is deprecated; use 'profile_filename' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if profile_filename is not None:
                raise ValueError(
                    "Give only one of 'profile_filename' and 'profile_filenae'."
                )
            profile_filename = profile_filenae

        match density_type:
            case "POWD":
                self.geometry_params = {"Density Type": "POWD", "N": n_pwd}
                if n_pwd > 1:
                    thickness = " ".join([str(v) for v in thickness])
                    p = " ".join([str(v) for v in p])
                self.geometry_params["Y"] = thickness
                self.geometry_params["p"] = p
            case "EXPD":
                self.geometry_params = {
                    "Density Type": "EXPD",
                    "Y": thickness,
                    "sigma": sigma,
                }
            case "RDW" | "RDWA":
                self.geometry_params = {"Density Type": density_type, "Y": thickness}
            case "USER_SUPPLIED" | "USR_SUPPLD":
                if profile_filename is None:
                    raise ValueError(
                        "USER_SUPPLIED density requires 'profile_filename'."
                    )
                self.geometry_params = {
                    "Density Type": "USER_SUPPLIED",
                    "profile filename": str(profile_filename),
                }
            case _:
                raise ValueError(
                    "Density Type must be one of [POWD, EXPD, RDW, RDWA, or "
                    "USER_SUPPLIED (alias USR_SUPPLD)]."
                )

    def set_central_radiation(self, central: bool = True) -> None:
        if central == False:
            self.external_radiation = {"central": "OFF", "external": "ON"}

    def set_blackbody(
        self, temperature: Union[float, List[float]], l_ratios: List[float] = None
    ) -> None:
        if not isinstance(temperature, list):
            self.spectral_shape = {
                "Spectral Shape": "BLACK_BODY",
                "Number of BB": 1,
                "Temperature": temperature,
            }
        else:
            if len(temperature) != len(l_ratios):
                raise ValueError(
                    "Size of temperatures and luminosity ratios must be equal."
                )
            self.spectral_shape = {
                "Spectral Shape": "BLACK_BODY",
                "Number of BB": len(temperature),
                "Temperature": ", ".join(map(lambda x: str(x), temperature)),
                "Luminosities": ", ".join(map(lambda x: str(x), l_ratios)),
            }

    def set_star(self, filename: str, shape: str) -> None:
        self.spectral_shape = {
            "Spectral Shape": shape + "\n" + filename,
        }

    def set_radiation_strenght(
        self, scale_type: str, scale_value: Union[float, List[float]]
    ) -> None:
        self.scale = {"Scale": scale_type}

        match scale_type.upper():
            case "FLUX":
                self.scale["Fe"] = scale_value
            case "LUM_R1":
                self.scale["L"] = f"{scale_value[0]}"
                self.scale["R1"] = f"{scale_value[1]:.5e}"
            case "T1":
                self.scale["Td"] = scale_value
            case _:
                raise ValueError("Scale type must one of [FLUX, LUM_R1, T1].")

    def set_grains_abund(
        self,
        predef_abund: dict,
        subl_temp: float = 1500,
        nk_files: Optional[List[str]] = None,
        nk_abunds: Optional[List[float]] = None,
    ) -> None:
        for key in predef_abund.keys():
            if key not in self.grains_abund.keys():
                raise ValueError(
                    f"Predefined abundances must be one of these: {self.grains_abund.keys()}"
                )
            self.grains_abund[key] = predef_abund[key]
        self.grains_abund["subl_temp"] = subl_temp

        if nk_files is not None:
            self.abund_user["nk_n"] = len(nk_files)
            self.abund_user["nk_files"] = nk_files
            self.abund_user["nk_abunds"] = [str(v) for v in nk_abunds]

    def set_grain_size_dist(
        self,
        grain_distribution: str = "MRN",
        amin: float = 0.005,
        amax: float = 0.25,
        q: float = 3.5,
    ) -> None:
        match grain_distribution:
            case "MRN":
                self.grains_params = {"Size Distribution": "MODIFIED_MRN"}
            case "KMH":
                self.grains_params = {"Size Distribution": "KMH"}
            case _:
                raise ValueError("Size distribution must be one of [MRN or KMH].")
        self.grains_params["q"] = q
        self.grains_params["a(min)"] = amin
        self.grains_params["a(max)"] = amax

    def set_optical_depth(
        self, tau_grid: str, lambda0: float, taumin: float, taumax: float, n_models: int
    ) -> None:
        self.tau_params["tau_grid"] = tau_grid
        self.tau_params["lambda0"] = lambda0
        self.tau_params["tau(min)"] = taumin
        self.tau_params["tau(max)"] = taumax
        self.tau_params["n_models"] = n_models

    def set_comments(self, comments: Union[str, List[str]]) -> None:
        if isinstance(comments, list):
            self.comments.extend(comments)
        else:
            self.comments.append(comments)
        self.comments.append("*---------------")

    def _set_options_to_text(self) -> str:
        text = "-------------FLAGS-----------\n"
        for key in self.options.keys():
            if key == "flux conservation":
                text += f"{key} = {self.options[key]}\n"
                continue
            text += f"fname.{key}### = {self.options[key]}\n"
        text += "-------------------------------"

        return text

    def _tau_to_text(self) -> str:
        text = ""
        for key in self.tau_params.keys():
            text += f"{key} = {self.tau_params[key]}\n"
        text += "\n"

        return text

    def print_inp_file(self, to_console: bool = False) -> None:
        """Write ``<model_name>.inp`` from the current settings.

        Idempotent: the text is rebuilt from scratch on every call, so calling
        it several times writes the same file. The result is also kept in
        ``self.output_text``.
        """
        t: List[str] = [">\n"]
        for comment in self.comments:
            t.append(f"* {comment}\n")
        t.append("\n\n")

        def spectrum_and_scale() -> None:
            for key in self.spectral_shape.keys():
                t.append(f"{key} = {self.spectral_shape[key]}\n")
            t.append("\n")
            for key in self.scale.keys():
                t.append(f"{key} = {self.scale[key]}\n")
            t.append("\n")

        # Geomtry, spectral shape, and scale
        t.append(f"GEOMETRY = {self.geometry['GEOMETRY']}\n\n")
        t.append(f"central = {self.external_radiation['central']}\n\n")
        if self.external_radiation["central"] == "ON":
            spectrum_and_scale()
            t.append(f"external = {self.external_radiation['external']}\n\n")
        else:
            t.append(f"external = {self.external_radiation['external']}\n\n")
            spectrum_and_scale()

        # Dust properties
        if self.abund_user["nk_n"] > 0:
            t.append("optical properties index = COMMON_AND_ADDL_GRAIN_COMPOSITE\n")
        else:
            t.append("optical properties index = COMMON_GRAIN_COMPOSITE\n")
        for key in list(self.grains_abund.keys())[:6]:
            t.append(f"    {key:10}")
        t.append("\n")
        t.append("x = ")
        for key in list(self.grains_abund.keys())[:6]:
            t.append(f"  {str(self.grains_abund[key]):^10}")
        t.append("\n\n")

        ### other nk if specified
        if self.abund_user["nk_n"] > 0:
            t.append(
                f"Number of additional components = {self.abund_user['nk_n']}, "
                "properties listed in files\n"
            )
            for file in self.abund_user["nk_files"]:
                t.append(f"    {file}\n")
            t.append(
                "Abundance for these components = "
                + ", ".join(self.abund_user["nk_abunds"])
                + "\n\n"
            )

        ### grain distribution
        for key in self.grains_params.keys():
            t.append(f"{key} = {self.grains_params[key]}\n")
        t.append("\n")

        t.append(f"Sublimation Temperature = {self.grains_abund['subl_temp']}\n\n")

        # Density of the shell
        for key in self.geometry_params.keys():
            if key == "profile filename":
                # DUSTY reads the file name as the next non-empty line, verbatim
                t.append(f"{self.geometry_params[key]}\n")
                continue
            t.append(f"{key} = {self.geometry_params[key]}\n")
        t.append("\n")

        t.append(self._tau_to_text())
        t.append(self._set_options_to_text())

        self.output_text = "".join(t)

        if to_console:
            print(self.output_text)

        with open(str(self.full_model_name) + ".inp", "w") as o:
            o.write(self.output_text)

    def _clean_outputs(self) -> None:
        """Remove outputs of a previous run of this model (keeps the .inp)."""
        for f in self.output_path.glob(f"{glob.escape(self.model_name)}.*"):
            if f.suffix != ".inp" and f.is_file():
                f.unlink()

    def run(self, verbose=False) -> None:
        self._clean_outputs()

        data_path = pkg.files("pydusty").joinpath("fortran/dustyV4")
        cmd = [
            str(Path(self.exe_path).resolve()),
            f"{self.model_name}.inp",
            "2",
            str(data_path),
        ]

        start_timer = 0.0
        if verbose:
            print("\nRunning DUSTY (v4)")
            print(f"Model name: {self.model_name}")
            start_timer = time.time()

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.output_path,
                capture_output=not verbose,
                text=True,
            )
        except OSError as e:
            raise DustyError(f"Could not execute DUSTY ({cmd[0]}): {e}") from e

        if verbose:
            end_timer = time.time() - start_timer
            print(f"Ended after: {end_timer:.2f}s\n")

        expected = [Path(str(self.full_model_name) + ".out")]
        if self.options.get("s") == 2:
            expected.append(Path(str(self.full_model_name) + ".s001"))
        missing = [f.name for f in expected if not f.is_file()]

        if proc.returncode != 0 or missing:
            tail = ""
            if not verbose:
                out = ((proc.stdout or "") + (proc.stderr or "")).strip()
                tail = "\n".join(out.splitlines()[-20:])
            raise DustyError(
                f"DUSTY failed for model '{self.full_model_name}' "
                f"(returncode={proc.returncode}, missing outputs={missing}).\n"
                + tail
            )


@dataclass
class DustyReader:
    model_name: Union[str, Path]

    def __post_init__(self):
        self.n_models: int = 0
        self.Psi0: float = 0.0
        self._comments = ""
        with open(str(self.model_name) + ".inp", "r") as o:
            for line in o:
                if "n_models" in line:
                    self.n_models = int(line.split("=")[1])
        with open(str(self.model_name) + ".out", "r") as o:
            for line in o:
                if line[0] == "*":
                    self._comments += line.rstrip().lstrip() + "\n"
                if "IE97" in line:
                    self.Psi0 = float(line.split("=")[1])

    @property
    def _get_output_data(self) -> dict:
        """Parse the RESULTS table of the ``.out`` file.

        Kept as a property for backward compatibility; prefer
        :meth:`get_output_data`.
        """
        output_data = {key: [] for key in _OUT_BASE_KEYS}

        with open(str(self.model_name) + ".out", "r") as f:
            lines = f.read().splitlines()

        # Locate the (last) RESULTS block, then its header row and separator.
        results_idx = None
        for i, row in enumerate(lines):
            if "RESULTS" in row:
                results_idx = i
        if results_idx is None:
            raise ValueError(f"No RESULTS block found in {self.model_name}.out")

        header = None
        sep_idx = None
        for i in range(results_idx + 1, len(lines)):
            row = lines[i]
            if header is None:
                if row.lstrip().startswith("###") and "tau0" in row.lower():
                    header = row.split()[1:]
            elif row.strip().startswith("==="):
                sep_idx = i
                break
        if header is None or sep_idx is None:
            raise ValueError(
                f"Could not find the RESULTS table header in {self.model_name}.out"
            )

        keys = [_OUT_COLUMN_KEYS.get(name, name) for name in header]
        for key in keys:
            output_data.setdefault(key, [])

        n_rows = 0
        for row in lines[sep_idx + 1 :]:
            if n_rows >= self.n_models:
                break
            # Drop the optional trailing "#WARNING: ..." text.
            row = row.split("#", 1)[0]
            tokens = _OUT_TOKEN_RE.findall(row)
            # Data rows start with the integer model index.
            if not tokens or not tokens[0].isdigit():
                break
            values = tokens[1:]
            if len(values) < len(keys):
                raise ValueError(
                    f"Malformed RESULTS row in {self.model_name}.out: {row!r}"
                )
            for key, value in zip(keys, values):
                output_data[key].append(_ffloat(value))
            n_rows += 1

        if n_rows < self.n_models:
            logger.warning(
                "%s.out: expected %d model rows in RESULTS, found %d.",
                self.model_name,
                self.n_models,
                n_rows,
            )

        return output_data

    def get_comments(self) -> str:
        return self._comments

    def get_output_data(self) -> dict:
        """Return the RESULTS table of the ``.out`` file as a dict of lists."""
        return self._get_output_data

    def get_spectra(
        self,
        y_cont: str = "fTot",
        x_unit: str = "um",
    ) -> tuple:

        if y_cont not in _SPECTRA_COLUMNS:
            raise ValueError(
                "y_cont must be one of [fTot, xAtt, xDs, xDe, fInp, TauTot, albedo]."
            )
        if x_unit.lower() not in ("um", "micron", "ang", "aa", "angstroms",
                                  "um-1", "micron-1"):
            raise ValueError("x_unit must be one of [um, Ang, um-1].")
        if self.n_models < 1:
            raise ValueError(
                f"No models to read: n_models={self.n_models} in "
                f"{self.model_name}.inp."
            )

        usecols = (0, _SPECTRA_COLUMNS[y_cont])
        skiprows = 6
        seds = []
        for i in range(self.n_models):
            spec_filename = str(self.model_name) + f".s{i + 1:03}"
            try:
                wave, flux = np.loadtxt(
                    spec_filename,
                    skiprows=skiprows,
                    usecols=usecols,
                    comments="#",
                    unpack=True,
                )
            except ValueError:
                # Rare Fortran number formats (e.g. "1.0-100", "1.0D+02",
                # "*****"): fall back to the slower tolerant converter.
                wave, flux = np.loadtxt(
                    spec_filename,
                    skiprows=skiprows,
                    usecols=usecols,
                    comments="#",
                    unpack=True,
                    converters=_ffloat,
                    dtype=float,
                )
            seds.append(flux)

        match x_unit.lower():
            case "um" | "micron":
                wave *= 1
            case "ang" | "aa" | "angstroms":
                wave *= 1e4
            case "um-1" | "micron-1":
                wave = 1 / wave

        return np.array(wave), np.array(seds)

    def get_fbol(self) -> List[float]:
        fbol = []
        for i in range(self.n_models):
            with open(str(self.model_name) + f".s{i+1:03}", "r") as f:
                fbol.append(_ffloat(f.readlines()[2].split("=")[1]))
        return fbol
