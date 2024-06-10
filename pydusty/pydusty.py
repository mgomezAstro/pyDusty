import numpy as np
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Union, List, Optional


@dataclass
class DustyInp:
    model_name: str
    exe_path: Union[str, Path]

    def __post_init__(self) -> None:
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

        self.output_path = Path.cwd() / "output/"
        if not self.output_path.exists():
            self.output_path.mkdir()
        self.full_model_name = self.output_path / self.model_name

    def set_sphere(self) -> None:
        self.geometry = {"GEOMETRY": "SPHERE"}

    def set_density_profile(
        self,
        density_type: str,
        n_pwd: Optional[int] = 1,
        thickness: Optional[Union[float, List[float]]] = 1e4,
        p: Optional[Union[float, List[float]]] = 0,
        sigma: Optional[float] = None,
        profile_filenae: Optional[str] = None,
    ) -> None:
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
                    "simga": sigma,
                }
            case "RDW" | "RDWA":
                self.geometry_params = {"Denisty Type": density_type, "Y": thickness}
            case "USR_SUPPLD":
                self.geometry_params = {
                    "Density Type": density_type,
                    "profile filename": profile_filenae,
                }
            case _:
                raise ValueError(
                    "Density Type must be one of [POWD, EXPD, RDW, RWDA, or USR_SUPPLD]."
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

    def set_radiation_strenght(
        self, scale_type: str, scale_value: Union[float, List[float]]
    ) -> None:
        self.scale = {"Scale": scale_type}

        match scale_type.upper():
            case "FLUX":
                self.scale["Fe"] = scale_value
            case "LUM_R1":
                self.scale["L"] = scale_value[0]
                self.scale["R1"] = scale_value[1]
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
        for comment in self.comments:
            self.output_text += f"* {comment}\n"
        self.output_text += "\n\n"

        # Geomtry, spectral shape, and scale
        self.output_text += f"GEOMETRY = {self.geometry['GEOMETRY']}\n\n"
        if self.external_radiation["central"] == "ON":
            self.output_text += f"central = {self.external_radiation['central']}\n\n"
            for key in self.spectral_shape.keys():
                self.output_text += f"{key} = {self.spectral_shape[key]}\n"
            self.output_text += "\n"
            for key in self.scale.keys():
                self.output_text += f"{key} = {self.scale[key]}\n"
            self.output_text += "\n"
            self.output_text += f"external = {self.external_radiation['external']}\n\n"
        else:
            self.output_text += f"central = {self.external_radiation['central']}\n\n"
            self.output_text += f"external = {self.external_radiation['external']}\n\n"
            for key in self.spectral_shape.keys():
                self.output_text += f"{key} = {self.spectral_shape[key]}\n"
            self.output_text += "\n"
            for key in self.scale.keys():
                self.output_text += f"{key} = {self.scale[key]}\n"
            self.output_text += "\n"

        # Dust properties
        if self.abund_user["nk_n"] > 0:
            self.output_text += (
                "optical properties index = COMMON_AND_ADDL_GRAIN_COMPOSITE\n"
            )
        else:
            self.output_text += "optical properties index = COMMON_GRAIN_COMPOSITE\n"
        for key in list(self.grains_abund.keys())[:6]:
            self.output_text += f"    {key:10}"
        self.output_text += "\n"
        self.output_text += "x = "
        for key in list(self.grains_abund.keys())[:6]:
            self.output_text += f"  {str(self.grains_abund[key]):^10}"
        self.output_text += "\n\n"

        ### other nk if specified
        if self.abund_user["nk_n"] > 0:
            self.output_text += f"Number of additional components = {self.abund_user['nk_n']}, properties listed in files\n"
            for file in self.abund_user["nk_files"]:
                self.output_text += f"    {file}\n"
            self.output_text += (
                f"Abundance for these components = "
                + ", ".join(self.abund_user["nk_abunds"])
                + "\n\n"
            )

        ### grain distribution
        for key in self.grains_params.keys():
            self.output_text += f"{key} = {self.grains_params[key]}\n"
        self.output_text += "\n"

        self.output_text += (
            f"Sublimation Temperature = {self.grains_abund['subl_temp']}\n\n"
        )

        # Density of the shell
        for key in self.geometry_params.keys():
            self.output_text += f"{key} = {self.geometry_params[key]}\n"
        self.output_text += "\n"

        self.output_text += self._tau_to_text()
        self.output_text += self._set_options_to_text()

        if to_console:
            print(self.output_text)

        with open(str(self.full_model_name) + ".inp", "+w") as o:
            o.write(self.output_text)

    def run(self) -> None:
        script = f"{self.exe_path} {self.full_model_name}.inp"
        # print("\nRunning DUSTY (v4)")
        # print(f"Model name: {self.model_name}")
        start_timer = time.time()
        proc = subprocess.Popen(script, shell=True, stdout=subprocess.PIPE, stdin=None)
        proc.communicate()
        end_timer = time.time() - start_timer
        # print(f"Ended after: {end_timer:.2f}s\n")


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
        output_data = {
            "tau0": [],
            "Ps1/Ps0": [],
            "Fi(W/m2)": [],
            "R1": [],
            "R1/Rc": [],
            "theta1": [],
            "T1(K)": [],
            "Td(K)": [],
            "RPr": [],
            "error%": [],
        }

        with open(str(self.model_name) + ".out", "r") as f:
            lines = f.readlines()
            results_index = 0
            for row in lines:
                if row.find("RESULTS") != -1:
                    results_index = lines.index(row) + 5

            for i in range(results_index, results_index + self.n_models):
                line = lines[i]

                data = line.split()
                output_data["tau0"].append(float(data[1]))
                output_data["Ps1/Ps0"].append(float(data[2]))
                output_data["Fi(W/m2)"].append(float(data[3]))
                output_data["R1"].append(float(data[4]))
                output_data["R1/Rc"].append(float(data[5]))
                output_data["theta1"].append(float(data[6]))
                output_data["T1(K)"].append(float(data[7]))
                output_data["Td(K)"].append(float(data[8]))
                output_data["RPr"].append(float(data[9]))
                output_data["error%"].append(float(data[10]))
                if len(data) > 11:
                    if "Mdot" not in output_data:
                        output_data["Mdot"] = []
                    output_data["Mdot"].append(float(data[11]))
                    if "Ve" not in output_data:
                        output_data["Ve"] = []
                    output_data["Ve"].append(float(data[12]))
                    if "M>" not in output_data:
                        output_data["M>"] = []
                    output_data["M>"].append(float(data[13]))

        return output_data

    def get_comments(self) -> str:
        return self._comments

    def get_output_data(self) -> np.recarray:
        return self._get_output_data

    def get_spectra(
        self,
        y_cont: str = "fTot",
        x_unit: str = "um",
    ) -> tuple:

        skiprows = 1
        seds = []
        for i in range(self.n_models):
            spec_filename = str(self.model_name) + f".s{i + 1:03}"
            match y_cont:
                case "fTot":
                    wave, flux = np.loadtxt(
                        spec_filename,
                        skiprows=skiprows,
                        usecols=(0, 1),
                        comments="#",
                        unpack=True,
                    )
                case "xAtt":
                    wave, flux = np.loadtxt(
                        spec_filename,
                        skiprows=skiprows,
                        usecols=(0, 2),
                        comments="#",
                        unpack=True,
                    )
                case "xDs":
                    wave, flux = np.loadtxt(
                        spec_filename,
                        skiprows=skiprows,
                        usecols=(0, 3),
                        comments="#",
                        unpack=True,
                    )
                case "xDe":
                    wave, flux = np.loadtxt(
                        spec_filename,
                        skiprows=skiprows,
                        usecols=(0, 4),
                        comments="#",
                        unpack=True,
                    )
                case "fInp":
                    wave, flux = np.loadtxt(
                        spec_filename,
                        skiprows=skiprows,
                        usecols=(0, 5),
                        comments="#",
                        unpack=True,
                    )
                case "TauTot":
                    wave, flux = np.loadtxt(
                        spec_filename,
                        skiprows=skiprows,
                        usecols=(0, 6),
                        comments="#",
                        unpack=True,
                    )
                case "albedo":
                    wave, flux = np.loadtxt(
                        spec_filename,
                        skiprows=skiprows,
                        usecols=(0, 7),
                        comments="#",
                        unpack=True,
                    )
                case _:
                    raise ValueError(
                        f"y_cont must be one of [fTot, xAtt, xDs, xDe, fInp, TauTot, albedo]."
                    )
            seds.append(flux)

        match x_unit.lower():
            case "um" | "micron":
                wave *= 1
            case "ang" | "aa" | "angstroms":
                wave *= 1e4
            case "um-1" | "micron-1":
                wave = 1 / wave
            case _:
                raise ValueError("x_unit must be one of [um, Ang, um-1].")

        return np.array(wave), np.array(seds)

    def get_fbol(self) -> List[float]:
        fbol = []
        for i in range(self.n_models):
            with open(str(self.model_name) + f".s{i+1:03}", "r") as f:
                fbol.append(float(f.readlines()[2].split("=")[1]))
        return fbol
