#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:22:29 2026

@author: magm
"""

import numpy as np
import emcee
from .pydusty import DustyInp, DustyReader
from dataclasses import dataclass
from abc import abstractmethod, ABC
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import tempfile


@dataclass
class Model(ABC):

    @abstractmethod
    def compute(self, **args) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def scale_parameter(**args) -> float:
        return 1.0

    @abstractmethod
    def _model(self, **args) -> callable:
        pass


@dataclass
class Parameter:
    name: str
    value: float
    vary: bool
    min: float
    max: float


class Parameters:
    def __init__(self, *params: Parameter):
        self.param_dict: dict[str, Parameter] = {
            param.name: param for param in params
        }

    def add(self, param: Parameter | list | tuple) -> None:
        if isinstance(param, tuple | list):
            param = Parameter(*param)
        self.param_dict[param.name] = param

    def __getitem__(self, name: str) -> Parameter:
        return self.param_dict[name]

    def __iter__(self):
        return iter(self.param_dict.values())

    def __len__(self) -> int:
        return len(self.param_dict)

    def __repr__(self) -> str:
        return f"Parameters({', '.join(repr(param) for param in self.param_dict.values())})"


@dataclass
class DustyModel(Model):

    params: Parameters | None = None
    teff: float | None = None
    td: float | None = None
    tau: float | None = None
    dust_abund: float | None = None
    dust_type_1: str = "silicate"
    dust_type_2: str = "amcarb"
    n_power_law: int = 1
    exponent_power_law: float | list = 2
    thickness: float = 2.0
    model_name: str = "model"

    def __post_init__(self):
        if self.params is None:
            if any(
                [
                    self.teff is None,
                    self.td is None,
                    self.tau is None,
                    self.dust_abund is None,
                ]
            ):
                raise ValueError(
                    "You must specifiy either params or teff, td, tau, and dust_abund."
                )

            self.params = Parameters()
            self.params.add(
                Parameter("teff", self.teff, True, 2000.0, 30000.0)
            )
            self.params.add(Parameter("td", self.td, True, 50, 1900.0))
            self.params.add(Parameter("tau", self.tau, True, 1e-4, 100.0))
            self.params.add(
                Parameter("dust_abund", self.dust_abund, False, 0, 1.0)
            )

    @property
    def _dust_types(self):
        return {
            "silicate": "Sil-DL",
            "amcarb": "amC-Hn",
            "graphite": "grf-DL",
        }

    def _dust_model(self, teff, td, tau, dust_abund, project_dir):

        dust_1 = self._dust_types[self.dust_type_1]
        dust_2 = self._dust_types[self.dust_type_2]

        inp = DustyInp(
            model_name=self.model_name,
            project_dir=str(project_dir),
        )
        inp.set_sphere()
        inp.set_blackbody(temperature=teff)
        inp.set_central_radiation(True)
        inp.set_density_profile(
            density_type="POWD",
            n_pwd=self.n_power_law,
            thickness=self.thickness,
            p=self.exponent_power_law,
        )
        inp.set_grain_size_dist(grain_distribution="MRN")
        inp.set_grains_abund(
            predef_abund={dust_1: dust_abund, dust_2: 1.0 - dust_abund},
            subl_temp=2000.0,
        )
        inp.set_radiation_strenght(scale_type="T1", scale_value=td)
        inp.set_optical_depth(
            tau_grid="LINEAR",
            lambda0=0.554,
            taumin=tau,
            taumax=tau,
            n_models=1,
        )
        inp.print_inp_file()
        inp.run()

    def _model(self):

        not_varied_params = {}
        for param in self.params:
            if not param.vary:
                not_varied_params[param.name] = param.value

        return partial(self._dust_model, **not_varied_params)

    @staticmethod
    def scale_parameter(y_obs, y_obs_err, y_mod):

        num_log_s = np.sum(y_obs * y_mod / y_obs_err**2)
        den_log_s = np.sum(y_mod**2 / y_obs_err**2)

        return num_log_s / den_log_s

    def compute(self, project_dir: str | Path = "./output"):

        if isinstance(project_dir, str):
            project_dir = Path(project_dir)

        varied_params = {}
        for param in self.params:
            if param.vary:
                varied_params[param.name] = param.value

        self._model()(**varied_params, project_dir=project_dir)

        mod = DustyReader(model_name=str(project_dir / self.model_name))
        wave, flux = mod.get_spectra()
        # data_output = mod.get_output_data()
        flux = flux[0]

        wave = wave[flux > 0.0]
        flux = flux[flux > 0.0]

        return wave, flux


@dataclass
class EmceeRunner:
    x_obs: np.ndarray
    y_obs: np.ndarray
    y_err: np.ndarray
    model: Model
    params: Parameters

    continue_from_last: bool = False
    suffix: str = "_1"
    n_proc: int = 1
    steps: int = 1000
    chains: int = 32

    def sample_prior(self):
        stack = []
        for param in self.params:
            if param.vary:
                stack.append(
                    np.random.uniform(param.min, param.max, self.chains)
                )
        return np.column_stack(stack)

    def log_prior(self):
        log_prior = 0.0
        for param in self.params:
            if param.vary:
                if not param.min < param.value < param.max:
                    return -np.inf
        return log_prior

    def log_prob(self, theta):

        k = 0
        for param in self.params:
            if param.vary:
                self.params[param.name].value = theta[k]
                k += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            model = self.model(self.params)
            wave, flux = model.compute(project_dir=project_dir)

        flux = np.interp(self.x_obs, wave, flux)

        sigma2 = self.y_err**2
        scale = self.model.scale_parameter(
            y_obs=self.y_obs, y_obs_err=self.y_err, y_mod=flux
        )

        chi2 = np.sum((self.y_obs - flux * scale) ** 2 / sigma2)

        return -0.5 * chi2, np.log10(scale)

    def run(self):

        init_positions = None
        if not self.continue_from_last:
            init_positions = self.sample_prior()

        with emcee.backends.HDFBackend(f"emcee_{self.suffix}.h5") as backend:
            if not self.continue_from_last:
                backend.reset(self.chains, len(init_positions[0]))

            with Pool(self.n_proc) as pool:

                sampler = emcee.EnsembleSampler(
                    self.chains,
                    len(init_positions[0]),
                    self.log_prob,
                    backend=backend,
                    pool=pool,
                )

                sampler.run_mcmc(init_positions, self.steps, progress=True)

        return sampler
