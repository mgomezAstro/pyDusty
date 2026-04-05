#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:22:29 2026

@author: magm
"""

import numpy as np
from scipy.stats import norm
import emcee
from .pydusty import DustyInp, DustyReader
from .dust_utils import planck_bb
from dataclasses import dataclass
from abc import abstractmethod, ABC
from pathlib import Path
import importlib.resources as pkg
from multiprocessing import Pool
from functools import partial
import os
from tempfile import TemporaryDirectory


@dataclass
class Model(ABC):

    @abstractmethod
    def compute(self, **args) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def scale_parameter(y_obs, y_obs_err, y_mod) -> float:
        pass

    @abstractmethod
    def _fn_model(**args):
        pass

    def _model(self, **args) -> callable:
        not_varied_params = {}
        for param in self.params:
            if not param.vary:
                not_varied_params[param.name] = param.value

        return partial(self._fn_model, **not_varied_params)


@dataclass
class Parameter:
    name: str
    value: float
    vary: bool
    min: float
    max: float


class Parameters:
    def __init__(self, *params: Parameter):
        self.param_dict: dict[str, Parameter] = {param.name: param for param in params}

    def add(self, param: Parameter | list | tuple) -> None:
        if isinstance(param, tuple | list):
            param = Parameter(*param)
        self.param_dict[param.name] = param

    def get_free_param_values(self):
        p = {}
        for name in self.param_dict:
            if self.param_dict[name].vary:
                p[name] = self.param_dict[name].value
        return p

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
            self.params.add(Parameter("teff", self.teff, True, 2000.0, 30000.0))
            self.params.add(Parameter("td", self.td, True, 50, 1900.0))
            self.params.add(Parameter("tau", self.tau, True, 1e-4, 100.0))
            self.params.add(Parameter("dust_abund", self.dust_abund, False, 0, 1.0))

    @property
    def _dust_types(self):
        return {
            "silicate": "Sil-DL",
            "amcarb": "amC-Hn",
            "graphite": "grf-DL",
            "alumina": "Al2O3-comp.nk",
        }

    def _fn_model(self, teff, td, tau, dust_abund, project_dir):

        dust_1 = self._dust_types[self.dust_type_1]
        dust_2 = self._dust_types[self.dust_type_2]

        predef_abunds = {dust_1: dust_abund, dust_2: 1.0 - dust_abund}
        nk_files = None
        nk_abund = None

        if self.dust_type_1 == "alumina":
            predef_abunds = {dust_2: 1.0 - dust_abund}
            nk_files = [
                str(
                    pkg.files("pydusty").joinpath("fortran/dustyV4/data/Lib_nk")
                    / dust_1
                )
            ]
            nk_abund = [dust_abund]
        if self.dust_type_2 == "alumina":
            predef_abunds = {dust_1: dust_abund}
            nk_files = [
                str(
                    pkg.files("pydusty").joinpath("fortran/dustyV4/data/Lib_nk")
                    / dust_2
                )
            ]
            nk_abund = [1.0 - dust_abund]

        inp = DustyInp(
            model_name=self.model_name,
            project_dir=str(project_dir),
        )
        inp.set_sphere(set_matrix=True)
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
            predef_abund=predef_abunds,
            subl_temp=2000.0,
            nk_files=nk_files,
            nk_abunds=nk_abund,
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

    @staticmethod
    def scale_parameter(y_obs, y_obs_err, y_mod):

        num_log_s = np.sum(y_obs * y_mod / y_obs_err**2)
        den_log_s = np.sum(y_mod**2 / y_obs_err**2)

        return num_log_s / den_log_s

    def compute(
        self,
        project_dir: str | Path = "./output",
    ):

        if isinstance(project_dir, str):
            project_dir = Path(project_dir)

        if not project_dir.exists():
            project_dir.mkdir(parents=True, exist_ok=True)

        varied_params = self.params.get_free_param_values()

        self._model()(**varied_params, project_dir=project_dir)
        mod = DustyReader(model_name=str(project_dir / self.model_name))
        wave, flux = mod.get_spectra()

        flux = flux[0]

        wave = wave[flux > 0.0]
        flux = flux[flux > 0.0]

        return wave, flux


@dataclass
class BBModel(Model):
    params: Parameters | None = None
    teff: float | None = None
    log_scale: float | None = None
    model_name: str = "bb_model"

    def __post_init__(self):
        if self.params is None:
            if any(
                [
                    self.teff is None,
                    self.radius is None,
                ]
            ):
                raise ValueError(
                    "You must specifiy either params or teff and log_scale."
                )

            self.params = Parameters()
            self.params.add(Parameter("teff", self.teff, True, 2000.0, 30000.0))
            self.params.add(Parameter("log_scale", self.radius, True, -np.inf, np.inf))
        self.wave = np.linspace(0.1, 20, 1500)

    def _fn_model(self, teff, log_scale):
        flux = np.pi * planck_bb(self.wave / 1e4, teff, output_units="lam")
        lam = self.wave * 1e4

        shape_flux = lam * flux / np.trapezoid(flux, lam)

        return shape_flux * 10**log_scale

    @staticmethod
    def scale_parameter(y_obs, y_obs_err, y_mod):
        return 1.0

    def compute(
        self,
        project_dir: str | Path = "./output",
    ):

        varied_params = self.params.get_free_param_values()

        flux = self._model()(**varied_params)

        np.savetxt(
            project_dir / f"{self.model_name}_spectrum.txt",
            np.column_stack((self.wave, flux)),
        )

        return self.wave, flux


@dataclass
class EmceeRunner:
    x_obs: np.ndarray
    y_obs: np.ndarray
    y_err: np.ndarray
    model: Model
    params: Parameters
    mask_uplims: np.ndarray | None = None
    kargs_model: dict | None = None

    def __post_init__(self):
        if self.mask_uplims is None:
            self.mask_uplims = np.zeros_like(self.x_obs, dtype=bool)
            self._names_varied = [param.name for param in self.params if param.vary]
            self._tmp_models_path = None

    def sample_prior(self, chains):
        stack = []
        for param in self.params:
            if param.vary:
                stack.append(np.random.uniform(param.min, param.max, chains))
        return np.column_stack(stack)

    def log_prior(self):
        log_prior = 0.0
        for param in self.params:
            if param.vary:
                if not param.min < param.value < param.max:
                    return -np.inf
        return log_prior

    def log_prob(self, theta):

        for k, val in enumerate(theta):
            self.params[self._names_varied[k]].value = val

        lg_prior = self.log_prior()
        if not np.isfinite(lg_prior):
            return -np.inf, -np.inf

        if self.kargs_model is not None:
            model = partial(self.model, **self.kargs_model)(params=self.params)
        else:
            model = self.model(params=self.params)

        worker_id = os.getpid()
        scratch_dir = self._tmp_models_path / f"worker_{worker_id}"
        wave, flux = model.compute(project_dir=scratch_dir)

        flux = np.interp(self.x_obs, wave, flux)

        no_limits = np.invert(self.mask_uplims)
        sigma2 = self.y_err[no_limits] ** 2
        scale = self.model.scale_parameter(
            y_obs=self.y_obs[no_limits],
            y_obs_err=self.y_err[no_limits],
            y_mod=flux[no_limits],
        )

        chi2 = -0.5 * np.sum(
            (self.y_obs[no_limits] - flux[no_limits] * scale) ** 2 / sigma2
        )

        if self.mask_uplims.sum() > 0:
            for fl_obs, fl_mod, fl_obs_err in zip(
                self.y_obs[self.mask_uplims],
                flux[self.mask_uplims],
                self.y_err[self.mask_uplims],
            ):
                chi2 += norm.logcdf(fl_obs, loc=fl_mod, scale=fl_obs_err)

        return chi2, np.log10(scale)

    def run(
        self,
        continue_from_last: bool = False,
        suffix: str = "_1",
        n_proc: int = 1,
        steps: int = 1000,
        chains: int = 32,
        p0: list | None = None,
    ):

        init_positions = None
        backend = emcee.backends.HDFBackend(f"emcee_{suffix}.h5")
        ndim = len(self.params.get_free_param_values())
        if not continue_from_last:
            init_positions = p0
            if p0 is None:
                print("Not initial positions set. Using uniform initial values.")
                init_positions = self.sample_prior(chains)
                print(init_positions)
            backend.reset(chains, ndim)

        with TemporaryDirectory(dir="/dev/shm/", prefix="emcee_runs_") as tmpdir:

            self._tmp_models_path = Path(tmpdir)

            if n_proc > 1:
                with Pool(n_proc) as pool:

                    sampler = emcee.EnsembleSampler(
                        chains,
                        ndim,
                        self.log_prob,
                        backend=backend,
                        pool=pool,
                    )

                    sampler.run_mcmc(init_positions, steps, progress=True)
            else:
                sampler = emcee.EnsembleSampler(
                    chains,
                    ndim,
                    self.log_prob,
                    backend=backend,
                )

                sampler.run_mcmc(init_positions, steps, progress=True)

        return sampler
