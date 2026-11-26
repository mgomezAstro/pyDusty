#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:22:29 2026

@author: magm
"""

import importlib.resources as pkg
import inspect
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory

import emcee
import numpy as np
from scipy.stats import norm

from .dust_utils import distance_to_cm as _distance_to_cm
from .dust_utils import planck_bb as _planck_bb
from .dust_utils import thermal_emission as _thermal_emission
from .pydusty import MAX_TAU_MODELS
from .pydusty import DustyError
from .pydusty import DustyInp as _dustyinp
from .pydusty import DustyReader as _dustyreader

logger = logging.getLogger(__name__)


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
    """A model parameter.

    ``init_scale`` ("linear" or "log") only controls how
    :meth:`EmceeRunner.sample_prior` draws the walkers' *starting* positions
    when no ``p0`` is given: uniformly in the value, or uniformly in log10 of
    the value (requires ``min > 0``). It does not change the prior used in
    the fit, which stays uniform in the value between ``min`` and ``max``
    (see :meth:`EmceeRunner.log_prior`).
    """

    name: str
    value: float
    vary: bool
    min: float
    max: float
    init_scale: str = "linear"


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
            self.params.add(
                Parameter("tau", self.tau, True, 1e-4, 100.0, init_scale="log")
            )
            self.params.add(Parameter("dust_abund", self.dust_abund, False, 0, 1.0))

    @property
    def _dust_types(self):
        return {
            "silicate": "Sil-DL",
            "amcarb": "amC-Hn",
            "graphite": "grf-DL",
            "alumina": "Al2O3-comp.nk",
        }

    def _make_inp(self, teff, td, dust_abund, project_dir, omp_threads=None):
        """DustyInp with everything set except the optical depth."""
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

        inp = _dustyinp(
            model_name=self.model_name,
            project_dir=str(project_dir),
            omp_threads=omp_threads,
        )
        inp.set_sphere(set_matrix=False)
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
        return inp

    def _fn_model(
        self, teff, td, tau, dust_abund, project_dir, *args, omp_threads=None, **kwargs
    ):
        inp = self._make_inp(teff, td, dust_abund, project_dir, omp_threads)
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

    @staticmethod
    def denormalize_radius(rin_model: float, l_bol: float):
        return rin_model + 0.5 * (l_bol - 4.0)

    def compute(
        self,
        project_dir: str | Path = "./output",
        omp_threads: int | None = None,
    ):
        """Run DUSTY for the current parameters; return ``(wave, flux)``.

        ``omp_threads`` sets OMP_NUM_THREADS for DUSTY (``None``: inherit
        the environment).
        """

        if isinstance(project_dir, str):
            project_dir = Path(project_dir)

        if not project_dir.exists():
            project_dir.mkdir(parents=True, exist_ok=True)

        varied_params = self.params.get_free_param_values()

        self._model()(
            **varied_params, project_dir=project_dir, omp_threads=omp_threads
        )
        mod = _dustyreader(model_name=str(project_dir / self.model_name))
        wave, flux = mod.get_spectra()

        flux = flux[0]

        wave = wave[flux > 0.0]
        flux = flux[flux > 0.0]

        return wave, flux

    def compute_grid(
        self,
        taus,
        project_dir: str | Path = "./output",
        omp_threads: int | None = None,
    ):
        """Compute many optical depths in a single DUSTY run.

        Uses the current ``teff``, ``td`` and ``dust_abund`` values of
        ``self.params`` (the ``tau`` parameter is ignored) and DUSTY's
        USER_SUPPLIED tau grid, so ``taus`` can be any list of positive
        values (at most 999, DUSTY's limit).

        Returns ``(wave, fluxes, output_data)``: ``fluxes`` has shape
        ``(len(taus), len(wave))`` and ``output_data`` is the RESULTS table
        (see :meth:`DustyReader.get_output_data`), both in the order of
        ``taus`` as given. Unlike :meth:`compute`, points with non-positive
        flux are not removed, since they can differ between models.

        DUSTY solves the models in increasing tau, starting each one from
        the previous solution, so the spectra agree with separate
        :meth:`compute` calls only to within DUSTY's numerical accuracy
        (the smallest tau is identical; for the others ~1-2 % where the
        flux is above 1e-3 of the peak, more in the faint far wings), not
        bit for bit. This pays off for closely spaced taus (e.g. 10 taus
        in 0.1-5: ~2x faster than 10 :meth:`compute` calls); for widely
        spaced, large taus (0.02-50) it was ~1.5x slower. Intended for
        model grids; :class:`EmceeRunner` does not use it.
        """
        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        if taus.ndim != 1 or not 1 <= taus.size <= MAX_TAU_MODELS:
            raise ValueError(
                f"'taus' must be a 1-D list of 1 to {MAX_TAU_MODELS} values."
            )

        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        values = {param.name: param.value for param in self.params}
        inp = self._make_inp(
            values["teff"], values["td"], values["dust_abund"], project_dir,
            omp_threads,
        )
        inp.set_optical_depth(tau_grid="USER_SUPPLIED", lambda0=0.554, taus=taus)
        inp.print_inp_file()
        inp.run()

        name = project_dir / self.model_name
        missing = [
            f"{self.model_name}.s{i + 1:03}"
            for i in range(taus.size)
            if not Path(f"{name}.s{i + 1:03}").is_file()
        ]
        if missing:
            raise DustyError(
                f"DUSTY did not produce all models of the tau grid in "
                f"{project_dir} (missing: {', '.join(missing)})."
            )

        reader = _dustyreader(model_name=str(name))
        wave, fluxes = reader.get_spectra()
        out = reader.get_output_data()

        # DUSTY ran the taus sorted (stable for ties); undo that.
        order = np.argsort(taus, kind="stable")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        fluxes = fluxes[inverse]
        out = {
            key: [vals[i] for i in inverse] if len(vals) == taus.size else vals
            for key, vals in out.items()
        }
        return wave, fluxes, out


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
                    self.log_scale is None,
                ]
            ):
                raise ValueError(
                    "You must specifiy either params or teff and log_scale."
                )

            self.params = Parameters()
            self.params.add(Parameter("teff", self.teff, True, 2000.0, 30000.0))
            self.params.add(
                Parameter("log_scale", self.log_scale, True, -np.inf, np.inf)
            )
        self.wave = np.linspace(0.1, 20, 1500)

    def _fn_model(self, teff, log_scale, *args, **kwargs):
        flux = np.pi * _planck_bb(self.wave / 1e4, teff, output_units="lam")
        lam = self.wave * 1e4

        shape_flux = lam * flux / np.trapezoid(flux, lam)

        return shape_flux * 10**log_scale

    @staticmethod
    def scale_parameter(y_obs, y_obs_err, y_mod):
        return 1.0

    def compute(
        self,
        project_dir: str | Path = "./output",
        save_spectrum: bool = True,
    ):
        """Return ``(wave, flux)``; with ``save_spectrum`` also write
        ``<model_name>_spectrum.txt`` in ``project_dir``."""

        varied_params = self.params.get_free_param_values()

        flux = self._model()(**varied_params)

        if save_spectrum:
            if isinstance(project_dir, str):
                project_dir = Path(project_dir)

            if not project_dir.exists():
                project_dir.mkdir(parents=True, exist_ok=True)

            np.savetxt(
                project_dir / f"{self.model_name}_spectrum.txt",
                np.column_stack((self.wave, flux)),
            )

        return self.wave, flux


@dataclass
class ThermalEmissionModel(Model):
    params: Parameters | None = None
    teff: float | None = None
    td: float | None = None
    dust_mass: float | None = None
    distance: float = 1.0
    log_radius: float = 14.0
    distance_unit: str = "Mpc"
    a: float = 0.1
    dusty_type: str = "silicate"
    unit: str = "nuLnu"
    model_name: str = "thermal_emission_model"

    def __post_init__(self):
        if self.params is None:
            if any(
                [
                    self.teff is None,
                    self.td is None,
                    self.dust_mass is None,
                    self.log_radius is None,
                ]
            ):
                raise ValueError(
                    "You must specifiy either params or teff, td, dust_mass and log_radius."
                )

            self.params = Parameters()
            self.params.add(Parameter("teff", self.teff, True, 2000.0, 30000.0))
            self.params.add(Parameter("td", self.td, True, 50, 1900.0))
            self.params.add(Parameter("log_radius", self.log_radius, True, 13.0, 18.0))
            self.params.add(Parameter("dust_mass", self.dust_mass, True, -9, 0))

        self.wave = np.linspace(0.15, 20, 1500)

    @staticmethod
    def scale_parameter(y_obs, y_obs_err, y_mod):
        return 1.0

    def _fn_model(self, teff, td, dust_mass, log_radius, *args, **kwargs):
        d_cm = _distance_to_cm(self.distance, self.distance_unit)

        bb_flux = np.pi * _planck_bb(self.wave / 1e4, teff, output_units="nu")
        bb_flux *= (10**log_radius / d_cm) ** 2

        ir_flux = _thermal_emission(
            wave=self.wave,
            temperature=td,
            distance=self.distance,
            distance_unit=self.distance_unit,
            dust_mass=10**dust_mass,
            a=self.a,
            grain_type=self.dusty_type,
        )

        total_flux = (bb_flux + ir_flux) * (2.99792458e14 / self.wave)

        if self.unit == "nuLnu":
            total_flux *= (
                4 * np.pi * d_cm**2 / 3.828e33
            )  # in solar units

        return total_flux

    def compute(
        self,
        project_dir: str | Path = "./output",
        save_spectrum: bool = True,
    ):
        """Return ``(wave, flux)``; with ``save_spectrum`` also write
        ``<model_name>_spectrum.txt`` in ``project_dir``."""

        varied_params = self.params.get_free_param_values()

        flux = self._model()(**varied_params)

        if save_spectrum:
            if isinstance(project_dir, str):
                project_dir = Path(project_dir)

            if not project_dir.exists():
                project_dir.mkdir(parents=True, exist_ok=True)

            np.savetxt(
                project_dir / f"{self.model_name}_spectrum.txt",
                np.column_stack((self.wave, flux)),
            )

        return self.wave, flux


# Runner used by the log-probability function in Pool workers; set once per
# worker by _init_worker so that emcee only pickles theta for each task.
_WORKER_RUNNER = None


def _init_worker(runner):
    global _WORKER_RUNNER
    _WORKER_RUNNER = runner
    runner._scratch_dir()


def _worker_log_prob(theta):
    return _WORKER_RUNNER.log_prob(theta)


def _resolve_n_proc(n_proc) -> int:
    """``-1`` or ``"auto"`` -> os.cpu_count(); otherwise ``max(1, n_proc)``."""
    if isinstance(n_proc, str):
        if n_proc.lower() != "auto":
            raise ValueError(f"n_proc must be an int >= 1, -1 or 'auto' (got {n_proc!r}).")
        return os.cpu_count() or 1
    n_proc = int(n_proc)
    if n_proc == -1:
        return os.cpu_count() or 1
    # as before, any other value < 2 runs serially
    return max(1, n_proc)


@dataclass
class EmceeRunner:
    x_obs: np.ndarray
    y_obs: np.ndarray
    y_err: np.ndarray
    model: Model
    params: Parameters
    mask_uplims: np.ndarray | None = None
    vel_constrain: list[float] | None = None
    denorm_rin: bool = False
    kargs_model: dict | None = None

    def __post_init__(self):
        if self.mask_uplims is None:
            self.mask_uplims = np.zeros_like(self.x_obs, dtype=bool)
        self._names_varied = [param.name for param in self.params if param.vary]
        self._tmp_models_path = None
        self._omp_threads = None
        self._scratch = None
        self._model_compute_kwargs = None

    def _scratch_dir(self) -> Path:
        """This process's scratch directory, created once per process."""
        pid = os.getpid()
        if self._scratch is None or self._scratch[0] != pid:
            path = self._tmp_models_path / f"worker_{pid}"
            path.mkdir(parents=True, exist_ok=True)
            self._scratch = (pid, path)
        return self._scratch[1]

    def _compute_kwargs(self) -> dict:
        """Extra keyword arguments for ``model.compute`` in the fit.

        Only those the model's ``compute`` accepts are passed, so models
        with the plain ``compute(project_dir)`` signature keep working.
        """
        if self._model_compute_kwargs is None:
            try:
                accepted = inspect.signature(self.model.compute).parameters
            except (TypeError, ValueError):
                accepted = {}
            kwargs = {}
            if "save_spectrum" in accepted:
                kwargs["save_spectrum"] = False
            if "omp_threads" in accepted and self._omp_threads is not None:
                kwargs["omp_threads"] = self._omp_threads
            self._model_compute_kwargs = kwargs
        return self._model_compute_kwargs

    def sample_prior(self, chains: int, seed: int | None = None) -> np.ndarray:
        """Draw ``chains`` starting positions for the free parameters.

        Uses a local ``np.random.Generator`` seeded with ``seed`` (``None``:
        fresh OS entropy), so the global NumPy RNG is not touched.

        For each free parameter:

        * finite bounds, ``init_scale="linear"``: uniform in [min, max];
        * finite bounds, ``init_scale="log"``: uniform in log10 between
          min and max (requires ``min > 0``);
        * a non-finite bound (e.g. ``-inf..inf``): a small Gaussian ball
          around ``param.value`` (sigma = 1e-2 * max(1, |value|)), clipped
          to lie strictly inside the bounds.

        This only sets the walkers' starting positions; the prior used in the
        fit (:meth:`log_prior`) is unchanged.
        """
        rng = np.random.default_rng(seed)
        stack = []
        for param in self.params:
            if not param.vary:
                continue
            lo, hi = float(param.min), float(param.max)
            if np.isfinite(lo) and np.isfinite(hi):
                if param.init_scale == "log":
                    if lo <= 0:
                        raise ValueError(
                            f"Parameter '{param.name}': init_scale='log' "
                            f"requires min > 0 (got min={lo})."
                        )
                    col = 10 ** rng.uniform(np.log10(lo), np.log10(hi), chains)
                elif param.init_scale == "linear":
                    col = rng.uniform(lo, hi, chains)
                else:
                    raise ValueError(
                        f"Parameter '{param.name}': init_scale must be "
                        f"'linear' or 'log' (got {param.init_scale!r})."
                    )
            else:
                value = param.value
                if value is None or not np.isfinite(value):
                    raise ValueError(
                        f"Parameter '{param.name}' has non-finite bounds "
                        f"[{lo}, {hi}] and no finite value to start from; "
                        "pass initial positions with p0."
                    )
                sigma = 1e-2 * max(1.0, abs(value))
                col = value + sigma * rng.standard_normal(chains)
                # keep the starts strictly inside the (half-open) bounds
                col = np.clip(
                    col, np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf)
                )
            # uniform() may return the lower bound; log_prior is strict
            col = np.where(col <= lo, np.nextafter(lo, np.inf), col)
            col = np.where(col >= hi, np.nextafter(hi, -np.inf), col)
            stack.append(col)
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
            return -np.inf, -np.inf, -np.inf

        if self.kargs_model is not None:
            model = partial(self.model, **self.kargs_model)(params=self.params)
        else:
            model = self.model(params=self.params)

        scratch_dir = self._scratch_dir()
        try:
            wave, flux = model.compute(
                project_dir=scratch_dir, **self._compute_kwargs()
            )
        except DustyError:
            return -np.inf, -np.inf, -np.inf

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

        if not np.isfinite(scale) or scale <= 0:
            return -np.inf, -np.inf, -np.inf

        if self.mask_uplims.sum() > 0:
            m = self.mask_uplims
            chi2 += norm.logcdf(
                self.y_obs[m], loc=flux[m] * scale, scale=self.y_err[m]
            ).sum()

        rin_denorm = 0.0
        if self.denorm_rin or self.vel_constrain is not None:
            mod_table = _dustyreader(
                model_name=str(scratch_dir / model.model_name)
            ).get_output_data()
            rin_model = np.log10(float(mod_table["R1"][0]))
            rin_denorm = model.denormalize_radius(rin_model, np.log10(scale))

        if self.vel_constrain is not None:
            # calculate v_rad from the model
            r_out = rin_denorm + np.log10(model.thickness)
            vlog = r_out - np.log10(8.64e9 * self.vel_constrain[1])
            chi2 += -0.5 * ((vlog - np.log10(self.vel_constrain[0])) / 0.3) ** 2

        return chi2, np.log10(scale), rin_denorm

    def run(
        self,
        continue_from_last: bool = False,
        suffix: str = "_1",
        n_proc: int | str = 1,
        steps: int = 1000,
        chains: int = 32,
        p0: list | None = None,
        seed: int | None = None,
        omp_threads: int | None = None,
    ):
        """Run the sampler.

        ``seed`` is passed to :meth:`sample_prior` when ``p0`` is not given
        (it does not seed emcee's own moves).

        ``n_proc``: number of worker processes (``-1`` or ``"auto"``: one per
        CPU).

        ``omp_threads``: OpenMP threads for each DUSTY run (models whose
        ``compute`` accepts ``omp_threads``, i.e. :class:`DustyModel`).
        ``None`` (default) inherits ``OMP_NUM_THREADS`` from the environment.
        It is not limited automatically: emcee evaluates the walkers in
        synchronous batches, so one slow (large-tau) model sets the pace,
        and that model runs ~4x faster with 12 threads than with 1.
        ``omp_threads=1`` also makes DUSTY runs bit-reproducible: with
        several threads the DUSTY binary occasionally returns slightly
        different spectra for the same input.
        """
        n_proc = _resolve_n_proc(n_proc)
        self._omp_threads = omp_threads
        self._model_compute_kwargs = None
        self._scratch = None

        init_positions = None
        backend = emcee.backends.HDFBackend(f"emcee_{suffix}.h5")
        ndim = len(self.params.get_free_param_values())
        if not continue_from_last:
            init_positions = p0
            if p0 is None:
                logger.info(
                    "Not initial positions set. Using uniform initial values."
                )
                init_positions = self.sample_prior(chains, seed=seed)
                logger.info("Initial positions:\n%s", init_positions)
            backend.reset(chains, ndim)

        with TemporaryDirectory(prefix="emcee_runs_") as tmpdir:
            self._tmp_models_path = Path(tmpdir)

            if n_proc > 1:
                with Pool(
                    n_proc, initializer=_init_worker, initargs=(self,)
                ) as pool:
                    sampler = emcee.EnsembleSampler(
                        chains,
                        ndim,
                        _worker_log_prob,
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
