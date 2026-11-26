[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24194551.svg)](https://doi.org/10.5281/zenodo.24194551)

# pyDusty
Python wrapper for the radiative transfer code DUSTY (v4) (2000ASPC..196...77N). The original
repository is at  [ivezic](https://github.com/ivezic/dusty). The modified version for this 
python wrapper is here [dusty](https://github.com/mgomezAstro/dusty).

### Requirements

- python >= 3.10
- gfortran
- openmp support (for now is not optional; it will be compiled with `-fopenmp` flag)

A version of DUSTY (V4) is compiled with `gfortran` when the package is installed and placed inside the package
directory (pydusty/bin/dusty), where it is used as default.

### Installation

`pip install git+https://github.com/mgomezAstro/pyDusty.git`

or, from a clone of the repository, `pip install .` (or `pip install -e .` for an editable install).
The installation stops with an error if `gfortran` is not found; set the `FC` environment variable to use another
gfortran-compatible compiler. If the DUSTY executable is missing (e.g. it was deleted), running a model raises a
`DustyError`: reinstall pydusty to compile it again.


### Example

#### Running a simple DUSTY model

The high-level interface in `pydusty.models` builds the DUSTY input file, runs the code and
returns the resulting spectrum:

```python
from pydusty.models import DustyModel


teff = 2500
td = 250
tau = 10.0

mod = DustyModel(
    teff=teff,
    td=td,
    tau=tau,
    dust_abund=0.7,
    model_name=f"sphere_{teff}_{td}",
)
wave, flux = mod.compute(project_dir="./output")
```

`DustyModel` assumes a spherical shell with an MRN grain-size distribution, a blackbody
central source of temperature `teff`, and a power-law density profile. The main parameters are:

- `teff`: central source temperature in K.
- `td`: dust temperature in K (used to scale the radiation field).
- `tau`: optical depth at 0.554 micron.
- `dust_abund`: abundance of `dust_type_1`; the remaining fraction is `dust_type_2`.
- `dust_type_1` / `dust_type_2`: one of `"silicate"`, `"amcarb"`, `"graphite"`, or `"alumina"`.
  Defaults are `"silicate"` and `"amcarb"`.
- `n_power_law`, `exponent_power_law`, `thickness`: density profile parameters.
- `model_name`: base name of the DUSTY output files (default `"model"`).

`compute` returns `(wave, flux)` and writes the DUSTY output files to `project_dir`
(default `"./output"`).

#### Grids of optical depths

`DustyModel.compute_grid` runs many optical depths in a single DUSTY call, using the `teff`, `td`
and `dust_abund` of the model (its `tau` is ignored):

```python
import numpy as np

from pydusty.models import DustyModel


mod = DustyModel(teff=2500, td=250, tau=1.0, dust_abund=0.7, model_name="sphere_grid")
taus = np.linspace(0.1, 5.0, 10)
wave, fluxes, out = mod.compute_grid(taus, project_dir="./output")
# fluxes[i] is the spectrum for taus[i]; out is the RESULTS table of the .out file.
```

It returns `(wave, fluxes, output_data)`: `fluxes` has shape `(len(taus), len(wave))` and
`output_data` is the same dictionary as `DustyReader.get_output_data()`, both in the order of
`taus` as given (at most 999 values). Unlike `compute`, points with non-positive flux are not
removed.

DUSTY solves the models in increasing `tau`, starting each one from the previous solution. This is
faster for closely spaced optical depths (about 2x for 10 values between 0.1 and 5), but it can be
slower for widely spaced, large ones. The spectra agree with separate `compute` calls to about
1-2% (where the flux is above 1e-3 of the peak; more in the faint far wings), not bit for bit.
`EmceeRunner` does not use it.

#### Fitting a DUSTY model with `EmceeRunner`

`EmceeRunner` fits a model to observed data with `emcee`. The free parameters are defined with
`Parameters`, and the model class is passed via `model` (not an instance):

```python
import numpy as np

from pydusty.models import DustyModel, EmceeRunner, Parameters


# wave, nulnu and e_nulnu are the observed wavelengths, fluxes and uncertainties.
# mask_limits is a boolean array marking upper limits (use None if there are none).

params = Parameters()
params.add(("teff", 3000.0, True, 1900.0, 10000.0))
params.add(("td", 1400.0, True, 500.0, 1800.0))
params.add(("tau", 20.0, True, 1e-3, 40.0))
params.add(("dust_abund", 1.0, False, 0.0, 1.0))

# Initial positions: one row per walker, one column per free parameter.
p0 = np.array([p.value for p in params if p.vary])
p0 = p0 + [1e2, 1e1, 1e-1] * np.random.randn(6, 3)

mcmc = EmceeRunner(
    x_obs=wave,
    y_obs=nulnu,
    y_err=e_nulnu,
    model=DustyModel,
    params=params,
    mask_uplims=mask_limits,
    # kargs_model={"dust_type_2": "alumina"},
)
sampler = mcmc.run(
    suffix="_dust",
    n_proc=10,
    steps=1000,
    chains=6,
    p0=p0, #optional [None]
    # continue_from_last=True,
)
```

The `Parameters` tuple is `(name, value, vary, min, max)`. Extra fixed `DustyModel`
arguments (for example `dust_type_1`, `dust_type_2`, or `thickness`) can be passed through
`kargs_model`. The chains are saved to `emcee_<suffix>.h5`; set `continue_from_last=True`
to resume an existing run. If `p0` is not given, the initial positions are drawn uniformly
within the parameter limits; pass `seed=` to `run` to make them reproducible (it does not
seed emcee's own moves).

#### Parallel runs and reproducibility

- `n_proc`: number of worker processes used to evaluate the walkers (`n_proc=-1` or
  `n_proc="auto"` uses one per CPU). This pays off for `DustyModel`, where each evaluation is a
  DUSTY run. For the fast analytic models (`BBModel`, `ThermalEmissionModel`) the overhead of
  the processes is larger than the model itself, so `n_proc=1` (default) is usually faster.
- `omp_threads`: number of OpenMP threads of each DUSTY run (sets `OMP_NUM_THREADS` for the
  DUSTY process). The default `None` inherits the environment. It is not limited
  automatically: emcee evaluates the walkers in batches, so the slowest (large `tau`) model sets
  the pace, and that model runs much faster with several threads. `omp_threads` can also be
  given to `DustyModel.compute`, `DustyModel.compute_grid` and `DustyInp`.
- With more than one thread, the DUSTY binary occasionally returns slightly different spectra
  for the same input. Use `omp_threads=1` if you need bit-for-bit reproducible results.

```python
sampler = mcmc.run(suffix="_dust", n_proc="auto", steps=1000, chains=6, p0=p0, omp_threads=1)
```

#### Other models

`pydusty.models` also provides:

```python
from pydusty.models import BBModel, ThermalEmissionModel


bb = BBModel(teff=2500, log_scale=12.0)
wave, flux = bb.compute(project_dir="./output")

th = ThermalEmissionModel(
    teff=2500,
    td=250,
    dust_mass=-4.0,  # log10(Mdust / Msun)
    log_radius=14.0,
    distance=1.0,
    distance_unit="Mpc",
)
wave, flux = th.compute(project_dir="./output")
```

`EmceeRunner` can be used to fit any of these models to observed photometry/spectroscopy
with `emcee`. See its docstring for the required arguments and the `run` options.

#### Custom DUSTY input (advanced)

For full control over the DUSTY input file, use `DustyInp` directly:

```python
import importlib.resources as pkg

import pydusty


teff = 2500
td = 250
model_name = f"sphere_{teff}_{td}"

nk_dir = pkg.files("pydusty") / "fortran/dustyV4/data/Lib_nk"  # Optical constants shipped with DUSTY.

mod = pydusty.DustyInp(model_name=model_name, project_dir="./output")
mod.options["r"] = 0 # No radial profile output (see dusty manual).
mod.options["m"] = 1 # Messages in one file (see dusty manual).
mod.options["flux conservation"] = 0.1 # Flux conservation at 10% max error.
mod.set_sphere(set_matrix=False)
mod.set_blackbody(temperature=teff)
mod.set_central_radiation(central=True)
mod.set_density_profile(density_type="POWD", n_pwd=1, thickness=1000, p=2)
mod.set_grain_size_dist(grain_distribution="MRN", amin=0.005, amax=0.25, q=3.5)
mod.set_grains_abund(
    predef_abund={"Sil-DL": 0.7},
    nk_files=[str(nk_dir / "amC-zb1.nk")],  # Or the path to your own .nk file.
    nk_abunds=[0.3],
)
mod.set_radiation_strenght(scale_type="T1", scale_value=td)
mod.set_optical_depth(
    tau_grid="LINEAR", lambda0=0.554, taumin=5, taumax=150, n_models=6
)
mod.set_comments(
    [
        "This is a model for the test object.",
        f"Main parameters: Teff--{teff}, and Td--{td}.",
        "Grains of Silicates (Sil-DL) was set to 0.7 and aC to 0.3.",
    ]
)
mod.print_inp_file()
mod.run()
```

#### Reading the output

To read DUSTY output files directly:

```python
from pathlib import Path
import pydusty


model_name = Path("./output/sphere_2500_250")

mod = pydusty.DustyReader(model_name)
wave, spec = mod.get_spectra()
ot = mod.get_output_data() # A dictionary with the output table in the .out file.
```

See the docstrings for other methods included in pydusty.

#### Errors

If DUSTY fails (it exits with an error or does not write its output files) or its executable is
missing, pydusty raises `pydusty.DustyError` (a subclass of `RuntimeError`). The message includes
the last lines of the DUSTY output:

```python
import pydusty
from pydusty.models import DustyModel


try:
    wave, flux = DustyModel(teff=2500, td=250, tau=10.0, dust_abund=0.7).compute()
except pydusty.DustyError as err:
    print(f"DUSTY failed: {err}")
```

Inside `EmceeRunner`, a failed model gives a log-probability of `-inf` for that step instead of
stopping the run.

### License
pyDusty is released under the BSD 3-Clause License (see `LICENSE`).
It includes a modified version of DUSTY (https://github.com/ivezic/dusty),
Copyright (c) 2016, Zeljko Ivezic, distributed under the BSD 3-Clause License
(see `LICENSE-DUSTY`). pyDusty is not endorsed by the authors of DUSTY.

### Citing
If you use pyDusty in your research, please cite the following publications:
- Gómez-Muñoz M. A., et al., 2026: [arXiv](https://arxiv.org/abs/2609.12981)

Also cite the original DUSTY authors:
- Ivezić Z., et al., 1999: [SciX](https://scixplorer.org/abs/1999ascl.soft11001I/abstract)
- Nenkova M., Ivezić Z., Elitzur M. 2000: [SciX](https://scixplorer.org/abs/2000ASPC..196...77N/abstract)

### Acknowledgements
This project acknowledges to be funded by the European Union (**ERC, CET-3PO, 101042610**). Views and opinions expressed
are however those of the author(s) only and do not necessarily reflect those of the European Union or the European
Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for
them. We also acknowledge financial support from grant **CEX2024-001451-M** funded by **MICIU/AEI/10.13039/501100011033**
and the support from the State Research Agency (AEI) of the Ministry of Science, Innovation and Universities (MICIU)
of the Government of Spain under grant **PID2024-155585NA-I00** funded by **MICIU/AEI/10.13039/501100011033**.
