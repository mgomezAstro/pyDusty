# pyDusty
Python wrapper for the radiative transfer code DUSTY (v4) (2000ASPC..196...77N). The original
repository is at  [ivezic](https://github.com/ivezic/dusty). The modified version for this 
python wrapper is here [dusty](https://github.com/mgomezAstro/dusty).

### Requirements

- python >= 3.10
- gfortran
- openmp support (for now is not optional; it will be compiled with `-fopenmp` flag)

A version of DUSTY (V4) will be installed with this package inside the package directory (pydusty/bin/dusty) and will be 
used as default.

### Installation

`pip install git+https://github.com/mgomezAstro/pyDusty.git`


### Example

To run pydusty:

```python
import pydusty
from pathlib import Path


teff=2500
td = 250

model_name = f"sphere_{teff}_{td}"

mod = pydusty.DustyInp(model_name=model_name, project_dir="./output")
mod.options["r"] = 0 # No radial profile output (see dusty manual).
mod.options["m"] = 1 # Messages in one file (see dusty manual).
mod.options["flux conservation"] = 0.1 # Flux conservation at 10% max error.
mod.set_sphere(set_matrix=False)
mod.set_blackbody(temperature=2500)
mod.set_central_radiation(central=True)
mod.set_density_profile(density_type="POWD", n_pwd=1, thickness=1000, p=2)
mod.set_grain_size_dist(grain_distribution="MRN", amin=0.005, amax=0.25, q=3.5)
mod.set_grains_abund(
    predef_abund={"Sil-DL": 0.7},
    nk_files=["/path/to/your/nk_file/data/Lib_nk/amC-zb1.nk"],
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
        "Grains of Silicates (Sil-Oc) was set to 0.7 and aC to 0.3.",
    ]
)
mod.print_inp_file()

mod.run()
```

To read the output:

```python
import pydusty


model_name = Path("./output/sphere_2500_250")


mod = pydusty.DustyReader(model_name)
wave, spec = mod.get_spectra()
ot = mod.get_output_data() #A dictionary with the the output table in the .out file.
```

See the docstrings for other methods included in pydusty.
