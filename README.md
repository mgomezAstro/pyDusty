# pyDusty
Python wrapper for the radiate transfer code DUSTY (v4) (2000ASPC..196...77N).

### Requirements

You need to install DUSTY v4 from the [dusty](https://github.com/ivezic/dusty) repository.  A few changes need to e made to the original code. You need to modify the ``dusty.f90`` according to the following:

- Look for ``print*,`` and correct to ``print*``.
- Add the full installation path in the header of ``dusty.f90`` (where this file is located) and look for the following lines as:
```fortran
implicit none
character*260 :: dpath
PARAMETER dpath("full_path_to/dusty/release/dusty/")
.
.
.
open(4, file=trim(dpath)//'data/lambda_grid.dat', status = 'old')
.
.
.
if (iG.eq.1) write(stdf(iG),'(a)')trim(dpath)//"data/stnd_dust_lib/OssOdef.nk"
if (iG.eq.2) write(stdf(iG),'(a)')trim(dpath)//"data/stnd_dust_lib/OssOrich.nk"
if (iG.eq.3) write(stdf(iG),'(a)')trim(dpath)//"data/stnd_dust_lib/sil-dlee.nk"
if (iG.eq.4) write(stdf(iG),'(a)')trim(dpath)//"data/stnd_dust_lib/gra-par-draine.nk"
if (iG.eq.5) write(stdf(iG),'(a)')trim(dpath)//"data/stnd_dust_lib/gra-perp-draine.nk"
if (iG.eq.6) write(stdf(iG),'(a)')trim(dpath)//"data/stnd_dust_lib/amC-hann.nk"
if (iG.eq.7) write(stdf(iG),'(a)')trim(dpath)//"data/stnd_dust_lib/SiC-peg.nk"
```
- Run `make`.

You also need python 3.10 or above.

### Installation

`python setup.py install`


### Example

To run pydusty:

```python
import pydusty
from pathlib import Path


dusty_exe = Path("/path/to/dusty/executable/dusty")

teff=2500
td = 250

model_name = f"sphere_{teff}_{td}"

mod = pydusty.DustyInp(model_name=model_name, exe_path=dusty_exe)
mod.options["r"] = 0 # No radial profile output (see dusty manual).
mod.options["m"] = 1 # Messages in one file (see dusty manual).
mod.options["flux conservation"] = 0.1 # Flux conservation at 10% max error.
mod.set_sphere()
mod.set_blackbody(temperature=2500)
mod.set_central_radiation(central=True)
mod.set_density_profile(density_type="POWD", n_pwd=1, thickness=1000, p=2)
mod.set_grain_size_dist(grain_distribution="MRN", amin=0.005, amax=0.25, q=3.5)
mod.set_grains_abund(
    predef_abund={"Sil-Oc": 0.7},
    nk_files=["/path/to/your/nk_file/data/Lib_nk/amC-zb1.nk"],
    nk_abunds=[0.3],
)
mod.set_radiation_strenght(scale_type="T1", scale_value=td)
mod.set_optical_depth(
    tau_grid="LOGARITHMIC", lambda0=0.554, taumin=5, taumax=150, n_models=6
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
scale = mod.get_fbol()
```

See the docstrings for other methods included in pydusty.