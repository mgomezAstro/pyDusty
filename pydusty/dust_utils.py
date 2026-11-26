#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:22:20 2025

@author: magm
"""

import miepython as mie
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid as trapz
import importlib.resources as pkg
from functools import lru_cache


# Some constants (cgs units)
k_b = 1.380649e-16
c = 2.99792458e10
h = 6.62607015e-27
aa_to_cm = 1.0e-08
micron_to_cm = 1.0e-04
pc_to_cm = 3.08567758e18
kpc_to_cm = 1e03 * pc_to_cm
mpc_to_cm = 1e06 * pc_to_cm
solmass_to_grams = 1.98840987e33

_distance_units = {"mpc": mpc_to_cm, "kpc": kpc_to_cm, "pc": pc_to_cm}


def distance_to_cm(distance: float, distance_unit: str = "Mpc") -> float:
    """Convert a distance in pc, kpc or Mpc to cm."""
    unit = distance_unit.lower()
    if unit not in _distance_units:
        raise ValueError("Distance must be one of [Mpc, kpc, pc].")
    return distance * _distance_units[unit]


@lru_cache(maxsize=None)
def _read_nk(filename: str):
    """Read (and cache) a wavelength / n / k table."""
    return np.loadtxt(filename, skiprows=1, unpack=True)


def _q_abs(m, x):
    """Absorption efficiency Q_abs = Q_ext - Q_sca."""
    qext, qsca, _, _ = mie.efficiencies_mx(m, x)
    return qext - qsca


def planck_nu(nu: np.ndarray, temperature: float):
    A = 2 * h * nu**3 / c**2
    B = 1 / (np.exp(h * nu / (k_b * temperature)) - 1.0)
    return A * B


def planck_lam(wave: np.ndarray, temperature: float):
    A = 2 * h * c**2 / wave**5
    B = 1 / (np.exp(h * c / (wave * k_b * temperature)) - 1.0)
    return A * B * aa_to_cm


def planck_bb(
    wave: np.ndarray, temperature: float, output_units: str = "nu"
) -> np.ndarray:
    """


    Parameters
    ----------
    wave : np.ndarray
        Wavelength array in cm.
    temperature : float
        Temperature of the star or dust in K.
    output_units: str
        Either nu or lam for the output flux units.

    Returns
    -------
    flux : np.ndarray
        The blackbody flux in either in erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ or in erg s⁻¹ cm⁻² AA⁻¹ sr⁻¹.

    """
    if output_units == "nu":
        nu = c / wave
        return planck_nu(nu, temperature)

    return planck_lam(wave, temperature)


def get_opacity(
    wave: np.ndarray,
    a: float,
    grain_type: str = "silicate",
    m: np.ndarray | None = None,
    rho: float | None = None,
):
    """


    Parameters
    ----------
    wave : np.ndarray
        Wavelength in cm.
    a : float
        The radius of the grain in cm.
    grain_type : str, optional
        The type of the minearoloygy. Only silicate or graphite. The default is "silicate".
    m : np.ndarray, complex
        The complex refractive index of the material (m = n - k * j). If this is chosen grain_type is ignored.
    rho : float
        Density of the material in gr/ccm. If m is None, the this value is ignored.

    Raises
    ------
    ValueError
        The value of the grain type is not one the permited minearologies.

    Returns
    -------
    np.ndarray
        The opacity of the material assuming a single grain size.

    """

    data_path = pkg.files("pydusty").joinpath("data/")

    if m is not None and rho is not None:
        x = 2 * np.pi * a / wave
        Qabs = _q_abs(m, x)

        return 3.0 * Qabs / (4.0 * rho * a)

    elif (m is None and rho is not None) or (m is not None and rho is None):
        raise ValueError(
                "You must specify the density rho when using custom 'm'."
            )
        
    available_grains = {
        "silicate": ["silicate.txt", 3.3],
        "amcarb": ["am-carb.txt", 1.81],
        "graphite": ["graphite_", 2.26],
    }

    grain_type = grain_type.lower()
    if grain_type not in available_grains.keys():
        raise ValueError(
            f"Grain type not available.\nUse one of : {available_grains}."
        )

    if grain_type == "graphite":
        wl_par, real_par, im_par = _read_nk(
            str(data_path / (available_grains[grain_type][0] + "par.txt"))
        )
        wl_perp, real_perp, im_perp = _read_nk(
            str(data_path / (available_grains[grain_type][0] + "perp.txt"))
        )

        wl_par = wl_par * micron_to_cm
        wl_perp = wl_perp * micron_to_cm

        rho = available_grains[grain_type][1]
        m_par = real_par - 1.0j * im_par
        m_perp = real_perp - 1.0j * im_perp
        x_par = 2 * np.pi * a / wl_par
        x_perp = 2 * np.pi * a / wl_perp
        qabs_par = _q_abs(m_par, x_par)
        qabs_perp = _q_abs(m_perp, x_perp)

        qabs = (1.0 / 3.0) * qabs_par + (2.0 / 3.0) * qabs_perp

        Qabs = np.interp(wave, wl_par, qabs, left=0.0, right=0.0)
    else:
        wl, real, im = _read_nk(str(data_path / available_grains[grain_type][0]))
        wl = wl * micron_to_cm
        rho = available_grains[grain_type][1]
        m = real - 1.0j * im
        x = 2 * np.pi * a / wl
        qabs = _q_abs(m, x)

        Qabs = np.interp(wave, wl, qabs, left=0.0, right=0.0)

    return 3.0 * Qabs / (4.0 * rho * a)


@lru_cache(maxsize=32)
def _cached_opacity(wave_bytes: bytes, a: float, grain_type: str) -> np.ndarray:
    """Opacity for a predefined grain type, cached in memory per process."""
    wave = np.frombuffer(wave_bytes, dtype=np.float64)
    opac = get_opacity(wave=wave, a=a, grain_type=grain_type)
    opac.setflags(write=False)
    return opac


def thermal_emission(
    wave: np.ndarray,
    dust_mass: float,
    temperature: float,
    a: float,
    distance: float = 1.0,
    distance_unit: str = "Mpc",
    grain_type: str = "silicate",
    m: np.ndarray | None = None,
    rho: float | None = None,
):
    """


    Parameters
    ----------
    wave : np.ndarray
        Wavelength in microns.
    dust_mass : float
        Dust mass in solar masses.
    temperature : float
        Temperature in K.
    a : float
        Grain size in micron.
    distance : float
        Specify the distance and the units of the distance (only pc, kpc or Mpc). The default is [1.0, "Mpc"].
    distance_unit : str
        Unit of the distance. Must one of [Mpc, kpc, pc].
    grain_type : str, optional
        Give the grain type (supported silicate and amcarb at the moment). The default is "silicate".
    m : np.ndarray, complex
        The complex refractive index of the material (m = n - k * j). If this is chosen grain_type is ignored.
    rho : float
        Density of the material in gr/ccm. If m is None, the this value is ignored.

    Returns
    -------
    flux: np.ndarray
        Flux density in erg s⁻¹ cm⁻² Hz⁻¹.

    """
    distance = distance_to_cm(distance, distance_unit)

    bb = planck_bb(wave * micron_to_cm, temperature, output_units="nu")

    if m is not None and rho is not None:
        opac = get_opacity(
            wave=wave * micron_to_cm,
            a=a * micron_to_cm,
            m=m,
            rho=rho,
        )
    else:
        wave_cm = np.ascontiguousarray(wave * micron_to_cm, dtype=np.float64)
        opac = _cached_opacity(
            wave_cm.tobytes(), float(a * micron_to_cm), grain_type.lower()
        )

    return (dust_mass * solmass_to_grams) * bb * opac / distance**2


def ConvolveFilter(wave, flux, wfilt, ffilt, returnType="fnu"):
    """
    Return the magnitude in the desired system.

    Inputs:
    --------
    wave: array
        Wabelength in angstroms. [Ang]
    flux: array
        Flux in units flam. [erg/s/cm2/A]
    wfilt: array
        Wavelgtnh coverage of the filter  [Ang].
    ffilt: array
        Throughtput of the filter.
    returnType: str
        Desired system of the magnitude/flux. One of fnu, flam, or AB. (Default. fnu).

    Output:
    --------
    value: float (default flux in Fnu)
        Flux in the desired system.
    """

    filt_interp = interp1d(
        wfilt, ffilt, kind="linear", bounds_error=False, fill_value=0.0
    )

    x = wave.copy()
    y = flux.copy()

    I1 = trapz(y * filt_interp(x) * x, x)
    I2 = trapz(filt_interp(x) / x, x) * 2.99792458e18

    if returnType == "fnu":
        return I1 / I2
    elif returnType == "AB":
        return -2.5 * np.log10(I1 / I2) - 48.6
    elif returnType == "flam":
        I1 = trapz(y * filt_interp(x), x)
        I2 = trapz(filt_interp(x), x)
        return I1 / I2
