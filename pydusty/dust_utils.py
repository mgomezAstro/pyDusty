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
import os
import sys
import importlib.resources as pkg


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
        Qext, _, _, _ = mie.efficiencies_mx(m, x)

        return 3.0 * Qext / (4.0 * rho * a)

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
        wl_par, real_par, im_par = np.loadtxt(
            data_path + available_grains[grain_type][0] + "par.txt",
            skiprows=1,
            unpack=True,
        )
        wl_perp, real_perp, im_perp = np.loadtxt(
            data_path + available_grains[grain_type][0] + "perp.txt",
            skiprows=1,
            unpack=True,
        )

        wl_par *= micron_to_cm
        wl_perp *= micron_to_cm

        rho = available_grains[grain_type][1]
        m_par = real_par - 1.0j * im_par
        m_perp = real_perp - 1.0j * im_perp
        x_par = 2 * np.pi * a / wl_par
        x_perp = 2 * np.pi * a / wl_perp
        qext_par, _, _, _ = mie.efficiencies_mx(m_par, x_par)
        qext_perp, _, _, _ = mie.efficiencies_mx(m_perp, x_perp)

        qext = (1.0 / 3.0) * qext_par + (2.0 / 3.0) * qext_perp

        Qext = np.interp(wave, wl_par, qext, left=0.0, right=0.0)
    else:
        wl, real, im = np.loadtxt(
            data_path / available_grains[grain_type][0],
            unpack=True,
            skiprows=1,
        )
        wl *= micron_to_cm
        rho = available_grains[grain_type][1]
        m = real - 1.0j * im
        x = 2 * np.pi * a / wl
        qext, _, _, _ = mie.efficiencies_mx(m, x)

        Qext = np.interp(wave, wl, qext, left=0.0, right=0.0)

    return 3.0 * Qext / (4.0 * rho * a)


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
    distance_unit = distance_unit.lower()
    if distance_unit not in ["mpc", "kpc", "pc"]:
        raise ValueError("Distance must be one of [Mpc, kpc, pc].")

    if distance_unit == "mpc":
        distance *= mpc_to_cm
    if distance_unit == "kpc":
        distance *= kpc_to_cm
    if distance_unit == "pc":
        distance *= pc_to_cm

    bb = planck_bb(wave * micron_to_cm, temperature, output_units="nu")
    opac = np.ones_like(bb)

    if m is not None and rho is not None:
        if os.path.exists("./custom_opac.txt"):
            opac = np.loadtxt(f"./custom_opac.txt")
        else:
            opac = get_opacity(
                wave=wave * micron_to_cm,
                a=a * micron_to_cm,
                m=m,
                rho=rho,
            )
            np.savetxt("./custom_opac.txt", opac)
    else:
        if os.path.exists(f"./{grain_type}_opac.txt"):
            opac = np.loadtxt(f"./{grain_type}_opac.txt")
        else:
            opac = get_opacity(
                wave=wave * micron_to_cm,
                a=a * micron_to_cm,
                grain_type=grain_type,
            )
            np.savetxt(f"./{grain_type}_opac.txt", opac)

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
