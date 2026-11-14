"""Physical constants and unit conversions of the EMLS exomoon search.
"""
import numpy as np
import astropy.constants as ac


G = ac.G.value  # m3 kg-1 s-2

AU_TO_M = ac.au.value

AU_TO_RSUN = (ac.au / ac.R_sun).value

Msolar_to_kg = ac.M_sun.value

Mearth_to_kg = ac.M_earth.value

M_earth_to_M_sun = Mearth_to_kg / Msolar_to_kg

R_earth_to_R_sun = (ac.R_earth / ac.R_sun).value

EARTH_DENSITY = Mearth_to_kg / (4 / 3 * np.pi * ac.R_earth.value ** 3) / 1000  # g/cm3

MIN_RP_OVER_RSTAR = 1e-5

class NoUsableEpochsError(ValueError):
    """No planet transit epoch is left to search for a moon."""
