"""Moon survival against tidal migration (Barnes & O'Brien 2002) with a generous margin.

The bound is the time a moon takes to cross the whole region between the planet and the widest
stable orbit a_crit, which holds whether it migrates outwards (from the synchronous orbit) or
inwards (from a_crit, once the star has slowed the planet spin). Barnes & O'Brien derive it for
prograde primordial moons; for captured retrograde moons, which migrate inwards like Triton, the
same expression with the retrograde stability limit bounds the lifetime (their Eq. 12), so the
retrograde bound is (0.9309 / 0.4895)^6.5, about 65 times, more massive with the Domingos limits.
It is an order-of-magnitude estimate: k2/Q is an empirical Solar System relation that can be off
by one or two orders of magnitude for a given planet, the age is assumed, and the mass goes as
a_crit^6.5. It also stops applying to moons massive enough to lock the planet spin (above about
8 M_earth for a Jupiter-mass planet in Barnes & O'Brien), which are better treated as binary planets.
So it is used with a generous margin, a planet ten times less dissipative (k2/Q divided by
TIDAL_SURVIVAL_K2_Q_FACTOR) and a young 1 Gyr system, which lets a moon 50 times more massive survive
than the nominal 5 Gyr bound. The catalog driver drops a target only when even this generous bound
rules out the smallest moon searched, and EMLS reports the generous radius as a warning only.
"""
import astropy.constants as ac
import numpy as np
from sherlockpipe.exomoon.constants import AU_TO_M, G, M_earth_to_M_sun, Mearth_to_kg
from sherlockpipe.exomoon.orbits import MoonOrbits

# System age in Gyr and k2/Q factor of the generous bound; the nominal HEK VI bound is 5 Gyr and 1.
TIDAL_SURVIVAL_AGE_GYR = 1.0


TIDAL_SURVIVAL_K2_Q_FACTOR = 0.1


# Planet radius in Earth radii above which k2/Q stays at its Jupiter value.
TIDAL_K2_Q_MAX_RADIUS = 11.2


def planet_k2_over_q(planet_radius):
    """Return the planet tidal efficiency k2/Q from its radius in Earth radii.

    Empirical Solar System relation of Teachey, Kipping & Schmitt (2018, HEK VI, Fig. 1),
    log10(k2/Q) = -2.90 - 0.187 Rp/R_earth, fitted to the Moon, the Earth, Neptune and Jupiter.
    Above Jupiter's radius it is kept at Jupiter's value instead of extrapolated.
    """
    return 10 ** (-2.90 - 0.187 * min(float(planet_radius), TIDAL_K2_Q_MAX_RADIUS))


def max_surviving_moon_mass(star_mass, planet_mass, planet_radius, planet_period, planet_ecc=0.0,
                            age_gyr=TIDAL_SURVIVAL_AGE_GYR, k2_q_factor=TIDAL_SURVIVAL_K2_Q_FACTOR, retrograde=False):
    """Return the largest moon mass in Earth masses that survives tidal migration for the system age.

    Barnes & O'Brien (2002, ApJ 575, 1087) Eq. 7 solved for the moon mass, as in their Eq. 8:
    a moon is lost once tides carry it across the region between the planet and the widest
    stable orbit a_crit, which takes T = (2/13) a_crit^(13/2) sqrt(M_p / G) / (3 (k2/Q)_p M_m R_p^5),
    so M_m,max = (2/13) a_crit^(13/2) sqrt(M_p) / (3 (k2/Q)_p T R_p^5 sqrt(G)). a_crit is the
    Domingos et al. (2006) limit of the orbit sense (`MoonOrbits.max_moon_semimajor_axis`): prograde
    as Barnes & O'Brien Eq. 8, or retrograde for a captured moon migrating inwards (their Eq. 12).
    k2/Q comes from
    `planet_k2_over_q` times `k2_q_factor`. HEK VI (Teachey et al. 2018) and Kipping & Yahalomi (2023)
    apply it to the Kepler catalog with 5 Gyr and a factor 1; the defaults are the generous margin
    of the module docstring.

    Parameters
    ----------
    star_mass : float
        Stellar mass in solar masses.
    planet_mass, planet_radius : float
        Planet mass in Earth masses and radius in Earth radii.
    planet_period : float
        Planet period in days.
    planet_ecc : float, optional
        Planet eccentricity, default 0.
    age_gyr : float, optional
        System age in Gyr, default TIDAL_SURVIVAL_AGE_GYR.
    k2_q_factor : float, optional
        Factor on the empirical k2/Q, default TIDAL_SURVIVAL_K2_Q_FACTOR.
    retrograde : bool, optional
        Use the retrograde stability limit, default False.

    Returns
    -------
    float
        Mass in Earth masses, NaN when an input is not finite and positive.
    """
    values = (star_mass, planet_mass, planet_radius, planet_period, age_gyr, k2_q_factor)
    if not all(value is not None and np.isfinite(value) and value > 0 for value in values):
        return np.nan
    planet_semimajor_axis = MoonOrbits.compute_semimajor_axis(star_mass, planet_period)
    a_crit = MoonOrbits.max_moon_semimajor_axis(star_mass, planet_mass * M_earth_to_M_sun,
                                                planet_semimajor_axis, planet_ecc, retrograde=retrograde) * AU_TO_M
    mass = planet_mass * Mearth_to_kg
    radius = planet_radius * ac.R_earth.value
    age = age_gyr * 1e9 * 365.25 * 86400
    k2_q = planet_k2_over_q(planet_radius) * k2_q_factor
    moon_mass = 2 / 13 * a_crit ** 6.5 * np.sqrt(mass) / (3 * k2_q * age * radius ** 5 * np.sqrt(G))
    return float(moon_mass / Mearth_to_kg)


def moon_radius_from_mass(moon_mass):
    """Return a moon radius in Earth radii from its mass in Earth masses.

    Mean mass-radius relation of Chen & Kipping (2017, ApJ 834, 17, Table 1): R = 1.008 M^0.279
    for rocky bodies up to 2.04 M_earth, then R proportional to M^0.589 (continuous at the
    transition). Bodies more massive than 131.6 M_earth keep that radius.
    """
    if moon_mass is None or not np.isfinite(moon_mass) or moon_mass <= 0:
        return 0.0
    transition = 2.04
    rocky = 1.008 * min(moon_mass, transition) ** 0.279
    if moon_mass <= transition:
        return float(rocky)
    return float(rocky * (min(moon_mass, 131.6) / transition) ** 0.589)


def max_surviving_moon_radius(star_mass, planet_mass, planet_radius, planet_period, planet_ecc=0.0,
                              age_gyr=TIDAL_SURVIVAL_AGE_GYR, k2_q_factor=TIDAL_SURVIVAL_K2_Q_FACTOR, retrograde=False):
    """Return the radius in Earth radii of the largest moon that survives tidal migration.

    `max_surviving_moon_mass` converted with `moon_radius_from_mass`; NaN when an input is not usable.
    """
    mass = max_surviving_moon_mass(star_mass, planet_mass, planet_radius, planet_period, planet_ecc, age_gyr,
                                   k2_q_factor, retrograde)
    return float(mass) if not np.isfinite(mass) else moon_radius_from_mass(mass)
