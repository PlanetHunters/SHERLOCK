.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Plan Module
=============================================

The ``plan`` module in SHERLOCK provides tools for planning follow-up observations of exoplanet candidates. This module helps astronomers determine optimal observation windows and configurations for confirming candidate planets.

Overview
--------

The plan module offers capabilities for:

- Transit event prediction for follow-up observations
- Visibility calculations for specific observatories
- Weather and seeing condition integration
- Prioritization of targets based on scientific criteria
- Generation of observation plans and schedules

Usage
-----

The observation plan module can be run directly from the command line (within the target directory):

.. code-block:: bash

   # Basic usage for planning observations of a specific candidate
   # Note: Must specify either observatories file OR coordinates
   python3 -m sherlockpipe.plan --candidate 1 --observatories observatories.csv

   # Draw an event card for every observable transit instead of the 20 best
   python3 -m sherlockpipe.plan --candidate 1 --observatories observatories.csv --max_cards 0
   
   # Alternative: specify observatory coordinates directly
   python3 -m sherlockpipe.plan --candidate 1 --lat 28.3 --lon -16.5 --alt 2400
   
   # For example configuration format, see examples/properties/plan.yaml
   # `<https://github.com/franpoz/SHERLOCK/blob/master/examples/properties/plan.yaml>`_
   # Note: This module requires Bayesian fit results to be present in the target directory
   
   # Example observatories.csv file format:
   # `<https://github.com/franpoz/SHERLOCK/blob/master/examples/observatories.csv>`_
   

The input configuration file typically includes:

- Target information (coordinates, magnitudes)
- Observatory details (location, capabilities)
- Observation constraints (airmass, moon separation)
- Time period for planning

The report
----------

The command writes ``<candidate>_observation_plan.pdf`` (landscape A4) and ``observation_plan.csv`` in the plan
directory of the candidate. The report has, in this order:

- **Summary page:** the target, the candidate with its errors, the timing uncertainty, the constraints, a calendar
  of opportunities and a shortlist of the 10 best ones. In the calendar every row is an observatory, the faint lines
  are all the transits of the plan, each dot is an observable event (bigger and darker for a larger observable
  fraction), the numbers mark the best opportunities and the strip underneath is the moon illumination. An
  observatory without events says why. The shortlist is ranked by observable fraction x min(altitude, 60 deg) /
  60 deg x (1 - moon illumination / 2).
- **Event cards:** one per transit that at least one observatory can observe, the best first. A card compares
  the observatories on a common UTC axis: the night is shaded, the grey line is the altitude of the target, the
  green part is where it is observable (night, above the minimum altitude and with the moon far enough), the
  orange band is the transit and the pink one is the uncertainty of the mid-time. ``--max_cards`` sets how many
  cards are drawn (20 by default, 0 for all of them).
- **Appendix:** the observatories with what limits them and every observation window, by date.

API Reference
------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   
   sherlockpipe.observation_plan
   sherlockpipe.plan

Integration with External Packages
---------------------------------

The plan module integrates with astronomical packages for:

- Ephemeris calculations
- Observatory visibility computations
- Atmospheric condition predictions
- Telescope scheduling optimization
