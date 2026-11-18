.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Fit Module
=============================================

The ``fit`` module in SHERLOCK allows you to perform Bayesian parameter estimation for exoplanet transit models. This module integrates with ``allesfitter`` to provide a comprehensive fitting environment.

Overview
--------

The fit module is responsible for:

- Preparing light curve data for model fitting
- Setting up priors and model configurations
- Running MCMC or Nested Sampling algorithms
- Generating posterior distributions
- Creating diagnostic plots and fit reports

Usage
-----

The fit module can be run directly from the command line:

.. code-block:: bash

   # Basic usage with a specific candidate
   python3 -m sherlockpipe.fit --candidate 1
   

The input configuration file (when used) specifies parameters like:

- Transit model type
- Sampling method and parameters
- Prior distributions
- Data sources

Fit report
----------

When the fit ends, ``<object id>_fit.pdf`` is written next to the results (in ``fit_[n]/main``). It is a landscape
document with, for every fitted candidate (SOI), three pages, and a last page shared by all of them:

1. the summary: a verdict, the folded transit with the posterior model, the key parameters against their priors
   (including the stellar density, with the one of the star as a reference) and the limb darkening and noise;
2. the consistency: the odd and even fits against the main one, and every transit against the posterior model;
3. the corner plot of the radius ratio, the impact parameter, the stellar density, the duration, the depth and the
   period;
4. (once, at the end) whether the nested sampling converged and whether the noise, once the Gaussian process baseline
   is removed, is white.

A posterior that piles up against the edge of its prior, a stellar density that disagrees with the one of the
orbit, or odd and even fits that sit at the edge of their priors are reported as warnings. The files of
allesfitter (``results/ns_corner.pdf``, ``ns_trace.pdf``, ``ns_fit_*.pdf``, the tables, the ``odd`` and ``even``
folders and the logs) are kept as they are. A failure writing the report never breaks the fit.

API Reference
------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   
   sherlockpipe.fit
   sherlockpipe.bayesian_fit

Integration with External Packages
---------------------------------

The fit module uses ``allesfitter`` as its main backend for parameter estimation. This integration allows SHERLOCK to leverage:

- Multiple sampling algorithms (emcee, dynesty)
- Comprehensive model capabilities
- Publication-ready plotting functions
