# Changelog

## Unreleased

### Fixed

- EMLS: fix a set of defects in the exomoon search (`sherlockpipe/exomoon.py`)
  found by an audit of the algorithm. Among them: the first epoch was counted
  twice, the physical constants were approximate, alpha refinement used the
  wrong polynomial coefficient, the SDE was read at the wrong index, and the
  out-of-transit weighting differed between the null and moon models.
- EMLS: the maximum moon offset is now `a_m / v_p`, independent of the impact
  parameter and of the maximum moon radius.
- EMLS: moon depth and SNR are measured against the surrounding out-of-window
  flux, with the noise of both means, so baseline offsets and outlier clipping
  no longer count as signal.
- EMLS: each epoch is divided by its out-of-transit median before the planet
  fits, so a constant normalisation offset no longer leaves a step after the
  planet is subtracted.
- EMLS: alpha uses the same reference epoch in injection, search and recovery;
  injected and recovered moon times are paired by epoch; a NaN SNR no longer
  hides a period; and the planet duration comes from the fitted model rather
  than from the data sampling.
- EMLS catalog driver: KOI targets load and reach the search.

### Known limitations

- EMLS `max_sde` is still unreliable on noise-only data until the no-dip
  placeholder and the depth gate are reworked.

### Testing

- Add EMLS guards that run the complete search on synthetic curves, and
  regression tests for each fix.

## 1.6.1 — 2026-11-01

### Changed

- Replace the vendored GTLS submodule with the pinned `gputls==0.6.1` PyPI
  dependency. Keep CuPy as an optional, CUDA-specific installation.

### Testing

- Pin pytest 9.1.1 for tox because pytest-forked 1.7.5 fails to serialize
  reports for the plan/custom-class tests under pytest 9.2.0 in the release
  environment; the affected tests pass with the pinned runner.

## 1.6.0 — 2026-10-18

### Added

- Expose GTLS as SHERLOCK's GPU TLS backend, with GPU device selection and
  backend contract tests.
- Add ExoMoon analysis and calibrated EMLS-IR resources, alongside improved
  candidate vetting, scoring, and light-curve handling.

### Changed

- Update WATSON integration to `dearwatson==2.0.0`, with compatible
  `exoml==1.4.0`, `lcbuilder==0.27.0`, and `alexfitter==1.2.20` dependencies.
- Document the CUDA 13/CuPy installation profile and Poppler requirements for
  WATSON report generation.
