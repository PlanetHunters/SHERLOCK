# Changelog

## Unreleased

### Added

- `python -m sherlockpipe.emls --candidate N`: run the EMLS exomoon search on a
  candidate of a SHERLOCK results directory, like `vet` and `fit`. It reads the
  detrended curve the candidate was found in (`lc_<curve>.csv`), the star from
  `params_star.csv`, and the planet from `candidates.csv`, replaced by the
  `fit_[N]` results when they exist. Options: `--mask_candidates`,
  `--planet_mass` (otherwise forecast with Chen & Kipping below 8 R_earth and
  one Jupiter mass above), `--min_moon_radius`/`--max_moon_radius`,
  `--period_grid_size` (default 10000), `--curve`/`--lc_file` and `--no_fit`.
  Outputs go to `emls_N/`, including `emls_results.json`.

### Fixed

- EMLS planet fit: the folded transit is fitted in duration, impact parameter
  and radius (T14, b, rp) instead of a/R* bounded to 2 %, so a transit longer
  than the catalog stellar density allows no longer leaves a moon-like
  residual. On a/R* 15 % low, the fitted T14 is within 2 % and the stacked
  residual below 5 % of the depth. The fit starts from several impact
  parameters, because b, rp and T14 are degenerate for grazing transits.
- EMLS: the moon durations and templates use the fitted a/R*, inclination
  and impact parameter. The semi-major axis in AU still sets the Hill sphere
  and the moon offsets.
- EMLS: each transit's mid-time has a Gaussian prior around the linear
  ephemeris, as wide as the ephemeris uncertainty at that transit plus the
  largest planet TTV a searched moon can cause: the heaviest moon
  (the Chen & Kipping mass of the largest searched radius, capped at the
  largest moon density) on the widest searched orbit
  (half the Hill period) moves the planet a_m M_m / (M_p + M_m) from the
  barycentre. `--ttv_prior_minutes` overrides it, and 0 removes the prior.
  The ephemeris uncertainty comes from the T0 and period errors of the
  `fit_[n]` results, sqrt(sigma_T0^2 + (n sigma_P)^2). Without them it is the
  uncertainty of a linear ephemeris fitted to the other transits' timing
  precisions (Carter et al. 2008), so a transit's own noise is not counted
  twice. `emls_results.json` records the allowance and the ephemeris source.
  Both per-transit fits bound t0 to half a duration.
- EMLS: a transit whose fit does not converge or whose t0 stops at its bound
  is dropped and listed in `epochs_dropped_fit`. A per-transit radius at its
  bound is kept and listed in `epochs_flagged_rp`. A fold fit at a bound is
  warned about and recorded in `fold_fit`.
- EMLS: a transit with a high in-transit residual RMS is kept and listed in
  `epochs_flagged_rms` instead of being dropped, because a moon dip inside
  the transit also raises it.
- EMLS: the run measures how much of the best moon signal the planet fit
  absorbs, by injecting it with the fitted geometry before the fit and
  repeating the fit, and records it as `planet_fit_absorbed_fraction`. When
  the data already contain that moon, the value is a lower bound. A moon within 0.3 T14 of the planet
  loses 21 % of its depth with a radius bound of 0.05 R_earth, 63 % with
  0.5 R_earth and 70 % with 2 R_earth, so the `fit_[N]` radius error, which
  bounds the fold radius, matters.

- EMLS: the planet fits and moon templates are averaged over the exposure
  (`exposure_time` in `ExoMoonLeastSquares`; in `sherlockpipe.emls`,
  `--exposure_time` or the median point spacing of the light curve). On
  30-minute data the
  planet model now matches the exposure-averaged transit within about 2 ppm
  instead of 826 ppm, and the moon templates within 0.2 % of the depth
  instead of 21-30 %. Data at 2-minute cadence or shorter are unchanged.
  The averaging uses sub-exposure midpoints, because batman's own
  supersampling converges slowly at long cadence.
  The Pandora moon injected by `run(inject_params=...)` is averaged the same
  way, so injection tests compare like with like.
  The command measures one exposure for the whole curve as its median point
  spacing, which also covers binned curves. It does not read `EXPTIME` from
  `properties.yaml`, which only selects the downloaded cadences. A curve that
  mixes cadences is modelled with its median spacing, which follows the
  cadence with the most points. The run then warns and names each cadence
  with its share of the points and of the time, which `emls_results.json`
  records as `cadences` and `mixed_cadence`. A `--exposure_time` that
  matches none of the cadences is warned about too.
- EMLS: the SDE spectrum no longer contains placeholder scores. A hypothesis
  without a dip, or with a brightening, scores exactly as the null, and every
  positive dip is fitted, including dips shallower than `min_radius`. A
  best-SNR radius below the range is flagged (`best_snr_radius_below_range`,
  `best_radius_below_range`). On noise the SDE no longer piles up at
  sqrt(N - 1).
- EMLS: a best-SNR moon radius above `max_radius` is reported as measured and
  flagged (`best_snr_radius_above_range`, `best_radius_above_range` in
  `emls_results.json`) with a warning, because the dip is deeper than any moon
  searched for.
- EMLS: remove the call to the never-committed `sherlockpipe.exomoon_ir`
  assessment module, which logged a traceback on every run. The run now states
  that no false-alarm assessment is applied.
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
