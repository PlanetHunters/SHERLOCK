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

- EMLS: the planet fits and moon templates are averaged over the exposure
  (`exposure_time` in `ExoMoonLeastSquares`, `--exposure_time` or `EXPTIME`
  from `properties.yaml` in `sherlockpipe.emls`). On 30-minute data the
  planet model now matches the exposure-averaged transit within about 2 ppm
  instead of 826 ppm, and the moon templates within 0.2 % of the depth
  instead of 21-30 %. Data at 2-minute cadence or shorter are unchanged.
  The averaging uses sub-exposure midpoints, because batman's own
  supersampling converges slowly at long cadence.
  The Pandora moon injected by `run(inject_params=...)` is averaged the same
  way, so injection tests compare like with like.
  The command reads `EXPTIME` per target (`TARGETS[<target>]`) or globally,
  as SHERLOCK does, uses `EXPTIME_BINNING` as the exposure of binned points,
  and asks for `--exposure_time` when `EXPTIME` lists several values or a
  named cadence.
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
