# Changelog

## Unreleased

### Added

- EMLS synthetic planet false-alarm test
  (`sherlockpipe/regression_tests/emls_synthetic_planet/synthetic_planet.py`): removes
  the real transits, injects the planet at shifted ephemerides without a moon (exact
  model, limb darkening error, exposure smearing) and counts detections, with the FAP
  from the nulls of the real curve (`nulls` command, once per target). It reads an
  `emls_results.json` and a .csv or .npz curve; `plan` prints the job lines and
  `summary` the counts. Run it before changing how EMLS models the planet;
  `RESULTS.md` has the KOI-189.01 and Kepler-1514 b results.
- EMLS false-alarm probability (audit item H10): `--null_runs` (default 100)
  searches on pseudo-epochs, windows of the same light curve away from the
  planet transits and outside every searched moon window, moved onto the real
  epochs with the fitted planet injected. They keep the target's own noise,
  gaps and correlated noise and go through the same planet fit, grid and
  selection. A Gumbel fit to their SNR gives `fap`, with the empirical
  (k+1)/(N+1) as a check; `detected` is `fap_upper < --fap_threshold`
  (default 0.01), where `fap_upper` is the 95th percentile of the Gumbel
  `fap` over 1000 bootstrap resamples of the nulls. With N nulls no
  probability below 1/(N+1) is measured, so at the default 100 nulls every
  detection extrapolates the Gumbel tail past the largest null; on
  Kepler-1514 b the same injection gave 5.5e-5 or 7.5e-4 depending on the
  null set. `fap_extrapolated` marks such a `fap` and the log recommends about
  1000 nulls to confirm a candidate; `null_fit_rejected` is an
  Anderson-Darling rejection of the Gumbel at 5 %. The Gumbel stays rather
  than a free-shape GEV, whose shape comes out negative on the real nulls
  (a shorter tail), so it is the conservative tail. Without data away from
  the transits, or with fewer than 20 completed nulls, `fap` and `detected`
  are null and the log says why.
- EMLS single-event flag (audit item H12): each epoch's depth and SNR and the
  leave-one-out SNR at the best hypothesis are stored in `epoch_stats`.
  `single_event` is set when leaving one transit out keeps less than half the
  SNR, or with one transit; it does not change `detected`.
- EMLS period aliases (audit item H9): `alias_family` in
  `emls_results.json` lists the sampling aliases 1/P' = 1/P + k/P_planet of
  the selected period inside the grid, |k| up to 30, with their SNR and SNR
  below the best, and the log names those within 1 SNR. The injection alias
  check also goes to |k| = 30.
- `emls_results.json` records `fap`, `fap_upper`, `fap_empirical`,
  `fap_extrapolated`, `null_fit_rejected`, `fap_threshold`,
  `detected`, the null runs and their SNRs, `single_event`, `loo_snr_min`,
  `epoch_stats`, `sde` and the `sherlockpipe_version` (audit item I8).

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

- EMLS planet shape check scales its bin errors by the red-noise factor, so
  correlated noise no longer fails it: Kepler-1514 b (beta 4.1) failed with p from
  1e-5 to 1e-37, and now passes. It stays a flag and does not change `detected`: a
  moon inside the transit fails it as a planet residual does (injected Rp/5 moons on
  KOI-189.01 give chi2 140-235 for 20 bins, the unsmeared KOI-686.01 planet 90), and
  the same test on the residual of the best hypothesis, with its moon windows left
  out, failed an injected moon and passed the false one, so neither works as a veto.
  `planet_shape_correction_large` warns when the fitted impact-parameter correction
  exceeds 0.1, beyond the first-order range of its column.
- EMLS planet shape column for the geometry is the derivative in b^2 instead of b
  (`impact_squared`; `planet_shape_corrections["impact"]` is the resulting change of
  b). The flux is even in b, so the b derivative vanishes for a central transit and
  its coefficient was unbounded: Kepler-1514 b (b 0.004) and KOI-125.01 (b 0.003)
  reported impact corrections of 10.6 and 26.1.
- EMLS measures the exposure time by default. `ExoMoonLeastSquares` used an
  instantaneous model unless `exposure_time` was given, so the catalog driver and
  any direct use of the class modelled Kepler long cadence without smearing; only
  `python -m sherlockpipe.emls` measured it. On KOI-686.01 (ingress about 1 h) the
  unsmeared planet residual gave a false moon at SNR 5.7 and FAP 0.003; with the
  29.4 min exposure it drops to SNR 4.6 and FAP 0.063, and the planet shape check
  chi2 from 111 to 44 for 20 bins. The default is now `exposure_time="auto"`, the
  median point spacing; `None` keeps the instantaneous model and warns when the
  points are spaced by more than 2 min.
- EMLS outer moon orbit (audit items G5, G6): the widest searched orbit is the
  Domingos et al. (2006) prograde limit a = 0.4895 (1 - 1.0305 e_p) r_H, with
  the planet eccentricity, instead of half the Hill period (0.63 r_H, above
  every published prograde limit). The log states a/r_H and the period.
- EMLS eccentric planets (audit item G11): moon transit durations and the
  moon offsets a_m / v_p scale with the planet speed at conjunction,
  sqrt(1 - e^2) / (1 + e sin w), so the window matches the eccentric template
  (within 2 % of batman T14 at e = 0.3).
- EMLS moon offsets a_m / v_p use the planet speed measured by the input
  transit fit, v_p = 2 R* sqrt((1 + Rp/R*)^2 - b^2) / T14, instead of
  2 pi a / P with the input e and w, which catalogs rarely fit. On
  Kepler-1514 b the catalog duration is 1.65 times what the assumed e = 0.15,
  w = 0 give, so the offsets were 1.65 times too short and an injected 10 d
  moon was found at 19-21 d; it is now found at 9.6-9.8 d or a sampling
  alias of 10 d. The log compares this speed with the one from a, e and w
  and warns above 10 %; the fold-fitted T14 only checks the input one and
  warns above 10 %. A search duration is no T14 (`duration_from_fit=False`,
  set by `python -m sherlockpipe.emls` when the duration comes from
  candidates.csv and not from a fit): then the speed is 2 pi a / P with e and
  w until the fold fit, and the fold fit's T14 and b after it. The
  eccentricity still sets the Domingos stability limit. The real Kepler-706 b
  regression now injects its Pandora moon on the orbit the observed T14 and
  b give (a/R* 67.7), not on the logg-based one (62.2), whose speed the
  planet in the data does not have.
- EMLS injection geometry: `inject_moon` sets Pandora's a/R* to
  v_p P sqrt(1 - e^2) / (1 + e sin w) / (2 pi) with the same transit planet
  speed, so the injected planet crosses the star in the input T14 and the
  injected moon has the offsets a_m / v_p the search assumes, for any e and
  w (checked against Pandora to 0.1 %). With the input a, an assumed e or w
  injected a moon around a planet faster or slower than the one in the light
  curve (1.65 times faster on Kepler-1514 b), so injection-recovery tested
  another system. The log gives both a/R* and warns above 10 %.
- EMLS moon window on the planet transit (audit item G17):
  `planet_overlap_fraction` (share of epochs where the best moon window
  touches the planet transit), `planet_overlap_mean`, `best_tau` and
  `period_constrained` in `emls_results.json`; with an offset below T14/4 the
  period is flagged as not constrained by the data, and the log says the
  signal rests on the planet subtraction.
- EMLS default period grid (audit item G19): 10000 periods for the class too,
  as for the command (2000 captured 0.90-0.93 of moons at their period).
- EMLS fold fit (audit item B8): it ends with a bounded least-squares run on
  the residuals, so it converges on large folds where Powell's relative
  tolerance stopped early; chi2 is within 0.1 of a least-squares fit from the
  truth on a 62100-point fold.
- EMLS fold realignment (audit item B6): after the first fold, each epoch's
  mid-time is fitted with the C1 prior, the fold is rebuilt on those times and
  the shape refitted. The new shape is kept when the fold chi2 drops by more
  than N + 4 + 3 sqrt(2N) for N epochs. The moon stage keeps the linear
  ephemeris for the moon phases. A 30-minute TTV now gives rp within 0.3 %.
- EMLS fold rp start (audit item B2): the fold radius starts from the observed
  central depth with the input shape, and its +-1 sigma bound is centred
  there, so a catalog radius from another stellar radius no longer rejects
  the epochs; 20 % high now keeps 12/12 epochs.
- EMLS planet shape check (audit item D2): the stacked residual over
  +-0.75 T14 is tested with a chi2 in 20 phase bins; a p-value below 1e-3
  logs a warning and sets `planet_shape_check.ok` false (a moon inside the
  transit can trip it too).
- EMLS mask mode (audit items D4, E2, E3): an epoch without in-transit data
  or with an unusable fit is masked around its ephemeris time; the default
  mask is 1.1 T14 / 2 plus one cadence wide (`mask_duration_factor` still
  sets it), so no ingress or egress point is left; the run logs the mean
  fraction of the widest moons' transit time that survives the mask, as a
  warning below 50 %.
- EMLS detrending (audit item A1): the command now reads `lc_0.csv`, the
  curve before SHERLOCK detrending, and EMLS detrends it with a biweight
  (`--detrend_window`, default three planet durations) in which every point
  within the widest moon window of each transit is masked, so neither the
  planet nor a searched moon shapes the trend. All points are divided by the
  trend, which is a straight line across each mask. On the audit fixture at
  1000 ppm (2.5 R_earth moon at 0.25 and 1.5 T14) 0.99-1.00 of the moon and
  the planet survive, against 0.24-0.46 of the moon with the unmasked
  biweight. `--curve N` uses a SHERLOCK curve as before, and without
  `lc_0.csv` the candidate's curve is used with a warning. On noise the
  interpolated trend raises the median best SNR by about 9 % (2.87 to 3.12
  over six runs); the null runs go through the same detrending.
- EMLS detrending mask (`--detrend_mask`, default `auto`): where the moon
  window is more than twice the biweight window, the line across it misses
  the stellar variability. On Kepler-1514 b with a 5.3 M_J planet (5.8 d
  window, 2.7 d biweight) it left 222 ppm scatter and 170 ppm offsets
  between epochs inside the windows, of the size of the moons searched for.
  `auto` then masks only +-0.75 T14 around each epoch (`planet`): 94 ppm and
  19 ppm there, and injected moons at Rp/3, Rp/4 and Rp/5 went from SNR
  6.9/4.3/3.4 to 7.6/6.7/5.1 (Rp/4 now detected, FAP 5e-4), with the
  control at FAP 0.35. The cost is the part of a moon outside the mask that
  the biweight absorbs: about 20 % on Kepler at 30-min cadence, where the moon
  points stand out from the noise, and up to 44 % on the audit fixture
  (2-min cadence, moon within the point noise). So narrower windows keep the
  whole-window mask (`moon_window`). `none` masks nothing and divides the
  planet model out first. The choice is logged and written to
  `emls_results.json` (`detrending.mask`).
- EMLS planet shape fitted with each moon: the stacked fit also takes one
  impact parameter and two limb darkening (u1, u2) corrections shared by all
  epochs, the derivatives of the subtracted planet model. A limb darkening
  error of +-0.15 on KOI-189.01 (1.8 % deep) left a residual inside the
  transit that the moon window took as a moon in 10 of 20 synthetic planets
  without a moon. `fit_planet_shape = False` turns them off, and
  `emls_results.json` gives them (`planet_shape_corrections`). New inputs
  `--impact_param` and `--fix_impact_param` (`fix_impact_param=True`) set the
  impact parameter and keep it in the planet fold fit.
- EMLS planet depth and mid-times fitted with each moon: a moon inside the
  planet transit in most epochs deepened the fold fit by its mean depth and
  pulled each epoch's fitted mid-time towards itself (shift versus moon
  offset correlation +0.95 on Kepler-1514 b, +0.92 on KOI-189.01), and the
  subtraction left that depth and those shifts back in the moon window
  (+300 ppm on Kepler-1514 b, +700 ppm on KOI-189.01 with an Rp/5 moon).
  The search still stacks the epochs in the frame of the moon with one moon
  template; it now fits that window together with one planet depth
  correction shared by all epochs and one mid-time correction per epoch (the
  derivatives of the subtracted planet model with respect to its depth and
  to each epoch's mid-time), before the upper outlier clipping, and divides
  the SNR by the uncertainty those corrections add, and the chi-squared gain
  of the residual spectrum by its square (without it, noise runs reached a
  residual SDE of 6.9, above a 3 R_earth moon at 6.6). The null searches, the
  leave-one-out SNR and the epoch figures use the same fit. Rp/5 moons went
  from SNR 3.96 (FAP upper 0.016, not detected) to 5.71 (FAP 0.0005,
  detected) on Kepler-1514 b and from 6.2 to 14.1 on KOI-189.01, with
  controls at FAP 0.52 and 0.030; the search takes about 20 % longer and the
  nulls up to 70 % longer. `emls_results.json` gives the corrections at the
  best moon (`planet_depth_correction_ppm`, `planet_t0_shifts`);
  `fit_planet_t0 = False` keeps the depth correction only.
- EMLS injection before detrending (audit item A2): with the EMLS
  detrending, an injected moon is added to the undetrended curve, so
  recovery includes the detrending loss; the null pseudo-epochs are detrended
  the same way.
- EMLS per-epoch baseline (audit item A3): each epoch is divided by a straight
  line fitted to the points outside the widest moon window, not by a
  constant, so a residual slope does not move the planet mid-time (t0 rms
  0.24 min at +-3000 ppm/d, was 1.27 min) and a moon cannot shape it.
- EMLS SNR under correlated noise (audit item H11): the SNR is divided by a
  red-noise factor beta (Pont, Zucker & Queloz 2006), measured away from
  every transit and moon window by binning at the moon transit duration and
  comparing adjacent bins with the white-noise expectation; beta below one is
  set to one. With 20 % of the variance in AR(1) noise (0.1 d), runs above
  SNR 5 fell from 19/20 to 0/20; white noise is unchanged (beta 1.03).
  `emls_results.json` records `red_noise_factor` and `red_noise_measured`.
- EMLS template amplitude and moon radius (audit items H13, H14): the
  template amplitude is the weighted least-squares value, not the template
  peak set to the mean depth (about 12 % low), and the radius is inverted
  from it through the template set instead of sqrt(mean depth), which was
  19 % low at b = 0.9. Noiseless moons now give the amplitude within 1 % and
  the radius within 0.1 % for b up to 0.9.
- EMLS reports one hypothesis (audit items H7, H8): the SNR selects the
  period and phase, and the SNR, the SDE of the SNR spectrum (`sde`, a
  diagnostic) and the radius all come from that hypothesis, so they reproduce
  from one `search` call. The residual-spectrum SDE stays as `max_sde`, a
  diagnostic.
- EMLS injection recovery (audit item I1): `recovered` now needs a detection
  (false-alarm probability below the threshold), moon times within half a
  moon transit of the injected ones, and the period within 5 %. A sampling
  alias with matching times sets `recovered_alias` instead. Without a
  false-alarm probability both are None. Before, an undetectable moon was
  "recovered" 82 % of the time by chance. The period counts only when the
  injected period is the nearest member of its alias family
  1/P' = 1/P + k/P_planet: with the 5 % tolerance alone, the k = 1 alias of a
  10 d moon of Kepler-1514 b (217.8 d), 4.4 % away, was counted as the
  period. The nearest alias order is logged and kept as `injection_alias_k`.
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

### Removed

- `sherlockpipe/resources/emls_ir/` (audit item I7): the EMLS injection-recovery
  calibration built with the search before the audit fixes, with no provenance,
  100 nulls per set behind 99.9 % thresholds, and no code reading it. EMLS now
  computes a per-target false-alarm probability (`--null_runs`). The files,
  including the injection grid `grid.json`, remain in the git history.

### Known limitations

- EMLS is a screening search with a coplanar, circular, prograde moon orbit
  (audit item G12: the unused moon inclination, eccentricity and periastron
  fields are removed). Its moon phase is frozen at the planet mid-transit and
  every moon transit has one duration (audit items G3, G4), its moon orbit is
  planet-only rather than barycentric (G14, negligible for q = M_m/M_p below
  0.01), and planet-moon overlaps are not modelled (G16: up to 40 % of a
  single epoch's moon deficit near P_m = 5-20 d for giants). Each run logs
  tau v_m / (v_p T), the scale of the timing errors, and q for the largest
  searched moons, with a warning above 0.1 and 0.01. Candidates should be
  confirmed with a photodynamical fit such as Pandora.

- EMLS `sde` and `max_sde` are diagnostics: neither separates moons from
  noise (audit item H7). Use `max_snr` with `fap`.
- The EMLS null runs reuse the same light curve, so they are not independent:
  `null_independent_offsets` gives the number of independent pseudo-epoch
  positions with data per epoch; below 3 no null runs are made and no
  false-alarm probability is given. Curves cut around the
  transits, such as the Kepler-706 regression fixture, have no pseudo-epochs
  and get no false-alarm probability.

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
