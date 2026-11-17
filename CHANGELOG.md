# Changelog

## Unreleased

### Changed

- The Run plots are replaced by the Run sheet, with the same file names and new content. The function that drew
  them, `plot.plotting.save_transit_plot`, is removed.
- EMLS figures are paper ready: one style for every figure (`plots.PAPER_RC`, `plots.paper_figure`) with
  20-22 pt axis labels and tick labels, 19 pt titles, 16 pt legends, longer and thicker ticks, thicker lines
  and markers, and the Okabe & Ito colour-blind safe palette (EMLS model vermillion, injected moon blue,
  planet bluish green, nulls reddish purple; checked under simulated protanopia, deuteranopia and
  tritanopia). Colour is never the only cue: each element also has its own line style or marker. Legends
  go below the figure (`plots.legend_below`) so they never cover data, faint scatter points are shown at
  full opacity in the legend, and long titles and labels are wrapped. The planet-stage figures
  (`planet_figures.py`) use the same style, with times in hours from mid-transit, fluxes in ppm and
  timing offsets in minutes against the epoch time.
- EMLS catalog driver (`python -m sherlockpipe.exomoon`) now runs the set-up the validation used and the
  paper describes: no wotan pre-flattening (5 T14), EMLS detrends the raw curve itself with a 3 T14
  biweight and the 'auto' mask (`detrend_window`, `detrend_mask`), runs 100 pseudo-epoch nulls
  (`null_runs`) for the false-alarm probability, and models the exposure ('auto'). Before, the driver
  searched a pre-flattened curve, whose trend could absorb part of a moon, and gave no FAP. The
  settings are module constants (`DETREND_DURATIONS`, `NULL_RUNS`, `DRIVER_CPUS`), and
  `test_koi_driver_reaches_the_search` checks them.
- `sherlockpipe/exomoon.py` (5 659 lines) is now the package `sherlockpipe/exomoon/`, one module per
  stage: `constants`, `exposure` (per-point exposures, averaged models, weights), `orbits` (moon
  dynamics, limits, hypotheses), `planet` (fold fit, per-epoch times, subtraction, derivatives),
  `joint_fit` (moon window with the planet corrections, numba kernels), `stacking` (epoch stacking
  and clipping), `statistics` (SNR, template residuals, red noise, SDE), `templates`, `detrending`,
  `injection` (Pandora, absorbed fraction, recovery), `nulls` (pseudo-epochs, FAP), `search`
  (`ExoMoonLeastSquares`, which inherits one mixin class per stage module), `catalog` (KOI/TOI
  parameters, dynamical filters, cadence combination), `__main__` (the catalog driver, `python -m
  sherlockpipe.exomoon`), `plots` (was `exomoon_plots.py`) and `run` (was `exomoon_run.py`).
  The code moved verbatim; references to `ExoMoonLeastSquares` inside moved methods became
  `self.` or, for static methods, `cls.` (they are classmethods now), so patches on the class still
  apply. `from sherlockpipe.exomoon import ...` keeps working through the package's re-exports. A
  Pandora injection on the KOI-125.01 mixed-cadence curve gives the same SNR (30.0644855136334) as
  before the split.
- EMLS planet stage split further: `transit_fits` (`TransitFits`: single-epoch and fold fits,
  geometry from the shape, contact and ingress durations), `planet_derivatives`
  (`PlanetDerivatives`: depth, mid-time and shape derivatives for the joint fit) and
  `planet_figures` (`PlanetFigures`: fold, per-epoch and TTV figures); `PlanetModel` inherits
  them. `subtract_planet_transit` (355 lines) is now a short sequence of named steps:
  `input_transit_params`, `fold_epochs` (returns a `Fold` named tuple), `fold_fit_start` (B1/B2),
  `fit_planet_fold` (B4), `epoch_t0_priors` (C1), `realign_fold` (B6), `fit_epoch`,
  `epoch_fit_usable` (C4), `check_subtracted_epoch` (D1 and coverage; the coverage check was
  written twice), `masked_epoch_centre` (D4), `drop_rejected_epochs` and `ttv_diagnostics`. On
  KOI-125.01 and KOI-189.01 (mixed cadence) and Kepler-1514 b, in subtract and mask mode, with and
  without a per-epoch radius, the subtracted curves, epochs, TTVs, fitted geometry, epoch flags
  and figure lists are identical to before, as are the logs. Two behaviour changes in cases that
  failed before: with the fold fit on and no epoch with `min_cadences_per_t0` points it raises
  `NoUsableEpochsError` instead of a `NameError`, and without `object_dir` the TTV periodogram is
  skipped instead of raising.
- Fixed: since the per-point exposures (6ec025b) the per-epoch figures (`epochs/T0_<t0>.png`) and
  the figures after them were not written: the moon model looked up the exposure of `time`, the
  `time` module, instead of the curve's times, and the error only reached a warning. Found by
  `regression_tests/test_emls_entrypoint.py`, which was not in the runs of `tests/exomoon`.

### Added

- Run sheet: a one-page landscape image for every curve a Run searches (the PDCSAP flux and each detrend,
  `Run_<n>_PDCSAP-FLUX_<id>.png` and `Run_<n>_<ws|ks>=<value>_<id>.png`) and for the selected signal
  (`Run_<n>_SELECTED_<curve>_<id>.png`), with the file names the old Run plots had. Each page has the folded transit
  (odd and even curves, binned mean with standard errors, model from the TLS fit, residuals), the TLS SDE
  spectrum (as large as the page allows, with the selected period and its harmonics, and the harmonics
  spectrum when `use_harmonics_spectra` is on), the transit parameters with checks that flag what looks
  problematic (SDE and SNR against the search thresholds, FAP, border score, harmonic of another signal, odd/even
  mismatch, χ² between epochs and epochs whose depth disagrees with the mean), and the epochs. Epochs are
  drawn according to how many have data, because one panel per epoch does not scale: up to 24 one panel per
  epoch; up to 150 compact thumbnails, framed only when the epoch disagrees with the mean depth; above that a
  stacked epoch map where consecutive epochs of the same observing segment are stacked until each row reaches
  S/N 4, next to the depth of each row and the three most deviating and three highest S/N stacks. Every layout
  carries a short description of how to read it (the stacked map has a full one). Epochs are found with a
  binary search on the sorted time array, so a million points and thousands of epochs take about three
  seconds per image. TLS lists every epoch between the first and the last observation, also the empty ones; the sheet
  only counts epochs with data and says how many TLS listed. Like the target info report, the sheet can never
  stop a search: it is written through a guarded, lazily imported method.

- Target info report: every search writes `<object_id>_info.pdf` (A4 landscape) in the execution directory,
  after the preprocessing and the detrends and before the first Run. It gathers the star parameters, the search
  setup, the data used per sector, the alternative data that was available (sector by product, with the used
  exposure marked; long lists are split in blocks or collapsed into ranges), the preprocessing steps with the
  number of points each one removed (outliers and masks are recomputed from `lc.csv` and checked against
  `lc_predetrends.csv`; whatever they do not explain is reported as "Other steps"), the preprocessed curve per
  sector, the trend removed by every detrend window and its residual RMS, the initial and final periodicity, the
  TRICERATOPS and pixel field of view, and, when those steps ran, the stellar oscillations removed
  (`sa/signals.csv`), the auto-detrend and the high RMS mask. Nothing that was written before is removed. The
  report never stops a search if it fails, and can be made afterwards for an existing execution with
  `python -m sherlockpipe.search.target_info_report <execution_dir>`. Rendering the TRICERATOPS PDFs uses
  pypdfium2 (new dependency, ships its own PDFium, no system library); without it those pages are left out
  with a warning.

- EMLS grid search: every hypothesis stacks the same points (the widest moon-orbit window
  around each epoch), so `search_stack` builds them once with their weights, planet columns
  and the planet normal equations without the moon window, and `split_stacked` only adds the
  window terms (`_window_cross_terms`) and removes the clipped points per hypothesis; the
  upper clipping is astropy's sigma_clip algorithm without its overhead (`upper_clip`, same
  mask on 700 heavy-tailed test samples). The pool workers get the EMLS object once through an
  initializer instead of a copy with every chunk of 100 hypotheses (20.6 MB on the KOI-125.01
  mixed curve, mostly the per-point exposure table and the injection flux). On that curve a
  hypothesis takes 4 ms against 54 ms at 6ec025b (same stacked points and order, fluxes within
  1e-13, SNR within 3e-7 relative over 400 hypotheses), and a full 250 000-hypothesis search
  with 6 CPUs on a loaded machine went from 4:47 to 3:12 with the initializer alone; a null's
  own setup (detrending, planet subtraction) takes 2.4 s.
- EMLS search is about 5 times faster with the same results. The joint planet fit
  (`planet_fit`) sums its normal equations over each point's non-zero columns in numba
  (`_planet_normal_equations`, `_planet_model`) instead of building the dense design of
  points x (2 + 2 epochs + 3) columns, whose product took 79 % of the search time; each
  epoch's moon-orbit window is found by bisection on the sorted curve; the epochs are
  concatenated once instead of one by one; and the planet columns are built once per curve
  instead of once per hypothesis. On the KOI-125.01 1-minute + 30-minute curve (41 950 points
  in the windows, 33 epochs) a hypothesis went from 54 to 11 ms; over 400 hypotheses the SNR
  differs from the previous code by at most 5e-7 relative.
- EMLS handles light curves that mix cadences without assuming a mission. With
  `exposure_time="auto"` each point's exposure is measured from the spacing to its nearest
  point (5-point running median, spacings within 10 % grouped, small groups at a whole
  multiple of a larger one taken as missing points of it); when more than one cadence holds
  over 5 % of the points or of the time, every point is modelled with its own exposure in the
  planet fits, the injected moons, the null pseudo-epochs and the planet correction columns
  (`exposure_at`, `PerPointExposureModel`), and an array of per-point exposures can be passed.
  One cadence keeps one exposure as before. The flux errors of each cadence are put on one
  scale by their point-to-point scatter. The moon statistic (`box_snr`), the joint planet fit
  (`planet_fit`), the window depths, the upper-outlier clipping (now in units of each point's
  error) and the red-noise factor (bins weighted, each adjacent difference divided by its own
  expected noise) weight each point by its inverse variance, with the noise scale still
  measured from the scatter, so with equal errors they are the unweighted statistics. The moon
  templates use the exposure that carries most of the weight. Reason: the user wants the
  highest resolution, Kepler short cadence covers only some quarters, and on a mixed curve
  the unweighted mean lets the noisier 1-minute points dominate. On a synthetic curve with
  2, 10 and 30-minute segments (6 h box moon, 17 epochs, 300 noise realisations) the
  unweighted SNR was 3.39, the weighted 4.88 and binning everything to 30 minutes 4.83, with
  null scatter 0.92 in all three; one median exposure for every point left a 1.95 % planet
  with in-transit residuals of 0.74 sigma (chi-squared +792 over 1465 points), none with the
  per-point exposure. The KOI driver asks for Kepler 60 s and 1800 s, one build per cadence,
  and `combine_cadence_curves` keeps a quarter held at both once, at short cadence (lcbuilder
  sorts a multi-cadence curve by time, after which the two cannot be told apart). The CLI
  passes `exposure_time="auto"` to EMLS when the curve mixes cadences and no
  `--exposure_time` is given.
- EMLS figures (`sherlockpipe/exomoon_plots.py`) replace `stats.png`, `residuals_map.png`,
  `Residuals <n>.png`, `SDE <n>.png`, `SNR <n>.png` and `T0_<t0>_moon_model.png`:
  `search_overview.png` (SNR spectrum with the FAP threshold and the sampling aliases of the
  selected period, diagnostic spectra below), `snr_map.png` (SNR over period and phase),
  `hypothesis_<n>.png` (the selected hypothesis and the next SNR peaks at other periods,
  stacked, with a zoom), `null_distribution.png` (nulls, Gumbel fit, selected SNR and the
  FAP verdict), `epochs/T0_<t0>.png` (each transit with the planet corrections of the best
  fit) and `planet_epoch_depths.png`: each epoch's planet depth fitted without a moon
  against the share of it the selected moon would take, both about their means (the planet
  depth absorbs the means), with their correlation and the chi-squared of the depth changes
  with and without the moon (also in `epoch_depth_comparison`). A real moon drives those
  depth changes; spots do not. Epochs where the moon may be over the planet disk (for the
  coplanar orbit of the template) are drawn hollow and not counted: a moon there would
  have its dip hidden, but whether it is over the disk depends on its orbit inclination,
  which a few degrees change and EMLS does not know (on a coplanar KOI-189.01 Rp/5
  injection, Pandora gives no dip in 3 of 36 epochs). The
  15 extra hypothesis searches of the old figures are down to 3; the overview's SDE panel
  title carries the reported SDE values.
- EMLS fits one planet depth correction per epoch with each moon hypothesis
  (`fit_planet_epoch_depth`, default on), instead of one shared by all epochs.
  Real Kepler transits change depth between epochs: on KOI-189.01, KOI-125.01,
  KOI-855.01 and KOI-686.01 the mean in-transit residual scatters about twice as
  much between epochs as on the pseudo-epoch nulls or on identical synthetic
  planets processed through the whole pipeline, and the first and second halves
  of each transit move together (r 0.73-0.86), so whole transits are deeper or
  shallower. The nulls did not have that scatter, so the FAP was optimistic.
  A moon separates from the epoch depths because it covers a different part of
  each transit: on the Rp/5 injections the best SNR any method can reach drops
  by 8-15 %. `planet_epoch_depths` reports them at the best hypothesis.
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

- EMLS catalog driver: a planet without a catalog mass gets the same mass as in the
  `sherlockpipe.emls` command (Chen & Kipping forecast, one Jupiter mass for giants)
  instead of R^1.5, which made giants about 8 times too light and their Hill sphere,
  widest stable moon orbit and moon period grid too small (audit item F7). The
  eccentricity comes from the KOI catalog (`koi_eccen`, `koi_longp`) when present, else
  the orbit is circular, instead of e = 0.15 and w = 0 for every target, which lowered
  the outer moon limit from 0.49 to 0.41 Hill radii (F8). The TOI b = 0 start (B5) needs
  no change: the planet fold fit frees b.
- EMLS catalog driver: the companion filter checks every CONFIRMED or CANDIDATE planet of the system, not only those above the SNR cut, so a faint close companion (KOI-1070.03, SNR 5.2, next to KOI-1070.02) now drops its neighbour; rows without a period or radius are skipped. The Hill-Roche check asks that the Domingos et al. (2006) outer limit EMLS searches up to, 0.4895 (1 - 1.0305 e) r_H, lies beyond the fluid Roche limit, instead of an unsourced r_H > 2 r_Roche that ignored the eccentricity (the same for a circular orbit: r_H > 2.04 r_Roche). KOIs with SNR > 15: 730 -> 728 (KOI-1070.02 and KOI-2092.01 removed); the first targets are unchanged.
- EMLS catalog driver: the tidal survival filter is Barnes & O'Brien (2002) Eq. 8, applied
  per target as in HEK VI (Teachey, Kipping & Schmitt 2018) and Kipping & Yahalomi (2023, MNRAS 518, 3482): the
  largest moon that survives tidal migration for 5 Gyr, with the EMLS outer moon limit as
  a_crit and log10(k2/Q) = -2.90 - 0.187 Rp/R_earth (kept at Jupiter's value above
  11.2 R_earth), converted to a radius with the Chen & Kipping (2017) relation. A target is
  dropped when that radius is below the smallest moon searched (Rp/20), and the largest
  moon searched is capped at it. The previous rule, P > 10 d M^(1/6) age^(1/4), is not in
  the paper it cited (Heller et al. 2014) and grew stricter with planet mass, the opposite
  of tidal theory (audit item F6). With the Hill-Roche and companion cuts, the 2882 KOIs
  with SNR > 15 go to 730 targets (1046 before), and almost none below 10 d survive, as
  Dobos et al. (2021) find. The filter takes planet masses from the Chen & Kipping mean
  relation (`mean_planet_mass`, within about 10 % of the forecast median); only the
  targets searched get the forecast, which costs about 0.3 s a planet, so the selection
  takes seconds instead of about 12 minutes.
- EMLS in-transit coverage check counts the points within T14 of the fitted
  mid-time. It counted the points where the planet model dims, which with the
  exposure modelled reaches half an exposure beyond T14, so on Kepler long cadence
  an epoch one point short could pass: KOI-125.01 kept T0 1306.2 with 7 of the 10
  expected points (limit 7.7), an epoch after a data gap with a ramp of about
  1500 ppm inside the transit.
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
  reported impact corrections of 10.6 and 26.1; they are now +0.10 and +0.16 in b, with
  the same SNR and FAP. `planet_shape_correction_large` now marks a b^2 correction
  beyond 0.1, the range of the b^2 column (a change of b of 0.1 at b = 0.5).
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
