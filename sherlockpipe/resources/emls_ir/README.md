# EMLS injection-recovery calibration (outdated)

These files come from an injection-recovery campaign run with the EMLS code before the audit fixes
merged in PlanetHunters/SHERLOCK PR 5 and the follow-up branches (doubled first epoch, out-of-transit
weighting, placeholder SDE scores, depth gate, instantaneous models at long cadence and others).
No code in the repository reads them: the `sherlockpipe.exomoon_ir` assessment module was never
committed, and its call was removed (audit item I6).

| Path | Status |
|---|---|
| `grid.json` | Reusable. The list of injected moons (radius, moon period, alpha, number of transits, planet period, baseline, Roche and Hill limits). It records neither the noise level nor the cadence, so the old campaign cannot be reproduced exactly from it. |
| `ir_results/results.csv` | Outdated. Recovery results from the pre-fix code. |
| `nulls/nulls_N_tr_*.npz` | Outdated. Noise-only statistics from the pre-fix code. |
| `calibration/sde_thresholds.json`, `calibration/completeness.json` | Outdated. Derived from the files above, and set on SDE, which does not separate moons from noise on real data (audit item H7). |
| `plots/` | Outdated. Figures of the results above. |

A new calibration (audit items H10 and I7) needs new campaign code, run on the current EMLS, with the
noise level and cadence recorded next to the grid. Do not use these thresholds as a significance.
