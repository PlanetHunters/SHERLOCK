# EMLS false alarms on synthetic planets without a moon (2026-11-11)

`synthetic_planet.py`, 20 offsets (independent noise stretches of the target's own curve) per row,
FAP from the 100 nulls of the control search of the same target and code version. Detected means
FAP upper < 0.01. With a calibrated FAP, about 1 of 20 cases falls below FAP 0.05.

pre = 3a26b90 (before the planet corrections), post = 5229b31 (planet depth + per-epoch t0 fitted
with each moon).

| target | condition | version | detected | FAP < 0.01 | FAP < 0.05 | SNR median / max | paired SNR change |
|---|---|---|---|---|---|---|---|
| KOI-189.01 | match | pre | 0 | 0 | 0 | 2.49 / 3.53 | |
| KOI-189.01 | match | post | 0 | 0 | 0 | 3.67 / 4.50 | +1.11 |
| KOI-189.01 | ld | pre | 0 | 0 | 10 | 4.07 / 4.75 | |
| KOI-189.01 | ld | post | **10** | 11 | 17 | 5.25 / 6.59 | +1.28 |
| KOI-189.01 | smear | pre | 0 | 0 | 0 | 2.56 / 3.67 | |
| KOI-189.01 | smear | post | 0 | 0 | 4 | 4.10 / 4.92 | +1.64 |
| Kepler-1514 b | match | pre | 0 | 1 | 4 | 2.40 / 4.07 | |
| Kepler-1514 b | match | post | 0 | 0 | 3 | 2.48 / 4.12 | +0.06 |
| Kepler-1514 b | ld | pre | 0 | 0 | 2 | 2.27 / 3.80 | |
| Kepler-1514 b | ld | post | 0 | 0 | 2 | 2.30 / 3.80 | +0.07 |
| Kepler-1514 b | smear | pre | 0 | 0 | 4 | 2.37 / 3.99 | |
| Kepler-1514 b | smear | post | 0 | 0 | 2 | 2.45 / 3.99 | +0.07 |

Conditions: match = the model EMLS subtracts; ld = limb darkening u1 + 0.15, u2 - 0.15 (EMLS keeps
the catalog values fixed); smear = integrated over 29.4 min while EMLS runs without exposure_time.

Findings:
- A limb darkening error on a deep planet (KOI-189.01, 1.8 % depth, 29-38 epochs) is taken as a
  moon: 10 of 20 false detections after the planet corrections, none before (but already 10 of 20
  at FAP < 0.05 before). The moon window of KOI-189.01 always sits inside the transit, where the
  limb darkening residual is.
- Smearing adds a milder excess after the corrections (4 of 20 at FAP < 0.05).
- Kepler-1514 b (red noise factor about 4, 3-7 epochs) shows no change: the residuals are below its
  noise.
- With the exact model (match) the false-alarm rate stays nominal in both versions, although the
  corrections raise the SNR of KOI-189.01 by about 1.1 on the same noise.
