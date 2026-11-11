# EMLS false alarms on synthetic planets without a moon (2026-11-11)

`synthetic_planet.py`, 20 offsets (independent noise stretches of the target's own curve) per row,
FAP from the 100 nulls of the control search of the same target and code version. Detected means
FAP upper < 0.01. With a calibrated FAP, about 1 of 20 cases falls below FAP 0.05.

pre = 3a26b90 (before the planet corrections), post = 5229b31 (planet depth + per-epoch t0 fitted
with each moon), shape = 7e408e9 (post + shared impact parameter and limb darkening columns).
Paired SNR change: pre to post on the post rows, post to shape on the shape rows.

| target | condition | version | detected | FAP < 0.01 | FAP < 0.05 | SNR median / max | paired SNR change |
|---|---|---|---|---|---|---|---|
| KOI-189.01 | match | pre | 0 | 0 | 0 | 2.49 / 3.53 | |
| KOI-189.01 | match | post | 0 | 0 | 0 | 3.67 / 4.50 | +1.11 |
| KOI-189.01 | match | shape | 0 | 0 | 1 | 3.55 / 4.70 | +0.08 |
| KOI-189.01 | ld | pre | 0 | 0 | 10 | 4.07 / 4.75 | |
| KOI-189.01 | ld | post | **10** | 11 | 17 | 5.25 / 6.59 | +1.28 |
| KOI-189.01 | ld | shape | 0 | 0 | 0 | 3.40 / 4.36 | -1.78 |
| KOI-189.01 | smear | pre | 0 | 0 | 0 | 2.56 / 3.67 | |
| KOI-189.01 | smear | post | 0 | 0 | 4 | 4.10 / 4.92 | +1.64 |
| KOI-189.01 | smear | shape | 0 | 0 | 2 | 3.98 / 4.89 | -0.15 |
| Kepler-1514 b | match | pre | 0 | 1 | 4 | 2.40 / 4.07 | |
| Kepler-1514 b | match | post | 0 | 0 | 3 | 2.48 / 4.12 | +0.06 |
| Kepler-1514 b | match | shape | 0 | 0 | 2 | 2.49 / 4.26 | +0.01 |
| Kepler-1514 b | ld | pre | 0 | 0 | 2 | 2.27 / 3.80 | |
| Kepler-1514 b | ld | post | 0 | 0 | 2 | 2.30 / 3.80 | +0.07 |
| Kepler-1514 b | ld | shape | 0 | 0 | 2 | 2.47 / 4.29 | +0.10 |
| Kepler-1514 b | smear | pre | 0 | 0 | 4 | 2.37 / 3.99 | |
| Kepler-1514 b | smear | post | 0 | 0 | 2 | 2.45 / 3.99 | +0.07 |
| Kepler-1514 b | smear | shape | 0 | 0 | 2 | 2.45 / 4.26 | +0.01 |

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

Shape columns (7e408e9):
- The KOI-189.01 limb darkening false alarms disappear: 0 of 20 detected and 0 at FAP < 0.05
  (from 10 and 17), paired SNR -1.78. Smearing drops to 2 of 20 at FAP < 0.05, nominal.
- Real-moon sensitivity is kept. Rp/5 injections (SNR post -> shape): Kepler-1514 b alpha 1, 2.5, 4:
  5.71 -> 5.88, 3.65 -> 3.91, 4.29 -> 4.43; KOI-189.01: 14.10 -> 13.73, 10.71 -> 10.43,
  12.52 -> 12.11. Same detections (4 of 6), radius recovery unchanged (1.8-2.2 R_earth for 2 R_earth).
- Controls stay undetected: Kepler-1514 b FAP 0.58, KOI-189.01 FAP 0.020 (upper 0.031; post 0.030).
- Run time per search is unchanged within noise (KOI-189.01 144 s vs 147 s, Kepler-1514 b 61 s vs 56 s).

First five Kepler catalog targets (2026-11-11, 100 nulls each):

| target | 7e408e9 SNR / FAP | c39252d SNR / FAP | shape check chi2 / 20 (beta) |
|---|---|---|---|
| Kepler-1514 b | 2.10 / 0.58 | 2.07 / 0.59 | 4.4 (4.2) |
| KOI-189.01 | 4.82 / 0.020 | 4.21 / 0.11 | 17.0 (1.15) |
| KOI-125.01 | 3.83 / 0.19 | 4.08 / 0.21 | 25.3 (1.45) |
| KOI-855.01 | 4.50 / 0.11 | 4.43 / 0.12 | 22.3 (1.06) |
| KOI-686.01 | 5.73 / 0.0033, detected | 4.57 / 0.064 | 34.8 (1.12) |

7e408e9 modelled every target without exposure (the class default was instantaneous); c39252d
measures 29.42 min from the point spacing. KOI-686.01 (ingress about 1 h) was a false detection
from the unsmeared planet; it is undetected with the exposure modelled, and every shape check
passes with the red-noise factor.

KOI-686.01 synthetic planets (90fe46a, nulls from the c39252d control; match and ld injected with
the 29.4 min exposure EMLS models):

| condition | detected | FAP < 0.01 | FAP < 0.05 | SNR median / max | FAP median |
|---|---|---|---|---|---|
| match | 0 | 0 | 2 | 4.26 / 4.90 | 0.14 |
| ld | 0 | 0 | 2 | 4.06 / 4.83 | 0.22 |

A first run with match and ld injected instantaneously while EMLS smeared its model gave 2 of 20
detections in each: an exposure mismatch in the other direction is a false alarm source too.

Kepler-1514 b and KOI-125.01 (b < 0.005) reported impact corrections of 10.6 and 26.1 in c39252d:
the b derivative of the flux vanishes for a central transit. 8b57f45 takes that column in b^2:
the corrections become +0.10 and +0.16 in b (b^2 +0.011 and +0.026), with SNR / FAP unchanged
(2.07 / 0.59 and 4.08 / 0.21). The large-correction flag is then set on the b^2 change (> 0.1),
which neither target reaches.
