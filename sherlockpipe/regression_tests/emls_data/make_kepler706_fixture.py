"""Write the Kepler-706 (KIC 6522242) PDCSAP windows used by the realistic EMLS end-to-end test."""
import sys, warnings, numpy as np, pandas as pd, lightkurve as lk
warnings.filterwarnings('ignore')
PERIOD, EPOCH_BKJD, HALF_WINDOW = 41.408316686, 2455028.78745 - 2454833, 3.0
lcs = lk.search_lightcurve('KIC 6522242', mission='Kepler', author='Kepler', exptime=1800).download_all(flux_column='pdcsap_flux')
lc = lcs.stitch().remove_nans()
time, flux, err = (np.asarray(v, float) for v in (lc.time.value, lc.flux.value, lc.flux_err.value))
phase = (time - EPOCH_BKJD + PERIOD / 2) % PERIOD - PERIOD / 2
keep = np.abs(phase) < HALF_WINDOW
pd.DataFrame({'time': np.round(time[keep], 7), 'flux': np.round(flux[keep], 7), 'flux_err': np.round(err[keep], 7)}).to_csv(
    sys.argv[1], index=False, compression='gzip')
print(keep.sum(), 'points')
