"""Command line entrypoint for the EMLS exomoon search on a SHERLOCK results directory.

Example: ``python -m sherlockpipe.emls --candidate 3 --mask_candidates 1 2 4``
"""
from argparse import ArgumentParser

from sherlockpipe.exomoon_run import run_emls


def parse_args(argv=None):
    ap = ArgumentParser(description='Exomoon search (EMLS) around a SHERLOCK candidate')
    ap.add_argument('--object_dir', help="The SHERLOCK results directory. Defaults to the current one.",
                    required=False)
    ap.add_argument('--candidate', type=int, required=True, help="The candidate number to search moons around.")
    ap.add_argument('--mask_candidates', type=int, nargs='+', default=None,
                    help="Candidate numbers whose transits are masked before the search.")
    ap.add_argument('--planet_mass', type=float, default=None,
                    help="Planet mass in Earth masses. Forecast from the radius (Chen & Kipping 2017) if omitted.")
    ap.add_argument('--min_moon_radius', type=float, default=None,
                    help="Smallest moon radius in Earth radii. Default: planet radius / 20.")
    ap.add_argument('--max_moon_radius', type=float, default=None,
                    help="Largest moon radius in Earth radii. Default: planet radius / 2.")
    ap.add_argument('--period_grid_size', type=int, default=10000, help="Number of moon periods searched.")
    ap.add_argument('--radius_grid_size', type=int, default=20, help="Number of moon radius templates.")
    ap.add_argument('--curve', type=int, default=None,
                    help="Detrended curve number (lc_<curve>.csv). Default: the curve the candidate was found in.")
    ap.add_argument('--lc_file', default=None, help="Light curve CSV (time, flux, flux_err). Overrides --curve.")
    ap.add_argument('--no_fit', dest='use_fit', action='store_false',
                    help="Ignore the fit_[n] results even if they exist.")
    ap.add_argument('--exposure_time', type=float, default=None,
                    help="Exposure time in seconds. Default: the median point spacing of the light curve.")
    ap.add_argument('--max_moon_density', type=float, default=5.5, help="Maximum moon density in g/cm3.")
    ap.add_argument('--subtract_mode', choices=['subtract', 'mask'], default='subtract',
                    help="Whether the planet transit is subtracted or masked.")
    ap.add_argument('--cpus', type=int, default=None, help="The number of CPU cores to be used.")
    return ap.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    run_emls(args.object_dir, args.candidate, args.mask_candidates, args.planet_mass, args.min_moon_radius,
             args.max_moon_radius, args.period_grid_size, args.radius_grid_size, args.cpus, args.curve, args.lc_file,
             args.use_fit, args.max_moon_density, args.subtract_mode, args.exposure_time)
