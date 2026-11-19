"""Progress lines of the silenced null and scrambled loops: a 1000-null run used to log nothing for
hours, because the loop disables logging below WARNING to keep each search quiet."""
import logging

import numpy as np

from sherlockpipe.exomoon import nulls
from sherlockpipe.exomoon.nulls import NullDistribution


class StubNulls(NullDistribution):
    """Only what run_nulls needs, with a fake curve and statistic."""
    null_runs = 25
    null_seed = 0
    null_stop_fap = None
    planet_mass_grid = np.array([1.0])
    planet_mass = 1.0
    planet_period = 100.0
    planet_impact_param = 0.1

    def trim_half_window(self, mass):
        return 1.0

    def search_input(self, *args):
        class Window:
            half_moon_max_orbit_transit_duration = 1.0
        return Window()

    def null_offsets(self, trim_half_window, exclusion):
        return np.arange(3.0, 90.0, 3.0)

    def null_valid_offsets(self, before_trim, trim_half_window, offsets):
        return [offsets] * 4

    def independent_offsets(self, valid, trim_half_window):
        return len(valid)

    def null_curve(self, before_trim, rng, trim_half_window, valid_offsets):
        return (np.zeros(3), np.ones(3), np.ones(3), [0.0], None)

    def null_statistic(self, before_trim, *curve):
        self.calls = getattr(self, 'calls', 0) + 1
        return np.nan if self.calls % 5 == 0 else float(self.calls)


def progress_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("Null searches:")]


def test_null_loop_logs_progress_about_every_tenth(caplog):
    caplog.set_level(logging.INFO)
    snrs = StubNulls().run_nulls(None, np.array([1.0, 2.0]))
    lines = progress_lines(caplog)
    # Every ceil(25 / 10) = 3 runs, not at the last one, which the summary line reports.
    assert [int(line.split()[2].split('/')[0]) for line in lines] == [3, 6, 9, 12, 15, 18, 21, 24]
    assert "24/25 done, 4 failed" in lines[-1] and "min left" in lines[-1]
    assert len(snrs) == 20
    # The searches themselves stay silenced, and logging is restored afterwards.
    assert logging.root.manager.disable == logging.NOTSET


def test_progress_is_silent_when_logging_was_off_before(caplog):
    caplog.set_level(logging.INFO)
    previous = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        StubNulls().run_nulls(None, np.array([1.0, 2.0]))
    finally:
        logging.disable(previous)
    assert progress_lines(caplog) == []


def test_log_progress_step_and_estimate(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(nulls.clock, "monotonic", lambda: 600.0)
    logging.disable(logging.WARNING)
    try:
        for done in range(1, 1001):
            nulls.log_progress("Null searches", done, 1000, 0, [3.0, 5.0], 0.0, logging.NOTSET)
    finally:
        logging.disable(logging.NOTSET)
    lines = progress_lines(caplog)
    assert len(lines) == 9
    assert lines[0] == ("Null searches: 100/1000 done, 0 failed, SNR median 4.00, largest 5.00, "
                        "elapsed 10.0 min, about 90.0 min left")
