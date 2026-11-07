"""Regression tests for audit items H15 (radius above the searched range) and I6 (dead assessment step).

H15: the reported radius is measured from the dip depth, so a dip deeper than the largest searched
moon is reported as measured and flagged. I6: the assessment module was never in the repository, so
the run states that no false-alarm assessment is applied instead of logging a traceback.
"""
import logging
import os

from sherlockpipe.tests.exomoon import test_search_integrity as guard

MAX_RADIUS = 4.0  # the guard system searches moons of 1-4 R_earth


def test_dip_deeper_than_the_searched_range_is_flagged(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        emls = guard.execute(tmp_path, moon_radius=6.0)

    assert emls.best_snr_radius > MAX_RADIUS
    assert emls.best_snr_radius_above_range
    assert any('above the searched range' in record.getMessage() for record in caplog.records)


def test_moon_inside_the_searched_range_is_not_flagged(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        emls = guard.execute(tmp_path, moon_radius=3.0)

    assert emls.best_snr_radius <= MAX_RADIUS
    assert not emls.best_snr_radius_above_range
    assert not any('above the searched range' in record.getMessage() for record in caplog.records)


def test_run_states_that_no_assessment_is_applied(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        guard.execute(tmp_path)

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith('No false-alarm assessment is applied') for message in messages)
    assert not any('exomoon_ir' in message or 'assess_candid' in message for message in messages)
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    assert not os.path.exists(tmp_path / 'emls_assessment.json')
