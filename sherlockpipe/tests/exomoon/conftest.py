import functools

import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares


@pytest.fixture(autouse=True)
def instantaneous_exposure_by_default(monkeypatch):
    """Build EMLS with exposure_time=None unless a test passes it.

    The synthetic curves of these tests are instantaneous batman models at 4 to 10 min
    cadence, so the class default exposure_time="auto" (the median point spacing) would
    model them with the wrong exposure. Tests of the default pass exposure_time="auto".
    """
    original = ExoMoonLeastSquares.__init__

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        kwargs.setdefault("exposure_time", None)
        original(self, *args, **kwargs)

    monkeypatch.setattr(ExoMoonLeastSquares, "__init__", init)
