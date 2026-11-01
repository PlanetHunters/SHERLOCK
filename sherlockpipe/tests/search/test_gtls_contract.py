from importlib.metadata import distribution
from pathlib import Path
import subprocess
import sys

import pytest


def test_pypi_gtls_distribution_version_and_location():
    installed = distribution("gputls")
    assert installed.version == "0.6.1"
    package_path = Path(installed.locate_file("gputls")).resolve()
    assert package_path.is_dir()
    assert "sherlockpipe" not in package_path.parts


def test_pypi_public_api_when_cuda_dependency_is_available():
    pytest.importorskip("gputls")
    from gputls import gtls

    assert callable(gtls)
    assert hasattr(gtls, "power")


def test_importing_gpu_search_adapter_does_not_import_gtls():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import sherlockpipe.search.GtlsSearcher; assert 'gputls' not in sys.modules"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
