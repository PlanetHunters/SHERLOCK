import os
import unittest

import numpy as np
import ellc


class TestsEllcExposureIntegration(unittest.TestCase):
    def test_ellc_is_the_installed_package(self):
        """Importing sherlockpipe or alexfitter must not replace the ellc module."""
        import sherlockpipe  # noqa: F401
        import alexfitter  # noqa: F401
        import ellc as resolved_ellc

        self.assertIs(resolved_ellc, ellc)
        package_dir = os.path.dirname(os.path.abspath(resolved_ellc.__file__))
        self.assertEqual("ellc", os.path.basename(package_dir))
        for vendoring_package in (sherlockpipe, alexfitter):
            vendoring_dir = os.path.dirname(os.path.abspath(vendoring_package.__file__))
            self.assertFalse(package_dir.startswith(vendoring_dir + os.sep),
                             f"ellc loaded from inside {vendoring_dir}: {package_dir}")

    def test_exposure_integrated_models_average_all_sub_exposures(self):
        times = np.linspace(-0.1, 0.1, 2001)
        exposure_time = 0.02
        integration_points = 10
        model_parameters = dict(
            radius_1=0.1,
            radius_2=0.01,
            sbratio=0.0,
            incl=90.0,
            t_zero=0.0,
            period=3.1,
            ld_1="quad",
            ldc_1=[0.4, 0.2],
            verbose=False,
        )
        offsets = (
            np.arange(integration_points) / (integration_points - 1) - 0.5
        ) * exposure_time
        weights = np.ones(integration_points) / (integration_points - 1)
        weights[[0, -1]] *= 0.5

        models = (
            (ellc.lc, model_parameters),
            (ellc.fluxes, model_parameters),
            (ellc.rv, model_parameters | {"a": 10, "q": 0.5}),
        )
        for model, parameters in models:
            with self.subTest(model=model.__name__):
                actual = model(
                    times,
                    t_exp=exposure_time,
                    n_int=integration_points,
                    **parameters,
                )
                expected = [
                    model(times + offset, **parameters)
                    for offset in offsets
                ]
                if isinstance(actual, tuple):
                    expected = tuple(
                        sum(weight * sample[component]
                            for weight, sample in zip(weights, expected))
                        for component in range(len(actual))
                    )
                else:
                    expected = sum(
                        weight * sample
                        for weight, sample in zip(weights, expected)
                    )

                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
