from spock import FeatureClassifier, DeepRegressor
import rebound
from sherlockpipe.nbodies.stability_calculator import StabilityCalculator
import pandas as pd


class SpockStabilityCalculator(StabilityCalculator):
    """
    Runs the stability computation by computing the stability probability and the median expected instability time for
    each scenario
    """
    def run_simulation(self, simulation_input):
        sim = self.init_rebound_simulation(simulation_input)
        featureClassifierModel = FeatureClassifier()
        deepRegressorModel = DeepRegressor()
        stability_probability = featureClassifierModel.predict_stable(sim)
        median, lower, upper = deepRegressorModel.predict_instability_time(sim, samples=10000)
        return {"star_mass": simulation_input.star_mass,
                "periods": ",".join([str(planet_period) for planet_period in simulation_input.planet_periods]),
                "masses": ",".join([str(mass_value) for mass_value in simulation_input.mass_arr]),
                "eccentricities": ",".join([str(ecc_value) for ecc_value in simulation_input.ecc_arr]),
                "stability_probability": stability_probability, "median_expected_instability_time": median}

    def store_simulation_results(self, simulation_results, results_dir):
        result_file = results_dir + "/stability_spock.csv"
        results_df = pd.DataFrame(columns=['star_mass', 'periods', 'masses', 'eccentricities', 'stability_probability',
                                           'median_expected_instability_time'])
        results_df = results_df.append(simulation_results, ignore_index=True)
        results_df = results_df.sort_values('stability_probability', ascending=False)
        results_df.to_csv(result_file, index=False)

# grid = 5
# sc = SpockStabilityCalculator()
# par_e = np.linspace(0.0, 0.7, grid)
# par_e1 = np.linspace(0.0, 0.7, grid)
# for i in par_e:
#     for j in par_e1:
#         sc.run(0.53, [PlanetInput(1.17, 0.01749, 11.76943, i), PlanetInput(1.37, 0.03088, 2.97, j), PlanetInput(2.45, 0, 3.9, 0)])
