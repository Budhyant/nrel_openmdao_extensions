import numpy as np
from scipy.optimize import minimize
import openmdao.api as om


def compute_power(pitch_angle, wind_speed, drag_modifier):
    CD = np.pi * drag_modifier * np.deg2rad(pitch_angle) ** 2
    airfoil_power_boost = (drag_modifier - wind_speed * 2.0) ** 2.0 / 10.0
    return -((wind_speed - CD) ** 3) - airfoil_power_boost


def compute_power_constraint(pitch_angle, wind_speed, drag_modifier, P_rated):
    neg_power = compute_power(pitch_angle, wind_speed, drag_modifier)
    return neg_power + P_rated


class ComputePitchAngles(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("size")
        self.options.declare("P_rated")

    def setup(self):
        size = self.options["size"]

        self.add_input("wind_speeds", np.zeros(size))
        self.add_input("drag_modifier", 11.0)

        self.add_output("pitch_angles", np.zeros(size))
        self.add_output("powers", np.zeros(size))
        self.add_output("total_power")

    def compute(self, inputs, outputs):
        P_rated = self.options["P_rated"]
        drag_modifier = inputs["drag_modifier"]

        for i, wind_speed in enumerate(inputs["wind_speeds"]):
            constraints = [
                {
                    "type": "ineq",
                    "fun": compute_power_constraint,
                    "args": [wind_speed, drag_modifier, P_rated],
                }
            ]
            result = minimize(
                compute_power,
                1.0,
                args=(wind_speed, drag_modifier),
                method="SLSQP",
                bounds=[(-15.0, 15.0)],
                options={"disp": False},
                constraints=constraints,
            )
            outputs["pitch_angles"][i] = result.x
            outputs["powers"][i] = result.fun

        outputs["total_power"] = np.sum(outputs["powers"])


if __name__ == "__main__":
    wind_speeds = [4.0, 6.0, 8.0, 10.0]
    P_rated = 500.0

    prob = om.Problem()
    prob.model.add_subsystem(
        "compute_pitch_angles",
        ComputePitchAngles(size=len(wind_speeds), P_rated=P_rated),
        promotes=["*"],
    )

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.model.approx_totals(method="fd")

    prob.model.add_design_var("drag_modifier", lower=6.0, upper=15.0)
    prob.model.add_objective("total_power")

    prob.setup()

    prob.set_val("wind_speeds", wind_speeds)
    prob.run_driver()

    prob.model.list_outputs(print_arrays=True)
