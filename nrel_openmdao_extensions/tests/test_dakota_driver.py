import unittest
import numpy as np
from nrel_openmdao_extensions.dakota_driver.run_dakota import do_full_optimization
from openmdao.utils.assert_utils import assert_near_equal

try:
    import dakota
except ImportError:
    dakota = None


@unittest.skipIf(dakota is None, "only run if Dakota is installed.")
class TestDakotaOptimization(unittest.TestCase):

    def test_2D_opt_max_iterations(self):
        bounds = {'x' : np.array([[0.0, 1.0], [0.0, 1.0]])}
        desvars = {'x' : np.array([0., 0.25])}
        outputs = ['y']
        template_dir = 'template_dir/'
        model_string = 'from multifidelity_studies.models.testbed_components import simple_2D_high_model as model'
        output_scalers = [1.]
        options = {'method' : 'coliny_cobyla',
            'max_function_evaluations' : 3}
    
        do_full_optimization(template_dir, desvars, outputs, bounds, model_string, output_scalers, options)
    
        obj_values = []
        with open('dakota_data.dat') as f:
            for i, line in enumerate(f):
                if i > 0:
                    obj_values.append(float(line.split()[4]))
    
        assert_near_equal(np.min(np.array(obj_values)), -9.5)
        
    def test_2D_opt_EGO(self):
        bounds = {'x' : np.array([[0.0, 1.0], [0.0, 1.0]])}
        desvars = {'x' : np.array([0., 0.25])}
        outputs = ['y']
        template_dir = 'template_dir/'
        model_string = 'from multifidelity_studies.models.testbed_components import simple_2D_high_model as model'
        output_scalers = [1.]
        options = {'initial_samples' : 5,
                   'method' : 'efficient_global',
                   'seed' : 123456}
        
        do_full_optimization(template_dir, desvars, outputs, bounds, model_string, output_scalers, options)
        
        obj_values = []
        with open('dakota_data.dat') as f:
            for i, line in enumerate(f):
                if i > 0:
                    obj_values.append(float(line.split()[4]))
                    
        assert_near_equal(np.min(np.array(obj_values)), -9.999996864)
        

if __name__ == "__main__":
    unittest.main()
