import numpy as np
from mpi4py import MPI
import openmdao.api as om


rosenbrock_size = 4
num_procs = 2


class IntermittentComponent(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_iterations_between_calls', 3)
        self.frozen_outputs = {}
        self.actual_compute_calls = 0
    
    def compute(self, inputs, outputs):
        """
        This is a wrapper for the actual compute call, `internal_compute()`,
        which calls the compute method every `num_iterations_between_calls`.
        This allows more expensive analyses to be run less often at the expense
        of accuracy in an optimization context.
        If you want the compute to be run only once at the beginning of the
        optimization, set `num_iterations_between_calls` to a very large number.
        """
        
        num_iterations_between_calls = self.options['num_iterations_between_calls']
        
        # Determine if we are in a compute call in which we want to update the
        # frozen outputs
        regular_compute = (self.iter_count_without_approx % num_iterations_between_calls) == 0 and not self.under_approx
        approx_compute = ((self.iter_count_without_approx-1) % num_iterations_between_calls) == 0 and self.under_approx
        
        # If we're within one of those types of compute calls, call the actual
        # internal_compute() method to update the results
        if regular_compute or approx_compute:
            self.internal_compute(inputs, outputs)
            self.actual_compute_calls += 1
            
            # Save off the results to the frozen_outputs dict
            for key in outputs:
                self.frozen_outputs[key] = outputs[key]
        
        # If we're using the frozen results, simply set the outputs from those results
        else:
            for key in outputs:
                outputs[key] = self.frozen_outputs[key]
                
    def internal_compute(self, inputs, outputs):
        """
        This is the actual method where the computations should occur.
        """
        raise NotImplementedError("Please define an `internal_compute` method.")

class Rosenbrock1(IntermittentComponent):
    
    def initialize(self):
        # This is required only if you have something to add to the initialize(),
        # such as OpenMDAO options declarations
        super().initialize()
        self.options.declare('multiplier', 1.)
        
    def setup(self):
        # This is required to run the original setup() contained within IntermittentComponent
        super().setup()
        
        self.add_input("x", np.ones(rosenbrock_size))
        self.add_output("f1", 0.0)
                
    def internal_compute(self, inputs, outputs):
        """
        This is the actual method where the computations should occur.
        """
        x = inputs["x"]
        x_0 = x[:-1]
        x_1 = x[1:]
        outputs["f1"] = self.options['multiplier'] * sum((1 - x_0) ** 2)
        
class Rosenbrock2(IntermittentComponent):
    
    def initialize(self):
        # This is required only if you have something to add to the initialize(),
        # such as OpenMDAO options declarations
        super().initialize()
        self.options.declare('multiplier', 100.)
    
    def setup(self):
        # This is required to run the original setup() contained within IntermittentComponent
        super().setup()
        
        self.add_input("x", np.ones(rosenbrock_size))
        self.add_output("f2", 0.0)
                
    def internal_compute(self, inputs, outputs):
        """
        This is the actual method where the computations should occur.
        """
        x = inputs["x"]
        x_0 = x[:-1]
        x_1 = x[1:]
        outputs["f2"] = self.options['multiplier'] * sum((x_1 - x_0 ** 2) ** 2)


prob = om.Problem(model=om.Group(num_par_fd=num_procs))
prob.model.approx_totals(method='fd')
indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
indeps.add_output('x', 1.2*np.ones(rosenbrock_size))

prob.model.add_subsystem('rosenbrock1', Rosenbrock1(num_iterations_between_calls=900, multiplier=1.), promotes=['*'])
prob.model.add_subsystem('rosenbrock2', Rosenbrock2(num_iterations_between_calls=1, multiplier=100.), promotes=['*'])
prob.model.add_subsystem('objective_comp', om.ExecComp('f = f1 + f2'), promotes=['*'])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.opt_settings['tol'] = 1e-9

prob.model.add_design_var('x', lower=-1.5, upper=1.5)
prob.model.add_objective('f')

prob.setup()
prob.run_driver()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # minimum value
    print(f"Optimum found: {prob['f'][0]}")
    # location of the minimum
    print(f"Optimal design: {prob['x']}")
    print()
    print(f'Number of actual compute calls: {prob.model.rosenbrock1.actual_compute_calls} and {prob.model.rosenbrock2.actual_compute_calls}')