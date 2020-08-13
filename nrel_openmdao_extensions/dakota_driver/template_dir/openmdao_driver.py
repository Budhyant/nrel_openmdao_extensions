# Import  modules
import sys
from subprocess import call
import numpy as np
from yaml import safe_load


#########################################
#                                       #
#    Step 1: Use Dakota created         #
#    input files to prepare for         #
#    model run.                         #
#                                       #
#########################################
input_template = "input_template.yml"
inputs = "inputs.yml"
call(["dprepro", sys.argv[1], input_template, inputs])
call(['rm', input_template])

#########################################
#                                       #
#    Step 2: Run Model                  #
#                                       #
#########################################
# Load parameters from the yaml formatted input.
with open(inputs, "r") as f:
    params = safe_load(f)
    x = float(params["x"])
    y = float(params["y"])

import openmdao.api as om

# build the model
prob = om.Problem()

prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

prob.setup(derivatives=False)

# Set initial values.
prob.set_val('paraboloid.x', x)
prob.set_val('paraboloid.y', y)

prob.run_model()

# minimum value
obj = prob.get_val('paraboloid.f')[0]

print(x, y, obj)

#########################################
#                                       #
#    Step 3: Write Output in format     #
#    Dakota expects                     #
#                                       #
#########################################

# Write it to the expected file.
with open(sys.argv[2], "w") as fp:
    fp.write(str(obj) + '\n')
