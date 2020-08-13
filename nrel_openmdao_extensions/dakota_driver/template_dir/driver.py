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

obj = x**2 + y**2
c1 = np.sqrt(x + y) - 3
print(x, y, obj, c1)

#########################################
#                                       #
#    Step 3: Write Output in format     #
#    Dakota expects                     #
#                                       #
#########################################

# Write it to the expected file.
with open(sys.argv[2], "w") as fp:
    fp.write(str(obj) + '\n')
    fp.write(str(c1))
