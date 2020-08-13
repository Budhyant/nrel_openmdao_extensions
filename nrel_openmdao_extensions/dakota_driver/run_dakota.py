import subprocess


input_file = '''
# Dakota input file
environment
  tabular_data
    tabular_data_file "dakota_data.dat"

method
  coliny_cobyla 

variables
  continuous_design 2
  descriptors "x" "y"
  lower_bounds 3 2
  upper_bounds 8 10

interface
  fork
    asynchronous
    evaluation_concurrency 1
    parameters_file "params.in"
    results_file "results.out"
    copy_files "template_dir/*"

    analysis_driver "python openmdao_driver.py"

    work_directory
      named "run_history/run"
      directory_tag
      directory_save
      file_save

responses
  objective_functions = 1
  # nonlinear_equality_constraints = 1
  descriptors "obj"
  no_gradients
  no_hessians
'''

subprocess.call("rm *.in", shell=True)
subprocess.call("rm *.out", shell=True)
subprocess.call("rm *.rst", shell=True)
subprocess.call("rm *.dat", shell=True)
subprocess.call("rm -rf run_history", shell=True)

with open("dakota_input.in", "w") as text_file:
    text_file.write(input_file)

subprocess.call("dakota -i dakota_input.in -o dakota_output.out -write_restart dakota_restart.rst", shell=True)
subprocess.call("dakota_restart_util to_tabular dakota_restart.rst dakota_data.dat", shell=True)