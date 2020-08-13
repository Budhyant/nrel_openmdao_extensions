import os
import subprocess
import textwrap


def setup_directories(template_dir):
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        
    subprocess.call("rm *.in", shell=True)
    subprocess.call("rm *.out", shell=True)
    subprocess.call("rm *.rst", shell=True)
    subprocess.call("rm *.dat", shell=True)
    subprocess.call("rm -rf run_history", shell=True)

def create_input_file(template_dir, desvars, outputs):
      # Terrible string-list manipulation to get the DVs and outputs formatted correctly
    input_file = textwrap.dedent('''\
    # Dakota input file
    environment
      tabular_data
        tabular_data_file "dakota_data.dat"

    method
      coliny_cobyla 

    variables
    ''') + \
    f'  continuous_design {len(desvars)}\n' \
    '  descriptors ' + " ".join(['\"' + i + '\"' for i in desvars]) + \
    textwrap.dedent('''
      lower_bounds 3 2
      upper_bounds 8 10

    interface
      fork
        asynchronous
        evaluation_concurrency 1
        parameters_file "params.in"
        results_file "results.out"
    ''') + \
    f'    copy_files "{template_dir}*"' + \
    textwrap.dedent('''
        analysis_driver "python openmdao_driver.py"

        work_directory
          named "run_history/run"
          directory_tag
          directory_save
          file_save

    responses
    ''') + \
    f'  objective_functions {len(outputs)}\n' \
    '  descriptors ' + " ".join(['\"' + i + '\"' for i in outputs]) + \
    '''
  no_gradients
  no_hessians
    '''


    with open("dakota_input.in", "w") as text_file:
        text_file.write(input_file)
    
def create_input_yaml(template_dir, desvars):
    # Populate input.yml
    input_lines = [f'{i}: {{{i}}}' for i in desvars]
    with open(template_dir + "input_template.yml", "w") as f:
        for line in input_lines:
            f.write(line + '\n')
        
def create_driver_file(template_dir):
    # Create openmdao_driver.py
    analysis_text_block = textwrap.dedent('''\
    import openmdao.api as om

    prob = om.Problem()

    prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

    prob.setup()

    prob.set_val('paraboloid.x', float(desvars['x']))
    prob.set_val('paraboloid.y', float(desvars['y']))

    prob.run_model()

    # minimum value
    obj = prob.get_val('paraboloid.f')[0]

    outputs = [obj]

    print(float(desvars['x']), float(desvars['y']), obj)
    ''')

    driver_file = textwrap.dedent('''\
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
        desvars = safe_load(f)
    ''') + \
    analysis_text_block + \
    textwrap.dedent('''\
    #########################################
    #                                       #
    #    Step 3: Write Output in format     #
    #    Dakota expects                     #
    #                                       #
    #########################################

    # Write it to the expected file.
    with open(sys.argv[2], "w") as f:
        for output in outputs:
            f.write(str(output) + '\\n')
    ''')

    with open(template_dir + "openmdao_driver.py", "w") as text_file:
        text_file.write(driver_file)

def run_dakota():
    subprocess.call("dakota -i dakota_input.in -o dakota_output.out -write_restart dakota_restart.rst", shell=True)
    subprocess.call("dakota_restart_util to_tabular dakota_restart.rst dakota_data.dat", shell=True)
    
    
if __name__ == "__main__":
    template_dir = 'template_dir/'
    desvars = ['x', 'y']
    outputs = ['obj']
    
    setup_directories(template_dir)
    create_input_file(template_dir, desvars, outputs)
    create_input_yaml(template_dir, desvars)
    create_driver_file(template_dir)
    run_dakota()