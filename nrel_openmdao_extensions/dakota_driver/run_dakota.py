import os
import subprocess
import textwrap
import numpy as np


def setup_directories(template_dir):
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        
    subprocess.call("rm *.in", shell=True)
    subprocess.call("rm *.out", shell=True)
    subprocess.call("rm *.rst", shell=True)
    subprocess.call("rm *.dat", shell=True)
    subprocess.call("rm -rf run_history", shell=True)

def create_input_file(template_dir, desvars, outputs, bounds):
    flattened_bounds = []

    for key, value in bounds.items():
        if isinstance(value, (float, list)):
            value = np.array(value)
        flattened_value = np.squeeze(value.flatten()).reshape(-1, 2)
        flattened_bounds.extend(flattened_value)

    flattened_bounds = np.array(flattened_bounds)
    
    desvar_labels = []
    for key, value in desvars.items():
        if isinstance(value, (float, list)):
            value = np.array(value)
        flattened_value = np.squeeze(value.flatten())
        key = key.replace('.', '_')
        for i in range(len(flattened_value)):
            desvar_labels.append(f'{key}_{i}')
            
    desvar_shapes = {}
    total_size = 0

    for key, value in desvars.items():
        if isinstance(value, (float, list)):
            value = np.array(value)

        desvar_shapes[key] = value.shape
        total_size += value.size
    
    #### Flatten the DVs here and make names for each and append the names with the number according to the size of the DVs
    
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
    f'  continuous_design {len(flattened_bounds)}\n' + \
    f'  lower_bounds ' + " ".join([str(i) for i in flattened_bounds[:, 0]]) + '\n' + \
    f'  upper_bounds ' + " ".join([str(i) for i in flattened_bounds[:, 1]]) + \
    textwrap.dedent('''
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
    '  descriptors ' + " ".join(['\"' + key + '\"' for key in outputs]) + \
    '''
  no_gradients
  no_hessians
    '''

    with open("dakota_input.in", "w") as text_file:
        text_file.write(input_file)
    
    return desvar_labels, desvar_shapes
    
def create_input_yaml(template_dir, desvar_labels):
    # Populate input.yml
    input_lines = [f'cdv_{i+1}: {{cdv_{i+1}}}' for i in range(len(desvar_labels))]
    with open(template_dir + "input_template.yml", "w") as f:
        for line in input_lines:
            f.write(line + '\n')
        
def create_driver_file(template_dir, model_string, desvar_shapes, desired_outputs, output_scalers):
    desvar_shapes_lines = []
    for key in desvar_shapes:
        desvar_shapes_lines.append(f'desvar_shapes["{key}"] = {desvar_shapes[key]}')
    desvar_shapes_lines = """
{}
    """.format("\n".join(desvar_shapes_lines))
    
    desired_outputs_string = 'desired_outputs = [' + " ".join(['\"' + key + '\"' for key in desired_outputs]) + ']'
    
    write_outputs_string = []
    for i, key in enumerate(desired_outputs):
        string = f'f.write(str(float(outputs["{key}"]) * {output_scalers[i]}))'
        write_outputs_string.append(string)
    print(write_outputs_string)
    write_outputs_string = """
    {}
    """.format("\n".join(write_outputs_string))
        
    # Create openmdao_driver.py
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
        
    desvars_list = []
    for key in desvars:
        desvars_list.append(float(desvars[key]))
    flattened_desvars = np.array(desvars_list)
    
    desvar_shapes = {}
    ''') + \
    desvar_shapes_lines + \
    textwrap.dedent('''
    size_counter = 0
    desvars = {}
    for key, shape in desvar_shapes.items():
        size = int(np.prod(shape))
        desvars[key] = flattened_desvars[
            size_counter : size_counter + size
        ].reshape(shape)
        size_counter += size

    print()
    print('Design variables:')
    print(desvars)
    ''') + \
    model_string + '\n' + \
    textwrap.dedent('''\
    model_instance = model(desvars)
    outputs = model_instance.compute(desvars)
    #########################################
    #                                       #
    #    Step 3: Write Output in format     #
    #    Dakota expects                     #
    #                                       #
    #########################################
    ''') + \
    desired_outputs_string + '\n' + \
    textwrap.dedent('''\
    print('Outputs:')
    print(outputs)
    # Write it to the expected file.
    with open(sys.argv[2], "w") as f:''') + \
    write_outputs_string

    with open(template_dir + "openmdao_driver.py", "w") as text_file:
        text_file.write(driver_file)

def run_dakota():
    subprocess.call("dakota -i dakota_input.in -o dakota_output.out -write_restart dakota_restart.rst", shell=True)
    subprocess.call("dakota_restart_util to_tabular dakota_restart.rst dakota_data.dat", shell=True)
    
def do_full_optimization(template_dir, desvars, desired_outputs, bounds, model_string, output_scalers):
    setup_directories(template_dir)
    desvar_labels, desvar_shapes = create_input_file(template_dir, desvars, desired_outputs, bounds)
    create_input_yaml(template_dir, desvar_labels)
    create_driver_file(template_dir, model_string, desvar_shapes, desired_outputs, output_scalers)
    run_dakota()


if __name__ == "__main__":
    template_dir = 'template_dir/'
    desvars = {'x' : np.array([0.25])}
    bounds = {'x' : np.array([[0.0, 1.0]])}
    outputs = ['y']
    output_scalers = [1.] 
    
    model_string = 'from multifidelity_studies.models.testbed_components import simple_1D_high_model as model'
    
    do_full_optimization(template_dir, desvars, outputs, bounds, model_string, output_scalers)