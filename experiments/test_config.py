from experiments.evolution_config import *
import numpy as np

number_of_tests = 1

#neuron params
neuron_runtime = 20
neuron_pop_size = 5
neuron_parallel_runs = 2

test_data_set = []
config = ''
runtime = 0
stochastic = False
negative_reward = False
exposure_time = 100
rate_on = 100
rate_off = 500
max_fail_score = 0

if exec_thing == 'xor':
    inputs = 2
    outputs = 2
    exposure_time = 100
    max_current = 6. # 0Hz @ 0.6 - 1000hz @ 13.3
    spike_controlled = True
    config = 'xor_tf '
elif exec_thing == 'logic':
    truth_table = [0, 1, 1, 0] # done
    # truth_table = [0, 0, 0, 1] # done
    # truth_table = [0, 1, 1, 1] # done
    # truth_table = [1, 0, 0, 1] # done
    # truth_table = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    # truth_table = [0, 1, 1, 0, 1, 0, 0, 1]
    inputs = int(np.ceil(np.log2(len(truth_table))))
    outputs = 2
    exposure_time = 100
    max_current = 6. # 0Hz @ 0.6 - 1000hz @ 13.3
    spike_controlled = True
    config = 'logic_tf '
elif exec_thing == 'pen':
    inputs = 4
    outputs = 2
    exposure_time = 250
    receptive_fields = 2
    receptive_width = 4.8
    max_current = 6. # 0Hz @ 0.6 - 1000hz @ 13.3
    max_pen_rate = 1000.
    repeat_test = 3
    spike_controlled = True
    inputs *= receptive_fields
    velocity_info = True
    if not velocity_info:
        inputs /= 2
    penalise_oscillations = True
    pen_cutoff = 100
    config = 'pen_tf '
else:
    print("\nNot a correct test setting\n")
    raise Exception
if plasticity:
    if plasticity == 'all':
        config += 'pall '
    else:
        config += 'pl '
if structural:
    config += 'strc '
if averaging_weights:
    config += 'ave '
# if make_action:
#     config += 'action '
if spike_f:
    if spike_f == 'out':
        config += 'out-spikes '
    else:
        config += 'spk{} '.format(spike_f)
if size_f:
    config += 'size '
if reward_shape:
    config += 'shape_r '
if shape_fitness:
    config += 'shape_f '
if reset_pop:
    config += 'reset-{} '.format(reset_pop)
if base_mutate:
    config += 'mute-{} '.format(base_mutate)
if multiple_mutates:
    config += 'multate '
if noise_rate:
    config += 'n r-w-{}-{} '.format(noise_rate, noise_weight)
if constant_delays:
    config += 'const d-{} '.format(constant_delays)
else:
    config += 'max d-{} '.format(max_delay)
if fast_membrane:
    config += 'fast_mem '
if develop_neurons:
    config += 'dev_n '
if stdev_neurons:
    config += 'stdev_n '
if not force_i2o:
    config += 'ni2o '
if all_io:
    config += 'allio '
if exec_thing == 'pen' or exec_thing == 'xor' or exec_thing == 'logic':
    config += 'mc{} '.format(max_current)
if exec_thing == 'logic':
    config += '{} '.format(truth_table)
if free_label:
    config += '{} '.format(free_label)