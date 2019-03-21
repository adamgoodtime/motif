from methods.networks import motif_population
from methods.agents import agent_population
from methods.neurons import neuron_population
import numpy as np
import itertools
import traceback
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing
import pathos.multiprocessing

connections = []
weight_max = 0.1

agent_pop_size = 100
reward_shape = False
averaging_weights = True
reward = 1
noise_rate = 0
noise_weight = 0.01
fast_membrane = False

threading_tests = True
split = 1
new_split = 4  # agent_pop_size

#motif params
maximum_depth = [5, 15]
no_bins = [10, 375]
reset_pop = 0
size_f = False
spike_f = False#'out'
make_action = True
shape_fitness = True
random_arms = 0
viable_parents = 0.2
elitism = 0.2
exposure_time = 200
io_prob = 0.75  # 1.0 - (1.0 / 11.0)
read_pop = 0
# read_pop = 'Dirty place/good pendulum with plastic and high bins.csv'
keep_reading = 5
constant_delays = 0
base_mutate = 0
multiple_mutates = True
exec_thing = 'mnist'
plasticity = True
structural = False
develop_neurons = True
stdev_neurons = True
free_label = 0

#arms params
arms_runtime = 41000
arm1 = 0.8
arm2 = 0.1
arm3 = 0.1
arm_len = 1
arms = []
for i in range(arm_len):
    # arms.append([arm1, arm2])
    # arms.append([arm2, arm1])
    for arm in list(itertools.permutations([arm1, arm2, arm3])):
        arms.append(list(arm))
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1], [0, 1], [1, 0]]
'''top_prob = 1
low_prob = 0
med_prob = 0.1
hii_prob = 0.2
arms = [[low_prob, med_prob, top_prob, hii_prob, med_prob, low_prob, med_prob, low_prob], [top_prob, low_prob, low_prob, med_prob, hii_prob, med_prob, low_prob, med_prob],
        [hii_prob, top_prob, med_prob, low_prob, low_prob, med_prob, med_prob, low_prob], [med_prob, low_prob, low_prob, top_prob, med_prob, hii_prob, low_prob, med_prob],
        [low_prob, low_prob, low_prob, med_prob, top_prob, med_prob, hii_prob, med_prob], [low_prob, med_prob, low_prob, med_prob, med_prob, top_prob, low_prob, hii_prob],
        [med_prob, low_prob, hii_prob, low_prob, med_prob, low_prob, top_prob, med_prob], [low_prob, hii_prob, med_prob, med_prob, low_prob, med_prob, low_prob, top_prob]]
# '''

#pendulum params
pendulum_runtime = 181000
double_pen_runtime = 60000
max_fail_score = 0
no_v = False
encoding = 0
time_increment = 20
pole_length = 1
pole2_length = 0.1
pole_angle = [[0.1], [0.2], [-0.1], [-0.2]]
reward_based = 1
force_increments = 20
max_firing_rate = 1000
number_of_bins = 6
central = 1
bin_overlap = 2
tau_force = 0

#logic params
logic_runtime = 5000
score_delay = 200
stochastic = 1
truth_table = [0, 1, 1, 0]
# truth_table = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
input_sequence = []
segment = [0 for j in range(int(np.log2(len(truth_table))))]
input_sequence.append(segment)
for i in range(1, len(truth_table)):
    current_value = i
    segment = [0 for j in range(int(np.log2(len(truth_table))))]
    while current_value != 0:
        highest_power = int(np.log2(current_value))
        segment[highest_power] = 1
        current_value -= 2**highest_power
    input_sequence.append(segment)

#MNIST
max_freq = 5000
on_duration = 1000
off_duration = 1000
data_size = 100
mnist_runtime = data_size * (on_duration + off_duration)

#erbp params
erbp_runtime = 20
erbp_max_depth = [5, 100]

#breakout params
breakout_runtime = 181000
x_factor = 8
y_factor = 8
bricking = 0

inputs = 0
outputs = 0
test_data_set = []
config = ''
runtime = 0

def bandit(generations):
    print "starting"
    global connections, arms, max_fail_score, pole_angle, inputs, outputs, config, test_data_set, encoding, runtime, maximum_depth, make_action

    if exec_thing == 'br':
        runtime = arms_runtime
        inputs = (160 / x_factor) * (128 / y_factor)
        outputs = 2
        config = 'bout {}-{}-{} '.format(x_factor, y_factor, bricking)
        test_data_set = 'something'
        number_of_tests = 'something'
    elif exec_thing == 'xor':
        arms = [[0, 0], [0, 1], [1, 0], [1, 1]]
        config = 'xor '
        inputs = 2
        if reward == 1:
            outputs = 2
        else:
            outputs = 1
        max_fail_score = -1
        test_data_set = arms
        number_of_tests = len(arms)
    elif exec_thing == 'pen':
        runtime = pendulum_runtime
        encoding = 1
        inputs = 4
        if encoding != 0:
            inputs *= number_of_bins
        if no_v:
            inputs /= 2
        outputs = 2
        config = 'pend-an{}-{}-F{}-R{}-B{}-O{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap)
        test_data_set = pole_angle
        number_of_tests = len(pole_angle)
    elif exec_thing == 'rank pen':
        runtime = pendulum_runtime
        inputs = 4 * number_of_bins
        if no_v:
            inputs /= 2
        outputs = force_increments
        config = 'rank-pend-an{}-{}-F{}-R{}-B{}-O{}-E{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap, encoding)
        test_data_set = pole_angle
        number_of_tests = len(pole_angle)
    elif exec_thing == 'double pen':
        runtime = double_pen_runtime
        inputs = 6 * number_of_bins
        if no_v:
            inputs /= 2
        outputs = force_increments
        config = 'double-pend-an{}-{}-pl{}-{}-F{}-R{}-B{}-O{} '.format(pole_angle[0], len(pole_angle), pole_length, pole2_length, force_increments, max_firing_rate, number_of_bins, bin_overlap)
        test_data_set = pole_angle
        number_of_tests = len(pole_angle)
    elif exec_thing == 'arms':
        if isinstance(arms[0], list):
            number_of_arms = len(arms[0])
        else:
            number_of_arms = len(arms)
        runtime = arms_runtime
        test_data_set = arms
        inputs = 2
        outputs = number_of_arms
        config = 'bandit-{}-{}-{} '.format(arms[0][0], len(arms), random_arms)
        number_of_tests = len(arms)
    elif exec_thing == 'logic':
        runtime = logic_runtime
        test_data_set = input_sequence
        number_of_tests = len(input_sequence)
        inputs = len(input_sequence[0])
        outputs = 2
        if stochastic:
            config = 'logic-stoc-{}-run{}-sample{} '.format(truth_table, runtime, score_delay)
        else:
            config = 'logic-{}-run{}-sample{} '.format(truth_table, runtime, score_delay)
    elif exec_thing == 'mnist':
        runtime = mnist_runtime
        test_data_set = input_sequence
        number_of_tests = len(input_sequence)
        inputs = 28*28
        outputs = 10
        config = 'mnist-freq-{}-on-{}-off-{}-size-{} '.format(max_freq, on_duration, off_duration, data_size)
    elif exec_thing == 'erbp':
        maximum_depth = erbp_max_depth
        make_action = False
        runtime = erbp_runtime
        inputs = 0
        outputs = 0
        test_data_set = [[0], [1]]
        number_of_tests = len(test_data_set)
        config = 'erbp {} {} '.format(runtime, maximum_depth)
    else:
        print "\nNot a correct test setting\n"
        raise Exception
    if plasticity:
        config += 'pl '
    if averaging_weights:
        config += 'ave '
    if make_action:
        config += 'action '
    if spike_f:
        if spike_f == 'out':
            config += 'out-spikes '
        else:
            config += 'spikes '
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
        config += 'delay-{} '.format(constant_delays)
    if fast_membrane:
        config += 'fast_mem '
    if no_v:
        config += 'no_v '
    if develop_neurons:
        config += 'dev_n '
    if stdev_neurons:
        config += 'stdev_n '
    if free_label:
        config += '{} '.format(free_label)

    if stdev_neurons:
        neurons = neuron_population(inputs=inputs,
                                    outputs=outputs,
                                    pop_size=inputs+outputs+200+agent_pop_size*3,
                                    io_prob=io_prob,
                                    v_rest=-65.0,  # Resting membrane potential in mV.
                                    v_rest_stdev=5,
                                    cm=1.0,  # Capacity of the membrane in nF
                                    cm_stdev=0.3,
                                    tau_m=20.0,  # Membrane time constant in ms.
                                    tau_m_stdev=5,
                                    tau_refrac=0.1,  # Duration of refractory period in ms.
                                    tau_refrac_stdev=0.03,
                                    tau_syn_E=5,  # Rise time of the excitatory synaptic alpha function in ms.
                                    tau_syn_E_stdev=1.6,
                                    tau_syn_I=5,  # Rise time of the inhibitory synaptic alpha function in ms.
                                    tau_syn_I_stdev=1.6,
                                    e_rev_E=0.0,  # Reversal potential for excitatory input in mV
                                    e_rev_E_stdev=0,
                                    e_rev_I=-70.0,  # Reversal potential for inhibitory input in mV
                                    e_rev_I_stdev=3,
                                    v_thresh=-50.0,  # Spike threshold in mV.
                                    v_thresh_stdev=5,
                                    v_reset=-65.0,  # Reset potential after a spike in mV.
                                    v_reset_stdev=5,
                                    i_offset=3.0,  # Offset current in nA
                                    i_offset_stdev=1,
                                    v=-65.0,  # 'v_starting'
                                    v_stdev=5)
    else:
        neurons = neuron_population(inputs=inputs,
                                    outputs=outputs,
                                    io_prob=io_prob)

    motifs = motif_population(neurons,
                              max_motif_size=4,#maximum_depth[0],
                              no_weight_bins=no_bins,
                              no_delay_bins=no_bins,
                              weight_range=(0.005, weight_max),
                              constant_delays=constant_delays,
                              # delay_range=(10., 10.0000001),
                              neuron_types=(['excitatory', 'inhibitory']),
                              global_io=('highest', 'seeded', 'in'),
                              read_entire_population=read_pop,
                              keep_reading=keep_reading,
                              plasticity=plasticity,
                              structural=structural,
                              population_size=agent_pop_size*3+inputs+outputs)

    # todo :add number of different motifs to the fitness function to promote regularity

    agents = agent_population(motifs,
                              pop_size=agent_pop_size,
                              inputs=inputs,
                              outputs=outputs,
                              elitism=elitism,
                              sexuality=[7./20., 8./20., 3./20., 2./20.],
                              # sexuality=[7./20., 9./20., 4./20., 0],
                              base_mutate=base_mutate,
                              multiple_mutates=multiple_mutates,
                              # input_shift=0,
                              # output_shift=0,
                              maximum_depth=maximum_depth,
                              viable_parents=viable_parents)

    config += "reward-{}, max_d-{}, w_max-{}, rents-{}, elite-{}, psize-{}, bins-{}".format(
        reward, maximum_depth, weight_max, viable_parents, elitism, agent_pop_size,
        no_bins)

    if io_prob:
        config += ", io-{}".format(io_prob)
    else:
        config += " {}".format(motifs.global_io[1])
    if read_pop:
        config += ' read-{}'.format(keep_reading)

    for i in range(generations):

        print config

        if random_arms:
            arms = []
            for k in range(random_arms):
                total = 1
                arm = []
                for j in range(number_of_arms - 1):
                    arm.append(np.random.uniform(0, total))
                    total -= arm[j]
                arm.append(total)
                arms.append(arm)

        if i == 0:
            connections = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0])
        elif reset_pop:
            if i % reset_pop:
                connections = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0], create='reset')
        else:
            connections = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0], create=False)

        # config = 'test'
        if config != 'test':
            # arms = [0.1, 0.9, 0.2]
            # agents.bandit_test(connections, arms)
            if exec_thing == 'xor':
                execfile("../methods/exec_xor.py", globals())
            else:
                globals()['exec_thing'] = exec_thing
                execfile("../methods/exec_subprocess.py", globals())

        print "returned"

        fitnesses = agents.read_fitnesses(config, max_fail_score, make_action)

        print "1", motifs.total_weight

        agent_spikes = []
        for k in range(agent_pop_size):
            spike_total = 0
            for j in range(number_of_tests):
                if isinstance(fitnesses[j][k], list):
                    spike_total -= fitnesses[j][k][1] + fitnesses[j][k][2]
                    fitnesses[j][k] = fitnesses[j][k][0]
                else:
                    spike_total -= 1000000
            agent_spikes.append(spike_total)
        if spike_f:
            fitnesses.append(agent_spikes)

        agents.pass_fitnesses(fitnesses, max_fail_score, fitness_shaping=shape_fitness)

        agents.status_update(fitnesses, i, config, number_of_tests)

        print "\nconfig: ", config, "\n"

        print "2", motifs.total_weight

        motifs.adjust_weights(agents.agent_pop, reward_shape=reward_shape, iteration=i, average=averaging_weights, develop_neurons=develop_neurons)

        print "3", motifs.total_weight

        if config != 'test':
            motifs.save_motifs(i, config)
            agents.save_agents(i, config)
            if stdev_neurons:
                neurons.save_neurons(i, config)

        print "4", motifs.total_weight

        motifs.set_delay_bins(no_bins, i, generations)
        motifs.set_weight_bins(no_bins, i, generations)
        agents.set_max_d(maximum_depth, i, generations)

        agents.evolve(species=False)

        print "finished", i, motifs.total_weight


# adjust population weights and clean up unused motifs

# generate offspring
    # mutate the individual and translate it to a new motif
    # connection change
    # swap motif

bandit(1000)

print "done"

# if __name__ == '__main__':
#     main()

#ToDo
'''
anneal the amount of IO weight (unnecessary when added population of neurons)
add species to agents
keep a log of seperate motif components to affect the chance of certain nodes and connects being chosen randomly through mutation
    a separate population of weights and nodes (neurons and synapes)
variability in the plasticity rule
crossover not just random   
    select motifs based on IO ratio
    select based on weight
figure out the disparity between expected possible combinations and actual
'''