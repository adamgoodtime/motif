from methods.networks import motif_population
from methods.agents import agent_population
from methods.neurons import neuron_population
import numpy as np
import sys
import itertools
import traceback
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing
import pathos.multiprocessing

connections = []
weight_max = 0.1

agent_pop_size = 100
generations = 200
reward_shape = False
averaging_weights = True
noise_rate = 0
noise_weight = 0.01
fast_membrane = False

threading_tests = True
split = 1
new_split = 4  # agent_pop_size

#motif params
maximum_depth = [3, 10]
no_bins = [10, 375]
reset_pop = 0
size_f = False
spike_f = False#'out'
repeat_best_amount = 5
# depth fitness
make_action = True
shape_fitness = True
viable_parents = 0.2
elitism = 0.2
exposure_time = 200
io_prob = 0.95  # 1.0 - (1.0 / 11.0)
read_motifs = 0
# read_motifs = 'Dirty place/Motif pop xor pl 200 stdev_n.npy'
# read_motifs = 'Dirty place/Motif pop xor pl 5000 stdev_n.npy'
read_neurons = 0
# read_neurons = 'Dirty place/Neuron pop xor pl 200 stdev_n.npy'
# read_neurons = 'Dirty place/Neuron pop xor pl 5000 stdev_n.npy'
keep_reading = 5
constant_delays = 0
max_delay = 25.0
base_mutate = 0
multiple_mutates = True
exec_thing = 'arms'
plasticity = True
structural = False
develop_neurons = True
stdev_neurons = True
neuron_type = 'IF_cond_exp'
max_input_current = 0.8
calcium_tau = 50
free_label = '{}'.format(sys.argv[1])
parallel = False

# '''
constant_delays = float(sys.argv[2])
plasticity = bool(sys.argv[3])
develop_neurons = bool(sys.argv[4])
stdev_neurons = bool(sys.argv[5])
# '''

#arms params
arms_runtime = 20000
constant_input = 1
arms_stochastic = 1
arms_rate_on = 20
arms_rate_off = 5
random_arms = 0
arm1 = 0.8
arm2 = 0.1
arm3 = 0.1
arm_len = 1
arms = []
arms_reward = 1
for i in range(arm_len):
    arms.append([arm1, arm2])
    arms.append([arm2, arm1])
    # for arm in list(itertools.permutations([arm1, arm2, arm3])):
    #     arms.append(list(arm))
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1], [0, 1], [1, 0]]
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
pendulum_delays = 1
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
score_delay = 5000
logic_stochastic = 1
logic_rate_on = 20
logic_rate_off = 5
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

#Recall params
recall_runtime = 60000
recall_rate_on = 50
recall_rate_off = 0
recall_pop_size = 1
prob_command = 1./6.
prob_in_change = 1./2.
time_period = 200
recall_stochastic = 1
recall_reward = 0
recall_parallel_runs = 2

#MNIST
max_freq = 5000
on_duration = 1000
off_duration = 1000
data_size = 200
mnist_parallel_runs = 2
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
stochastic = -1
rate_on = 0
rate_off = 0

def bandit(generations):
    print "starting"
    global connections, arms, max_fail_score, pole_angle, inputs, outputs, config, test_data_set, encoding, runtime, maximum_depth, make_action, constant_delays, weight_max, max_input_current, stochastic, rate_on, rate_off

    if exec_thing == 'br':
        runtime = breakout_runtime
        inputs = (160 / x_factor) * (128 / y_factor)
        outputs = 2
        config = 'bout {}-{}-{} '.format(x_factor, y_factor, bricking)
        test_data_set = 'something'
        number_of_tests = 'something'
    elif exec_thing == 'pen':
        runtime = pendulum_runtime
        # constant_delays = pendulum_delays
        encoding = 1
        inputs = 4
        if encoding != 0:
            inputs *= number_of_bins
        outputs = 2
        config = 'pend-an{}-{}-F{}-R{}-B{}-O{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap)
        if no_v:
            inputs /= 2
            config += "\b-no_v "
        test_data_set = pole_angle
        number_of_tests = len(pole_angle)
    elif exec_thing == 'rank pen':
        runtime = pendulum_runtime
        constant_delays = pendulum_delays
        inputs = 4 * number_of_bins
        outputs = force_increments
        config = 'rank-pend-an{}-{}-F{}-R{}-B{}-O{}-E{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap, encoding)
        if no_v:
            config += "\b-no_v "
            inputs /= 2
        test_data_set = pole_angle
        number_of_tests = len(pole_angle)
    elif exec_thing == 'double pen':
        runtime = double_pen_runtime
        constant_delays = pendulum_delays
        inputs = 6 * number_of_bins
        if no_v:
            inputs /= 2
        outputs = force_increments
        config = 'double-pend-an{}-{}-pl{}-{}-F{}-R{}-B{}-O{} '.format(pole_angle[0], len(pole_angle), pole_length, pole2_length, force_increments, max_firing_rate, number_of_bins, bin_overlap)
        test_data_set = pole_angle
        number_of_tests = len(pole_angle)
    elif exec_thing == 'arms':
        stochastic = arms_stochastic
        rate_on = arms_rate_on
        rate_off = arms_rate_off
        if isinstance(arms[0], list):
            number_of_arms = len(arms[0])
        else:
            number_of_arms = len(arms)
        runtime = arms_runtime
        test_data_set = arms
        inputs = 2
        outputs = number_of_arms
        if random_arms:
            config = 'bandit-rand-{}-{} '.format(arms[0][0], len(arms))
        else:
            config = 'bandit-{}-{} '.format(arms[0][0], len(arms))
        if constant_input:
            if stochastic:
                config += 'stoc '
            config += 'on-{} off-{} r{} '.format(rate_on, rate_off, arms_reward)
        number_of_tests = len(arms)
    elif exec_thing == 'logic':
        stochastic = logic_stochastic
        runtime = logic_runtime
        rate_on = logic_rate_on
        rate_off = logic_rate_off
        test_data_set = input_sequence
        number_of_tests = len(input_sequence)
        inputs = len(input_sequence[0])
        outputs = 2
        if stochastic:
            config = 'logic-stoc-{}-run{}-sample{} '.format(truth_table, runtime, score_delay)
        else:
            config = 'logic-{}-run{}-sample{} '.format(truth_table, runtime, score_delay)
        config += 'on-{} off-{} '.format(rate_on, rate_off)
    elif exec_thing == 'recall':
        stochastic = recall_stochastic
        runtime = recall_runtime
        rate_on = recall_rate_on
        rate_off = recall_rate_off
        for j in range(recall_parallel_runs):
            test_data_set.append([j])
        number_of_tests = recall_parallel_runs
        inputs = 4 * recall_pop_size
        outputs = 2
        if stochastic:
            config = 'recall-stoc-pop_s{}-run{}-in_p{}-r_on{} '.format(recall_pop_size, runtime, prob_in_change, rate_on)
        else:
            config = 'recall-pop_s{}-run{}-in_p{}-r_on{} '.format(recall_pop_size, runtime, prob_in_change, rate_on)
    elif exec_thing == 'mnist':
        runtime = mnist_runtime
        for j in range(mnist_parallel_runs):
            test_data_set.append([j])
        number_of_tests = mnist_parallel_runs
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
        if plasticity == 'all':
            config += 'pall '
        else:
            config += 'pl '
    if structural:
        config += 'strc '
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
        config += 'const d-{} '.format(constant_delays)
    else:
        config += 'max d-{} '.format(max_delay)
    if fast_membrane:
        config += 'fast_mem '
    if develop_neurons:
        config += 'dev_n '
    if stdev_neurons:
        config += 'stdev_n '
        config += 'inc-{} '.format(max_input_current)
    if free_label:
        config += '{} '.format(free_label)

    neurons = neuron_population(inputs=inputs,
                                outputs=outputs,
                                pop_size=inputs+outputs+200+agent_pop_size*3,
                                io_prob=io_prob,
                                max_input_current=max_input_current,
                                read_population=read_neurons,
                                neuron_type=neuron_type,
                                default=not stdev_neurons)

    if neuron_type == 'IF_cond_exp':
        weight_max = 0.1
        config += 'cond '
    elif neuron_type == 'IF_curr_exp':
        weight_max = 4.8
        config += 'cure '
    elif neuron_type == 'IF_curr_alpha':
        weight_max = 23.5
        config += 'cura '
    elif neuron_type == 'calcium':
        weight_max = 4.8
        config += 'calc-{} '.format(calcium_tau)
    else:
        print "incorrect neuron type"
        raise Exception
    motifs = motif_population(neurons,
                              max_motif_size=4,#maximum_depth[0],
                              no_weight_bins=no_bins,
                              no_delay_bins=no_bins,
                              weight_range=(0.005, weight_max),
                              constant_delays=constant_delays,
                              delay_range=(1., max_delay),
                              neuron_types=(['excitatory', 'inhibitory']),
                              global_io=('highest', 'seeded', 'in'),
                              read_entire_population=read_motifs,
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

    # config += "max_d-{}, rents-{}, elite-{}, psize-{}, bins-{}".format(
    #     maximum_depth, viable_parents, elitism, agent_pop_size, no_bins)

    config += "max_d-{}, bins-{}".format(
        maximum_depth, no_bins)

    if io_prob:
        config += ", io-{}".format(io_prob)
    else:
        config += " {}".format(motifs.global_io[1])
    if read_motifs and read_neurons:
        config += ' readmn-{}'.format(keep_reading)
    elif read_motifs:
        config += ' readm-{}'.format(keep_reading)
    elif read_neurons:
        config += ' readn-{}'.format(keep_reading)

    print config

    best_performance_score = []
    best_performance_fitness = []

    for gen in range(generations):

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

        if gen == 0:
            connections = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0])
        elif reset_pop:
            if gen % reset_pop:
                connections = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0], create='reset')
        else:
            connections = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0], create=False)

        if gen != 0:
            for i in range(repeat_best_amount):
                connections.append(best_score_connections)
            for i in range(repeat_best_amount):
                connections.append(best_fitness_connections)

        # config = 'test'
        if config != 'test':
            globals()['exec_thing'] = exec_thing
            execfile("../methods/exec_subprocess.py", globals())

        print "returned"

        fitnesses = agents.read_fitnesses(config, max_fail_score, make_action)

        print "1", motifs.total_weight

        agent_spikes = []
        for k in range(len(connections)):
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

        if gen != 0:
            best_agent_repeat_score = []
            best_agent_repeat_fitness = []
            for i in range(len(fitnesses)):
                print fitnesses[i][agent_pop_size:agent_pop_size+repeat_best_amount]
                best_agent_repeat_score.append(fitnesses[i][agent_pop_size:agent_pop_size+repeat_best_amount])
                best_agent_repeat_fitness.append(fitnesses[i][agent_pop_size+repeat_best_amount:agent_pop_size+(2*repeat_best_amount)])
                fitnesses[i] = fitnesses[i][0:agent_pop_size]
            test_scores = [0 for i in range(repeat_best_amount)]
            test_fitness = [0 for i in range(repeat_best_amount)]
            for i in range(number_of_tests):
                for j in range(repeat_best_amount):
                    test_scores[j] += best_agent_repeat_score[i][j]
                    test_fitness[j] += best_agent_repeat_fitness[i][j]
            best_performance_score.append(np.average(test_scores))
            best_performance_fitness.append(np.average(test_fitness))

        agents.pass_fitnesses(fitnesses, max_fail_score, fitness_shaping=shape_fitness)

        [best_score_connections, best_fitness_connections] = agents.status_update(fitnesses, gen, config, number_of_tests, connections, best_performance_score, best_performance_fitness)

        if gen != 0:
            print "The performance of the best agent from last generation was:"
            print "Score:"
            for i in range(number_of_tests):
                print best_agent_repeat_score[i]
            print "Fitness:"
            for i in range(number_of_tests):
                print best_agent_repeat_fitness[i]

        print "\nconfig: ", config, "\n"

        print "2", motifs.total_weight

        if gen % 10 == 0:
            print "stop"

        motifs.adjust_weights(agents.agent_pop, reward_shape=reward_shape, iteration=gen, average=averaging_weights, develop_neurons=develop_neurons)

        print "3", motifs.total_weight

        if config != 'test':
            motifs.save_motifs(gen, config)
            agents.save_agents(gen, config)
            if stdev_neurons:
                neurons.save_neurons(gen, config)

        print "4", motifs.total_weight

        motifs.set_delay_bins(no_bins, gen, generations)
        motifs.set_weight_bins(no_bins, gen, generations)
        agents.set_max_d(maximum_depth, gen, generations)

        agents.evolve(species=False)

        print "finished", gen, motifs.total_weight


# adjust population weights and clean up unused motifs

# generate offspring
    # mutate the individual and translate it to a new motif
    # connection change
    # swap motif

bandit(generations)

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