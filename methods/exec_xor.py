import spynnaker8 as p
# from spynnaker.pyNN.connections. \
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
# from spinn_front_end_common.utilities.globals_variables import get_simulator
#
# import pylab
# from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
#     SpynnakerExternalDevicePluginManager as ex
import sys, os
import time
import socket
import numpy as np
from spinn_bandit.python_models.bandit import Bandit
from spinn_arm.python_models.arm import Arm
import math
import itertools
from copy import deepcopy
import operator
from spinn_front_end_common.utilities.globals_variables import get_simulator
import traceback
import math
from methods.networks import motif_population
import traceback
import csv
import threading
import pathos.multiprocessing
from spinn_front_end_common.utilities import globals_variables

max_fail_score = 0  # -int(runtime / exposure_time)
setup_retry_time = 60
new_split = 100

# stdp_model = p.STDPMechanism(
#     timing_dependence=p.SpikePairRule(tau_plus=20., tau_minus=20.0, A_plus=0.003, A_minus=0.003),
#     weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=0.1))


def split_ex_in(connections):
    excite = []
    inhib = []
    for conn in connections:
        if conn[2] > 0:
            excite.append(conn)
        else:
            inhib.append(conn)
    for conn in inhib:
        conn[2] *= -1
    return excite, inhib

def split_plastic(connections):
    plastic = []
    non_plastic = []
    for conn in connections:
        if conn[4] == 'plastic':
            plastic.append([conn[0], conn[1], conn[2], conn[3]])
        else:
            non_plastic.append([conn[0], conn[1], conn[2], conn[3]])
    return plastic, non_plastic

def get_scores(game_pop, simulator):
    g_vertex = game_pop._vertex
    scores = g_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
    return scores.tolist()

def thread_xor(connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                  reward=0, size_f=False, spike_f=False, top=True):
    def helper(args):
        return xor_test(*args)

    step_size = len(connections) / split
    if step_size == 0:
        step_size = 1
    if isinstance(arms[0], list):
        connection_threads = []
        all_configs = [[[connections[x:x + step_size], arm, split, runtime, exposure_time, noise_rate, noise_weight,
                         reward, spike_f, np.random.randint(1000000000)] for x in range(0, len(connections), step_size)] for arm in arms]
        for arm in all_configs:
            for config in arm:
                connection_threads.append(config)
    else:
        connection_threads = [[connections[x:x + step_size], arms, split, runtime, exposure_time, noise_rate,
                               noise_weight, reward, spike_f, np.random.randint(1000000000)] for x in range(0, len(connections), step_size)]

    pool = pathos.multiprocessing.Pool(processes=len(connection_threads))

    pool_result = pool.map(func=helper, iterable=connection_threads)

    pool.close()

    for i in range(len(pool_result)):
        if pool_result[i] == 'fail' and len(connection_threads[i][0]) > 1:
            print("splitting ", len(connection_threads[i][0]), " into ", new_split, " pieces")
            problem_arms = connection_threads[i][1]
            pool_result[i] = thread_xor(connection_threads[i][0], problem_arms, new_split, runtime,
                                                exposure_time, noise_rate, noise_weight, reward, spike_f, top=False)

    agent_fitness = []
    for thread in pool_result:
        if isinstance(thread, list):
            for result in thread:
                agent_fitness.append(result)
        else:
            agent_fitness.append(thread)

    if isinstance(arms[0], list) and top:
        copy_fitness = deepcopy(agent_fitness)
        agent_fitness = []
        for i in range(len(arms)):
            arm_results = []
            for j in range(pop_size):
                arm_results.append(copy_fitness[(i * pop_size) + j])
            agent_fitness.append(arm_results)
        if size_f:
            arm_results = []
            for i in range(pop_size):
                arm_results.append(connections[i][2] + connections[i][5])
            agent_fitness.append(arm_results)
    return agent_fitness

def connect_to_arms(pre_pop, from_list, arms, r_type, plastic, stdp_model):
    arm_conn_list = []
    for i in range(len(arms)):
        arm_conn_list.append([])
    for conn in from_list:
        arm_conn_list[conn[1]].append((conn[0], 0, conn[2], conn[3]))
        # print "out:", conn[1]
        # if conn[1] == 2:
        #     print '\nit is possible\n'
    for i in range(len(arms)):
        if len(arm_conn_list[i]) != 0:
            if plastic:
                p.Projection(pre_pop, arms[i], p.FromListConnector(arm_conn_list[i]),
                             receptor_type=r_type, synapse_type=stdp_model)
            else:
                p.Projection(pre_pop, arms[i], p.FromListConnector(arm_conn_list[i]),
                             receptor_type=r_type)

def connect_inputs(left, right, from_list, post_pop, r_type, plastic, stdp_model):
    input_conn_list = []
    for i in range(2):
        input_conn_list.append([])
    for conn in from_list:
        input_conn_list[conn[0]].append((0, conn[1], conn[2], conn[3]))
        # print "out:", conn[1]
        # if conn[1] == 2:
        #     print '\nit is possible\n'
    for i in range(2):
        if len(input_conn_list[i]) != 0:
            if i == 0:
                if plastic:
                    p.Projection(left, post_pop, p.FromListConnector(input_conn_list[i]),
                                 receptor_type=r_type, synapse_type=stdp_model)
                else:
                    p.Projection(left, post_pop, p.FromListConnector(input_conn_list[i]),
                                 receptor_type=r_type)
            else:
                if plastic:
                    p.Projection(right, post_pop, p.FromListConnector(input_conn_list[i]),
                                 receptor_type=r_type, synapse_type=stdp_model)
                else:
                    p.Projection(right, post_pop, p.FromListConnector(input_conn_list[i]),
                                 receptor_type=r_type)

def xor_test(connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                reward=0, spike_f=False, seed=0):
    np.random.seed(seed)
    seed = [seed, arms[0], arms[1]]
    sleep = 10 * np.random.random()
    # time.sleep(sleep)
    on_rate = 15
    off_rate = 2
    max_attempts = 2
    try_except = 0
    while try_except < max_attempts:
        left_input = []
        right_input = []
        input_count = -1
        output_pop = []
        excite = []
        excite_count = -1
        excite_marker = []
        inhib = []
        inhib_count = -1
        inhib_marker = []
        failures = []
        start = time.time()
        try_count = 0
        while time.time() - start < setup_retry_time:
            try:
                p.setup(timestep=1.0, min_delay=1, max_delay=127)
                p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
                print("\nfinished setup seed = ", seed, "\n")
                break
            except:
                traceback.print_exc()
                sleep = 1 * np.random.random()
                time.sleep(sleep)
            print("\nsetup", try_count, " seed = ", seed, "\n", "\n")
            try_count += 1
        print("\nfinished setup seed = ", seed, "\n")
        print(config)
        for i in range(len(connections)):
            [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
             i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out] = connections[i]
            if len(in2e) == 0 and len(in2i) == 0 and len(in2out) == 0:
                failures.append(i)
                print("agent {} was not properly connected to the game".format(i))
            else:
                input_count += 1
                if arms[0] == 1:
                    left_input.append(p.Population(1, p.SpikeSourcePoisson(rate=on_rate), label='left on'))
                else:
                    left_input.append(p.Population(1, p.SpikeSourcePoisson(rate=off_rate), label='left off'))
                if arms[1] == 1:
                    right_input.append(p.Population(1, p.SpikeSourcePoisson(rate=on_rate), label='right on'))
                else:
                    right_input.append(p.Population(1, p.SpikeSourcePoisson(rate=off_rate), label='right off'))
                # choice = []
                # choice.append(p.Population(1, p.IF_cond_exp(), label='0_pop{}-{}'.format(input_count, i)))
                # choice.append(p.Population(1, p.IF_cond_exp(), label='0_pop{}-{}'.format(input_count, i)))
                output_pop.append(p.Population(2, p.IF_cond_exp(), label='output_pop{}-{}'.format(input_count, i)))
                output_pop[input_count].record('spikes')
                if e_size > 0:
                    excite_count += 1
                    excite.append(
                        p.Population(e_size, p.IF_cond_exp(), label='excite_pop_{}-{}'.format(excite_count, i)))
                    if noise_rate:
                        excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(excite_noise, excite[excite_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if spike_f:
                        excite[excite_count].record('spikes')
                    excite_marker.append(i)
                if i_size > 0:
                    inhib_count += 1
                    inhib.append(p.Population(i_size, p.IF_cond_exp(), label='inhib_pop_{}-{}'.format(inhib_count, i)))
                    if noise_rate:
                        inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(inhib_noise, inhib[inhib_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if spike_f:
                        inhib[inhib_count].record('spikes')
                    inhib_marker.append(i)
                stdp_model = p.STDPMechanism(
                    timing_dependence=p.SpikePairRule(
                        tau_plus=20., tau_minus=20.0, A_plus=0.02, A_minus=0.02),
                    weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=0.1))
                if len(in2e) != 0:
                    [in_ex, in_in] = split_ex_in(in2e)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], plastic,
                                           excite[excite_count], 'excitatory', True, stdp_model)
                        if len(non_plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], non_plastic,
                                           excite[excite_count], 'excitatory', False, stdp_model)
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], plastic,
                                           excite[excite_count], 'inhibitory', True, stdp_model)
                        if len(non_plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], non_plastic,
                                           excite[excite_count], 'inhibitory', False, stdp_model)
                if len(in2i) != 0:
                    [in_ex, in_in] = split_ex_in(in2i)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], plastic,
                                           inhib[inhib_count], 'excitatory', True, stdp_model)
                        if len(non_plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], non_plastic,
                                           inhib[inhib_count], 'excitatory', False, stdp_model)
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], plastic,
                                           inhib[inhib_count], 'inhibitory', True, stdp_model)
                        if len(non_plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], non_plastic,
                                           inhib[inhib_count], 'inhibitory', False, stdp_model)
                if len(in2out) != 0:
                    [in_ex, in_in] = split_ex_in(in2out)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], plastic,
                                           output_pop[input_count], 'excitatory', True, stdp_model)
                        if len(non_plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], non_plastic,
                                           output_pop[input_count], 'excitatory', False, stdp_model)
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], plastic,
                                           output_pop[input_count], 'inhibitory', True, stdp_model)
                        if len(non_plastic) != 0:
                            connect_inputs(left_input[input_count], right_input[input_count], non_plastic,
                                           output_pop[input_count], 'inhibitory', False, stdp_model)
                if len(e2e) != 0:
                    [plastic, non_plastic] = split_plastic(e2e)
                    if len(plastic) != 0:
                        p.Projection(excite[excite_count], excite[excite_count], p.FromListConnector(plastic),
                                     receptor_type='excitatory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], excite[excite_count], p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(e2i) != 0:
                    [plastic, non_plastic] = split_plastic(e2i)
                    if len(plastic) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count], p.FromListConnector(plastic),
                                     receptor_type='excitatory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(i2e) != 0:
                    [plastic, non_plastic] = split_plastic(i2e)
                    if len(plastic) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count], p.FromListConnector(plastic),
                                     receptor_type='inhibitory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count], p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(i2i) != 0:
                    [plastic, non_plastic] = split_plastic(i2i)
                    if len(plastic) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count], p.FromListConnector(plastic),
                                     receptor_type='inhibitory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(e2out) != 0:
                    [plastic, non_plastic] = split_plastic(e2out)
                    if len(plastic) != 0:
                        p.Projection(excite[excite_count], output_pop[input_count], p.FromListConnector(plastic),
                                     receptor_type='excitatory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], output_pop[input_count], p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(i2out) != 0:
                    [plastic, non_plastic] = split_plastic(i2out)
                    if len(plastic) != 0:
                        p.Projection(inhib[inhib_count], output_pop[input_count], p.FromListConnector(plastic),
                                     receptor_type='inhibitory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], output_pop[input_count], p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')

        print("\nfinished connections seed = ", seed, "\n")
        simulator = get_simulator()
        try:
            print("\nrun seed = ", seed, "\n")
            if len(connections) == len(failures):
                p.end()
                print("nothing to run so ending and returning fail")
                return 'fail'
            p.run(runtime)
            try_except = max_attempts
            break
        except:
            traceback.print_exc()
            try:
                print("\nrun 2 seed = ", seed, "\n")
                globals_variables.unset_simulator()
                print("end was necessary")
            except:
                traceback.print_exc()
                print("end wasn't necessary")
            try_except += 1
            print("failed to run on attempt ", try_except, "\n")  # . total fails: ", all_fails, "\n"
            if try_except >= max_attempts:
                print("calling it a failed population, splitting and rerunning")
                return 'fail'
        print("\nfinished run seed = ", seed, "\n")

    scores = []
    agent_fitness = []
    fails = 0
    excite_spike_count = [0 for i in range(len(connections))]
    excite_fail = 0
    inhib_spike_count = [0 for i in range(len(connections))]
    inhib_fail = 0
    print("reading the spikes of ", config, '\n', seed)
    for i in range(len(connections)):
        print("started processing fitness of: ", i, '/', len(connections))
        if i in failures:
            print("worst score for the failure")
            fails += 1
            scores.append([0, 100000])
            # agent_fitness.append(scores[i])
            excite_spike_count[i] -= max_fail_score
            inhib_spike_count[i] -= max_fail_score
        else:
            if spike_f:
                if i in excite_marker:
                    print("counting excite spikes")
                    spikes = excite[i - excite_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            excite_spike_count[i] += 1
                else:
                    excite_fail += 1
                    print("had an excite failure")
                if i in inhib_marker:
                    print("counting inhib spikes")
                    spikes = inhib[i - inhib_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            inhib_spike_count[i] += 1
                else:
                    inhib_fail += 1
                    print("had an inhib failure")
            spikes = output_pop[i - fails].get_data('spikes').segments[0].spiketrains
            on_spike = 0
            off_spike = 0
            j = 0
            for neuron in spikes:
                for spike in neuron:
                    if j == 1:
                        off_spike += 1
                    else:
                        on_spike += 1
                j += 1
            if (arms[0] == 1 and arms[1] == 1) or (arms[0] == 0 and arms[1] == 0):
                if reward == 1:
                    if off_spike > on_spike:
                        scores.append([1, on_spike + off_spike])
                    elif off_spike < on_spike:
                        scores.append([0, on_spike + off_spike])
                    else:
                        scores.append([-1, on_spike + off_spike])
                else:
                    if on_spike:
                        scores.append([-1, on_spike + off_spike])
                    else:
                        scores.append([1, on_spike + off_spike])
            else:
                if reward == 1:
                    if off_spike < on_spike:
                        scores.append([1, on_spike + off_spike])
                    elif off_spike > on_spike:
                        scores.append([0, on_spike + off_spike])
                    else:
                        scores.append([-1, on_spike + off_spike])
                else:
                    if on_spike:
                        scores.append([1, on_spike + off_spike])
                    else:
                        scores.append([-1, on_spike + off_spike])
            # pop[i].stats = {'fitness': scores[i][len(scores[i]) - 1][0]}  # , 'steps': 0}
        print("\nfinished spikes", seed)
        if spike_f:
            agent_fitness.append([scores[i][0], excite_spike_count[i] + inhib_spike_count[i] + scores[i][1]])
        else:
            agent_fitness.append(scores[i][0])
        # print i, "| e:", excite_spike_count[i], "-i:", inhib_spike_count[i], "|\t", scores[i]
    print(seed, "\nThe scores for this run of {} agents are:".format(len(connections)))
    for i in range(len(connections)):
        print("c:{}, s:{}".format(len(connections), len(scores)))
        e_string = "e: {}".format(excite_spike_count[i])
        i_string = "i: {}".format(inhib_spike_count[i])
        score_string = ""
        for j in range(len(scores[i])):
            score_string += "{:4}".format(scores[i][j])
        print("{:3} | {:8} {:8} - ".format(i, e_string, i_string), score_string)
    print("before end = ", seed)
    p.end()
    print("\nafter end = ", seed, "\n")
    print(config)
    return agent_fitness

def print_fitnesses(fitnesses):
    with open('fitnesses {}.csv'.format(config), 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        for fitness in fitnesses:
            writer.writerow(fitness)
        file.close()
    # with open('done {}.csv'.format(config), 'w') as file:
    #     writer = csv.writer(file, delimiter=',', lineterminator='\n')
    #     writer.writerow('', '')
    #     file.close()

# xor_test(connections, arms, split=4, runtime=runtime, exposure_time=exposure_time, noise_rate=noise_rate, noise_weight=0.01,
#                 reward=0, spike_f=False, seed=0)

fitnesses = thread_xor(connections, arms, split, runtime, exposure_time, noise_rate, noise_weight, reward, size_f, spike_f, True)

print_fitnesses(fitnesses)