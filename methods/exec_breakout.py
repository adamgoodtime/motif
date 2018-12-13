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
from spinn_breakout.python_models.breakout import breakout
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
from spinn_breakout import Breakout as b_out

max_fail_score = -int(runtime / exposure_time)
setup_retry_time = 60
new_split = 100

def row_col_to_input_breakout(row, col, is_on_input, row_bits, event_bits=1, colour_bits=2, row_start=0):
    row_bits = np.uint32(row_bits)
    idx = np.uint32(0)

    if is_on_input:
        idx = 1

    row += row_start
    idx = idx | (row << (colour_bits))  # colour bit
    idx = idx | (col << (row_bits + colour_bits))

    # add two to allow for special event bits
    idx = idx + 2

    return idx

def subsample_connection(x_res, y_res, subsamp_factor_x, subsamp_factor_y, weight,
                         coord_map_func):
    # subY_BITS=int(np.ceil(np.log2(y_res/subsamp_factor)))
    connection_list_on = []
    connection_list_off = []

    sx_res = int(x_res) // int(subsamp_factor_x)
    row_bits = int(np.ceil(np.log2(x_res)))
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i // subsamp_factor_x
            sj = j // subsamp_factor_y
            # ON channels
            subsampidx = sj * sx_res + si
            connection_list_on.append((coord_map_func(j, i, 1, row_bits),
                                       subsampidx, weight, 1.))
            # OFF channels only on segment borders
            # if((j+1)%(y_res/subsamp_factor)==0 or (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
            connection_list_off.append((coord_map_func(j, i, 0, row_bits),
                                        subsampidx, weight, 1.))

    return connection_list_on, connection_list_off

def parse_connections(connections, x_res, y_res, subsamp_factor_x, subsamp_factor_y, pre_or_post):
    connection_list_on = []
    connection_list_off = []
    for conn in connections:
        if pre_or_post == 'pre':
            id = conn[0]
        else:
            id = conn[1]
        # do something with the neuron id to make it the actual id on breakout

    return connection_list_on, connection_list_off

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

def get_scores(game_pop, simulator):
    g_vertex = game_pop._vertex
    scores = g_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
    return scores.tolist()

def thread_breakout(connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                  reward=0, size_f=False, spike_f=False, top=True):
    def helper(args):
        return breakout_test(*args)

    step_size = len(connections) / split
    if step_size == 0:
        step_size = 1
    if isinstance(arms[0], list):
        connection_threads = []
        all_configs = [[[connections[x:x + step_size], arm, split, runtime, exposure_time, noise_rate, noise_weight,
                         reward, spike_f, np.random.randint(1000000000)] for x in xrange(0, len(connections), step_size)] for arm in arms]
        for arm in all_configs:
            for config in arm:
                connection_threads.append(config)
    else:
        connection_threads = [[connections[x:x + step_size], arms, split, runtime, exposure_time, noise_rate,
                               noise_weight, reward, spike_f, np.random.randint(1000000000)] for x in xrange(0, len(connections), step_size)]

    pool = pathos.multiprocessing.Pool(processes=len(connection_threads))

    pool_result = pool.map(func=helper, iterable=connection_threads)

    pool.close()

    for i in range(len(pool_result)):
        if pool_result[i] == 'fail' and len(connection_threads[i][0]) > 1:
            print "splitting ", len(connection_threads[i][0]), " into ", new_split, " pieces"
            problem_arms = connection_threads[i][1]
            pool_result[i] = thread_breakout(connection_threads[i][0], problem_arms, new_split, runtime,
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


def breakout_test(connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                reward=0, spike_f=False, seed=0):
    np.random.seed(seed)
    sleep = 10 * np.random.random()
    # time.sleep(sleep)
    max_attempts = 2
    try_except = 0
    while try_except < max_attempts:
        breakout = []
        breakout_count = -1
        output = []
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
                print "\nfinished setup seed = ", seed, "\n"
                break
            except:
                traceback.print_exc()
                sleep = 1 * np.random.random()
                time.sleep(sleep)
            print "\nsetup", try_count, " seed = ", seed, "\n", "\n"
            try_count += 1
        print "\nfinished setup seed = ", seed, "\n"
        for i in range(len(connections)):
            [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
             i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out] = connections[i]
            if len(in2e) == 0 and len(in2i) == 0 and len(in2in) == 0 and len(in2out) == 0:
                failures.append(i)
                print "agent {} was not properly connected to the game".format(i)
            else:
                breakout_count += 1
                breakout.append(
                    p.Population(len(arms), b_out(x_factor=x_factor, y_factor=y_factor, bricking=bricking,
                                                   label='breakout_pop_{}-{}'.format(breakout_count, i))))
                output.append(
                    p.Population(len(arms), p.IF_cond_exp(), label='output_{}-{}'.format(breakout_count, i)))
                p.Projection(output[breakout_count], breakout[breakout_count], p.AllToAllConnector(), p.StaticSynapse())
                if e_size > 0:
                    excite_count += 1
                    excite.append(
                        p.Population(e_size, p.IF_cond_exp(), label='excite_pop_{}-{}'.format(excite_count, i)))
                    excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
                    p.Projection(excite_noise, excite[excite_count], p.OneToOneConnector(),
                                 p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if spike_f:
                        excite[excite_count].record('spikes')
                    excite_marker.append(i)
                if i_size > 0:
                    inhib_count += 1
                    inhib.append(p.Population(i_size, p.IF_cond_exp(), label='inhib_pop_{}-{}'.format(inhib_count, i)))
                    inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
                    p.Projection(inhib_noise, inhib[inhib_count], p.OneToOneConnector(),
                                 p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if spike_f:
                        inhib[inhib_count].record('spikes')
                    inhib_marker.append(i)
                if len(in2e) != 0:

                    [in_ex, in_in] = split_ex_in(in2e)
                    if len(in_ex) != 0:
                        p.Projection(breakout[breakout_count], excite[excite_count], p.FromListConnector(in_ex),
                                     receptor_type='excitatory')
                    if len(in_in) != 0:
                        p.Projection(breakout[breakout_count], excite[excite_count], p.FromListConnector(in_in),
                                     receptor_type='inhibitory')
                if len(in2i) != 0:
                    [in_ex, in_in] = split_ex_in(in2i)
                    if len(in_ex) != 0:
                        p.Projection(breakout[breakout_count], inhib[inhib_count], p.FromListConnector(in_ex),
                                     receptor_type='excitatory')
                    if len(in_in) != 0:
                        p.Projection(breakout[breakout_count], inhib[inhib_count], p.FromListConnector(in_in),
                                     receptor_type='inhibitory')
                if len(in2in) != 0:
                    [in_ex, in_in] = split_ex_in(in2in)
                    if len(in_ex) != 0:
                        p.Projection(breakout[breakout_count], breakout[breakout_count], p.FromListConnector(in_ex),
                                     receptor_type='excitatory')
                    if len(in_in) != 0:
                        p.Projection(breakout[breakout_count], breakout[breakout_count], p.FromListConnector(in_in),
                                     receptor_type='inhibitory')
                if len(in2out) != 0:
                    [in_ex, in_in] = split_ex_in(in2out)
                    if len(in_ex) != 0:
                        p.Projection(breakout[breakout_count], output[breakout_count], p.FromListConnector(in_ex),
                                     receptor_type='excitatory')
                    if len(in_in) != 0:
                        p.Projection(breakout[breakout_count], output[breakout_count], p.FromListConnector(in_in),
                                     receptor_type='inhibitory')
                if len(e2in) != 0:
                    p.Projection(excite[excite_count], breakout[breakout_count], p.FromListConnector(e2in),
                                 receptor_type='excitatory')
                if len(i2in) != 0:
                    p.Projection(inhib[inhib_count], breakout[breakout_count], p.FromListConnector(i2in),
                                 receptor_type='inhibitory')
                if len(e2e) != 0:
                    p.Projection(excite[excite_count], excite[excite_count], p.FromListConnector(e2e),
                                 receptor_type='excitatory')
                if len(e2i) != 0:
                    p.Projection(excite[excite_count], inhib[inhib_count], p.FromListConnector(e2i),
                                 receptor_type='excitatory')
                if len(i2e) != 0:
                    p.Projection(inhib[inhib_count], excite[excite_count], p.FromListConnector(i2e),
                                 receptor_type='inhibitory')
                if len(i2i) != 0:
                    p.Projection(inhib[inhib_count], inhib[inhib_count], p.FromListConnector(i2i),
                                 receptor_type='inhibitory')
                if len(e2out) != 0:
                    p.Projection(excite[excite_count], output[breakout_count], p.FromListConnector(e2out),
                                 receptor_type='excitatory')
                if len(i2out) != 0:
                    p.Projection(inhib[inhib_count], output[breakout_count], p.FromListConnector(i2out),
                                 receptor_type='inhibitory')
                if len(out2e) != 0:
                    [out_ex, out_in] = split_ex_in(out2e)
                    if len(out_ex) != 0:
                        p.Projection(output[breakout_count], excite[excite_count], p.FromListConnector(out_ex),
                                     receptor_type='excitatory')
                    if len(out_in) != 0:
                        p.Projection(output[breakout_count], excite[excite_count], p.FromListConnector(out_in),
                                     receptor_type='inhibitory')
                if len(out2i) != 0:
                    [out_ex, out_in] = split_ex_in(out2i)
                    if len(out_ex) != 0:
                        p.Projection(output[breakout_count], inhib[inhib_count], p.FromListConnector(out_ex),
                                     receptor_type='excitatory')
                    if len(out_in) != 0:
                        p.Projection(output[breakout_count], inhib[inhib_count], p.FromListConnector(out_in),
                                     receptor_type='inhibitory')
                if len(out2in) != 0:
                    [out_ex, out_in] = split_ex_in(out2in)
                    if len(out_ex) != 0:
                        p.Projection(output[breakout_count], breakout[breakout_count], p.FromListConnector(out_ex),
                                     receptor_type='excitatory')
                    if len(out_in) != 0:
                        p.Projection(output[breakout_count], breakout[breakout_count], p.FromListConnector(out_in),
                                     receptor_type='inhibitory')
                if len(out2out) != 0:
                    [out_ex, out_in] = split_ex_in(out2out)
                    if len(out_ex) != 0:
                        p.Projection(output[breakout_count], output[breakout_count], p.FromListConnector(out_ex),
                                     receptor_type='excitatory')
                    if len(out_in) != 0:
                        p.Projection(output[breakout_count], output[breakout_count], p.FromListConnector(out_in),
                                     receptor_type='inhibitory')
                if len(e2out) != 0:
                    p.Projection(excite[excite_count], output[breakout_count], p.FromListConnector(e2out),
                                 receptor_type='excitatory')
                if len(i2out) != 0:
                    p.Projection(inhib[inhib_count], output[breakout_count], p.FromListConnector(i2out),
                                 receptor_type='inhibitory')

        print "\nfinished connections seed = ", seed, "\n"
        simulator = get_simulator()
        try:
            print "\nrun seed = ", seed, "\n"
            p.run(runtime)
            try_except = max_attempts
            break
        except:
            traceback.print_exc()
            try:
                print "\nrun 2 seed = ", seed, "\n"
                globals_variables.unset_simulator()
                print "end was necessary"
            except:
                traceback.print_exc()
                print "end wasn't necessary"
            try_except += 1
            print "failed to run on attempt ", try_except, "\n"  # . total fails: ", all_fails, "\n"
            if try_except >= max_attempts:
                print "calling it a failed population, splitting and rerunning"
                return 'fail'
        print "\nfinished run seed = ", seed, "\n"

    scores = []
    agent_fitness = []
    fails = 0
    excite_spike_count = [0 for i in range(len(connections))]
    excite_fail = 0
    inhib_spike_count = [0 for i in range(len(connections))]
    inhib_fail = 0
    print "reading the spikes of ", config, '\n', seed
    for i in range(len(connections)):
        print "started processing fitness of: ", i, '/', len(connections)
        if i in failures:
            print "worst score for the failure"
            fails += 1
            scores.append([[max_fail_score], [max_fail_score], [max_fail_score], [max_fail_score]])
            # agent_fitness.append(scores[i])
            excite_spike_count[i] -= max_fail_score
            inhib_spike_count[i] -= max_fail_score
        else:
            if spike_f:
                if i in excite_marker:
                    print "counting excite spikes"
                    spikes = excite[i - excite_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            excite_spike_count[i] += 1
                else:
                    excite_fail += 1
                    print "had an excite failure"
                if i in inhib_marker:
                    print "counting inhib spikes"
                    spikes = inhib[i - inhib_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            inhib_spike_count[i] += 1
                else:
                    inhib_fail += 1
                    print "had an inhib failure"
            scores.append(get_scores(game_pop=breakout[i - fails], simulator=simulator))
            # pop[i].stats = {'fitness': scores[i][len(scores[i]) - 1][0]}  # , 'steps': 0}
        print "\nfinished spikes", seed, "\n"
        if spike_f:
            agent_fitness.append([scores[i][len(scores[i]) - 1][0], excite_spike_count[i] + inhib_spike_count[i]])
        else:
            agent_fitness.append(scores[i][len(scores[i]) - 1][0])
        # print i, "| e:", excite_spike_count[i], "-i:", inhib_spike_count[i], "|\t", scores[i]
    print seed, "\nThe scores for this run of {} agents are:".format(len(connections))
    for i in range(len(connections)):
        print "c:{}, s:{}, si:{}, si0:{}".format(len(connections), len(scores), len(scores[i]), len(scores[i][0]))
        e_string = "e: {}".format(excite_spike_count[i])
        i_string = "i: {}".format(inhib_spike_count[i])
        score_string = ""
        for j in range(len(scores[i])):
            score_string += "{:4},".format(scores[i][j][0])
        print "{:3} | {:8} {:8} - ".format(i, e_string, i_string), score_string
    print "before end seed = ", seed
    p.end()
    print "\nafter end seed = ", seed, "\n"

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



fitnesses = thread_breakout(connections, arms, split, runtime, exposure_time, noise_rate, noise_weight, reward, size_f, spike_f, True)

print_fitnesses(fitnesses)