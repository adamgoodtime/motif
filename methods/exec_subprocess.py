import sys, os
import time
import socket
import numpy as np
import math
import itertools
import sys
import termios
import contextlib
from copy import deepcopy
import operator
from spinn_front_end_common.utilities.globals_variables import get_simulator
import traceback
import math
from methods.networks import motif_population
import traceback
import csv
import threading
import subprocess
import pathos.multiprocessing
from spinn_front_end_common.utilities import globals_variables
from ast import literal_eval
from types import ModuleType

# max_fail_score = 0  # -int(runtime / exposure_time)
setup_retry_time = 60
# new_split = agent_pop_size

# stdp_model = p.STDPMechanism(
#     timing_dependence=p.SpikePairRule(tau_plus=20., tau_minus=20.0, A_plus=0.003, A_minus=0.003),
#     weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=0.1))

def wait_timeout(processes, seconds):
    """Wait for a process to finish, or raise exception after timeout"""
    start = time.time()
    end = start + seconds
    interval = 1

    while True:
        finished = 0
        for process in processes:
            result = process.poll()
            if result is not None:
                finished += 1
            elif time.time() >= end:
                process.kill()
                print "\nhad to kill a process, it timed out\n"
                fail = 'fail'
                np.save('fitnesses {} {}.npy'.format(config, processes.index(process)), fail)
                finished += 1
        time.sleep(interval)
        if finished == len(processes):
            return True

def read_results(test_length):
    all_fitnesses = []
    not_a_file = []
    for i in range(test_length):
        try:
            pop_fitness = np.load('fitnesses {} {}.npy'.format(config, i))
            all_fitnesses.append(pop_fitness.tolist())
        except:
            pop_fitness = ['fail', 'fail']
            not_a_file.append(i)
            all_fitnesses.append(pop_fitness)
        # file_name = 'fitnesses {} {}.csv'.format(config, i)
        # with open(file_name) as from_file:
        #     csvFile = csv.reader(from_file)
        #     for row in csvFile:
        #         metric = []
        #         for thing in row:
        #             metric.append(literal_eval(thing))
        #             # if thing == 'fail':
        #             #     metric.append(worst_score)
        #             # else:
        #             #     metric.append(literal_eval(thing))
        #         pop_fitnesses.append(metric)
    remove_results(test_length, not_a_file)
    return all_fitnesses

def remove_results(test_length, not_a_file):
    for i in range(test_length):
        if i not in not_a_file:
            os.remove('fitnesses {} {}.npy'.format(config, i))
            os.remove('data {} {}.npy'.format(config, i))

def write_globals(file_id):
    # non_modules = {}
    # for thing in globals():
    #     if not isinstance(globals()[thing], ModuleType):
    #         non_modules[thing] = globals()[thing]
    # np.save('globals {}.npy'.format(file_id), non_modules)
    with open('globals {}.csv'.format(file_id), 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        for thing in globals():
            if thing != 'connections':
                writer.writerow([thing, globals()[thing]])
        file.close()

def subprocess_experiments(connections, test_data_set, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                  size_f=False, spike_f=False, make_action=True, top=True, parallel=True):
    global new_split
    step_size = int(np.ceil(float(len(connections)) / float(split)))
    if step_size == 0:
        step_size = 1
    if isinstance(test_data_set[0], list):
        connection_threads = []
        if parallel:
            all_configs = [[[connections[x:x + step_size], test_data, split, runtime, exposure_time, noise_rate, noise_weight,
                             spike_f, make_action, exec_thing, np.random.randint(1000000000)] for x in xrange(0, len(connections), step_size)]
                           for test_data in test_data_set]
            for test in all_configs:
                for set_up in test:
                    connection_threads.append(set_up)
        else:
            all_configs = [[[connections[x:x + step_size], test_data_set, split, runtime, exposure_time, noise_rate, noise_weight,
                             spike_f, make_action, exec_thing, np.random.randint(1000000000)] for x in xrange(0, len(connections), step_size)]]
            for test in all_configs:
                for set_up in test:
                    connection_threads.append(set_up)
    else:
        connection_threads = [[connections[x:x + step_size], test_data_set, split, runtime, exposure_time, noise_rate,
                               noise_weight, spike_f, make_action, exec_thing, np.random.randint(1000000000)]
                              for x in xrange(0, len(connections), step_size)]

    write_globals(config)
    process_list = []
    test_id = 0
    for conn_thread in connection_threads:
        if exec_thing == 'erbp':
            call = [sys.executable,
                    '../methods/learn_a_sinusoid.py',
                    config,
                    str(test_id),
                    neuron_type
                    ]
        else:
            call = [sys.executable,
                    '../methods/test_pop.py',
                    config,
                    str(test_id),
                    neuron_type
                    ]
        np.save('data {} {}.npy'.format(config, test_id), conn_thread)
        p = subprocess.Popen(call, stdout=None, stderr=None)
        process_list.append(p)

        test_id += 1
    if exec_thing == 'erbp':
        wait_timeout(process_list, (runtime * 15) + 600)
    else:
        wait_timeout(process_list, ((runtime / 1000) * 15) + 600)

    print "all finished"

    pool_result = read_results(test_id)

    for i in range(len(pool_result)):
        try:
            test_pool = pool_result[i][0]
            test_pool = pool_result[i][1]
        except:
            traceback.print_exc()
            print "it broke and the except caught the return failure"
            pool_result[i] = ['fail', 'fail']
        if pool_result[i][0] == 'fail' and len(connection_threads[i][0]) > 1:
            pool_result[i] = pool_result[i][1]
            if plasticity == 'pall':
                # new_fail = False
                # connection_threads[i].append(pool_result[i])
                # while not new_fail:
                #     random_key = np.random.random()
                #     try:
                #         np.load("failed pop size-{} {} {}".format(len(connection_threads[i][0]), random_key, config))
                #     except:
                #         new_fail = True
                #         np.save("failed pop size-{} {} {}".format(len(connection_threads[i][0]), random_key, config), connection_threads[i])
                # del connection_threads[i][len(connection_threads[i]) - 1]
                split = 2
            elif not top:
                if parallel:
                    split = agent_pop_size
                else:
                    split = 8
            else:
                split = new_split
            print "splitting ", len(connection_threads[i][0]), " into ", split, " pieces"
            problem_arms = connection_threads[i][1]
            pool_result[i] = subprocess_experiments(connection_threads[i][0], problem_arms, split, runtime,
                                                exposure_time, noise_rate, noise_weight, spike_f, top=False, parallel=parallel, make_action=make_action)
        elif pool_result[i][0] == 'fail' and len(connection_threads[i][0]) == 1:
            pool_result[i] = pool_result[i][1]
            new_fail = False
            connection_threads[i].append(pool_result[i])
            while not new_fail:
                random_key = np.random.random()
                try:
                    np.load("failed agent {} {}".format(random_key, config))
                except:
                    new_fail = True
                    np.save("failed agent {} {}".format(random_key, config), connection_threads[i])
            pool_result[i] = 'fail'
        elif pool_result[i][0] == 'complete':
            pool_result[i] = pool_result[i][1]
            print "good return"
        else:
            print "fully bad return"
            if len(connection_threads[i][0]) > 1:
                if plasticity == 'pall':
                    split = 2
                elif not top:
                    if parallel:
                        split = agent_pop_size
                    else:
                        split = int(len(connection_threads[i][0]) / 8)
                else:
                    split = new_split
                print "splitting ", len(connection_threads[i][0]), " into ", split, " pieces"
                problem_arms = connection_threads[i][1]
                pool_result[i] = subprocess_experiments(connection_threads[i][0], problem_arms, split, runtime,
                                                    exposure_time, noise_rate, noise_weight, spike_f, top=False, parallel=parallel, make_action=make_action)
            else:
                pool_result[i] = 'fail'

    agent_fitness = []
    for thread in pool_result:
        if isinstance(thread, list):
            for result in thread:
                agent_fitness.append(result)
        else:
            agent_fitness.append(thread)

    if isinstance(test_data_set[0], list) and top:
        copy_fitness = deepcopy(agent_fitness)
        agent_fitness = []
        for i in range(len(test_data_set)):
            test_results = []
            for j in range(len(connections)):
                try:
                    test_results.append(copy_fitness[(i * len(connections)) + j])
                except:
                    traceback.print_exc()
                    print "\nfailed adding result: set", i, "/", len(test_data_set), "& agent", j, "/", len(connections)
                    print "\ncopy fitness [", len(copy_fitness), "]x[", len(copy_fitness[0]), "]:", copy_fitness
                    print "\nresult so far [", len(test_results), "]x[", len(test_results[0]), "]:", test_results
                    print "\nagent data so far [", len(agent_fitness), "]x[", len(agent_fitness[0]), "]:", agent_fitness
                    print "\npool result [", len(pool_result), "]x[", len(pool_result[0]), "]:", pool_result
                    raise Exception
                    # maybe just run it again, it ain't worth the trouble :(
            agent_fitness.append(test_results)
        if size_f:
            test_results = []
            for i in range(len(connections)):
                test_results.append(connections[i][6] + connections[i][9])
            agent_fitness.append(test_results)

    return agent_fitness

def print_fitnesses(fitnesses):
    # with open('fitnesses {}.csv'.format(config), 'w') as file:
    #     writer = csv.writer(file, delimiter=',', lineterminator='\n')
    #     for fitness in fitnesses:
    #         writer.writerow(fitness)
    #     file.close()
    np.save('fitnesses {}.npy'.format(config), fitnesses)

if threading_tests:
    fitnesses = subprocess_experiments(connections, test_data_set, split, runtime, exposure_time, noise_rate, noise_weight,
                                   size_f, spike_f, make_action, True, parallel)
else:
    fitnesses = pop_test(connections, test_data=arms[0], split=split, runtime=runtime, exposure_time=exposure_time,
                         noise_rate=noise_rate, noise_weight=noise_weight, spike_f=spike_f,
                         make_action=make_action, seed=0)

print_fitnesses(fitnesses)