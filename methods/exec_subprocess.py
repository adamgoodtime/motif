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
from python_models.pendulum import Pendulum
from rank_inverted_pendulum.python_models.rank_pendulum import Rank_Pendulum
from spinn_arm.python_models.arm import Arm
# from spinn_breakout import Breakout
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
    for i in range(test_length):
        pop_fitness = np.load('fitnesses {} {}.npy'.format(config, i))
        all_fitnesses.append(pop_fitness.tolist())
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
    remove_results(test_length)
    return all_fitnesses

def remove_results(test_length):
    for i in range(test_length):
        os.remove('fitnesses {} {}.npy'.format(config, i))
        os.remove('data {} {}.npy'.format(config, i))

def write_globals(file_id):
    with open('globals {}.csv'.format(file_id), 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        for thing in globals():
            if thing != 'connections':
                writer.writerow([thing, globals()[thing]])
        file.close()

def subprocess_experiments(connections, test_data_set, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                  reward=0, size_f=False, spike_f=False, top=True):

    step_size = len(connections) / split
    if step_size == 0:
        step_size = 1
    if isinstance(test_data_set[0], list):
        connection_threads = []
        all_configs = [[[connections[x:x + step_size], test_data, split, runtime, exposure_time, noise_rate, noise_weight,
                         reward, spike_f, exec_thing, np.random.randint(1000000000)] for x in xrange(0, len(connections), step_size)]
                       for test_data in test_data_set]
        for test in all_configs:
            for set_up in test:
                connection_threads.append(set_up)
    else:
        connection_threads = [[connections[x:x + step_size], test_data_set, split, runtime, exposure_time, noise_rate,
                               noise_weight, reward, spike_f, exec_thing, np.random.randint(1000000000)]
                              for x in xrange(0, len(connections), step_size)]

    write_globals(config)
    process_list = []
    test_id = 0
    for conn_thread in connection_threads:
        call = [sys.executable,
                '../methods/test_pop.py',
                config,
                str(test_id)
                ]
        np.save('data {} {}.npy'.format(config, test_id), conn_thread)
        p = subprocess.Popen(call, stdout=None, stderr=None)
        process_list.append(p)

        test_id += 1
    
    wait_timeout(process_list, 600)
        
    pool_result = read_results(test_id)

    for i in range(len(pool_result)):
        if pool_result[i] == 'fail' and len(connection_threads[i][0]) > 1:
            print "splitting ", len(connection_threads[i][0]), " into ", new_split, " pieces"
            problem_arms = connection_threads[i][1]
            pool_result[i] = subprocess_experiments(connection_threads[i][0], problem_arms, new_split, runtime,
                                                exposure_time, noise_rate, noise_weight, reward, spike_f, top=False)

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
            for j in range(agent_pop_size):
                test_results.append(copy_fitness[(i * agent_pop_size) + j])
            agent_fitness.append(test_results)
        if size_f:
            test_results = []
            for i in range(agent_pop_size):
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
                                   reward, size_f, spike_f, True)
else:
    fitnesses = pop_test(connections, test_data=arms[0], split=split, runtime=runtime, exposure_time=exposure_time,
                         noise_rate=noise_rate, noise_weight=noise_weight, reward=reward, spike_f=spike_f, seed=0)

print_fitnesses(fitnesses)