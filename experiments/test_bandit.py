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
from ast import literal_eval
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

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

def get_scores(game_pop, simulator):
    g_vertex = game_pop._vertex
    scores = g_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
    return scores.tolist()

def read_agent(printing=False):
    agent_connections = []
    with open(file) as from_file:
        csvFile = csv.reader(from_file)
        for row in csvFile:
            if printing:
                print row
            if row[0] == 'fitness':
                break
            agent_connections.append(literal_eval(row[0]))
    return agent_connections

def thread_bandit():
    def helper(args):
        return test_agent(*args)

    if isinstance(arms[0], list):
        pool = pathos.multiprocessing.Pool(processes=len(arms))

        pool.map(func=helper, iterable=arms)
    else:
        test_agent(arms[0], arms[1])

def test_agent(arm1, arm2):

    arm = [arm1, arm2]

    print "arm = ", arm

    connections = read_agent()

    p.setup(timestep=1.0, min_delay=1, max_delay=127)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
    [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
     i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out] = connections
    bandit_model = Bandit(arms=arm,
                         reward_delay=exposure_time,
                         reward_based=reward,
                         rand_seed=[np.random.randint(0xffff) for k in range(4)],
                         label='bandit_pop')
    input_pop_size = bandit_model.neurons()
    bandit_model = p.Population(input_pop_size, bandit_model)
    # added to ensure that the arms and bandit are connected to and from something
    null_pop = p.Population(1, p.IF_cond_exp(), label='null')
    p.Projection(bandit_model, null_pop, p.AllToAllConnector())
    input_arms = []
    for j in range(outputs):
        input_arms.append(p.Population(int(np.ceil(np.log2(outputs))),
                                           Arm(arm_id=j, reward_delay=exposure_time,
                                               rand_seed=[np.random.randint(0xffff) for k in range(4)],
                                               no_arms=outputs, arm_prob=1),
                                               label='arm_pop{}'.format(j)))
        p.Projection(input_arms[j], bandit_model, p.AllToAllConnector(), p.StaticSynapse())
        p.Projection(null_pop, input_arms[j], p.AllToAllConnector())
    if e_size > 0:
        excite = p.Population(e_size, p.IF_cond_exp(), label='excite_pop')
        if noise_rate:
            excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
            p.Projection(excite_noise, excite, p.OneToOneConnector(),
                         p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
        excite.record('spikes')
    if i_size > 0:
        inhib = p.Population(i_size, p.IF_cond_exp(), label='inhib_pop')
        if noise_rate:
            inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
            p.Projection(inhib_noise, inhib, p.OneToOneConnector(),
                         p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
        inhib.record('spikes')
    stdp_model = p.STDPMechanism(
        timing_dependence=p.SpikePairRule(
            tau_plus=20., tau_minus=20.0, A_plus=0.02, A_minus=0.02),
        weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=0.1))
    if len(in2e) != 0:
        [in_ex, in_in] = split_ex_in(in2e)
        if len(in_ex) != 0:
            [plastic, non_plastic] = split_plastic(in_ex)
            if len(plastic) != 0:
                p.Projection(bandit_model, excite, p.FromListConnector(plastic),
                             receptor_type='excitatory', synapse_type=stdp_model)
            if len(non_plastic) != 0:
                p.Projection(bandit_model, excite, p.FromListConnector(non_plastic),
                             receptor_type='excitatory')
        if len(in_in) != 0:
            [plastic, non_plastic] = split_plastic(in_in)
            if len(plastic) != 0:
                p.Projection(bandit_model, excite, p.FromListConnector(plastic),
                             receptor_type='inhibitory', synapse_type=stdp_model)
            if len(non_plastic) != 0:
                p.Projection(bandit_model, excite, p.FromListConnector(non_plastic),
                             receptor_type='inhibitory')
    if len(in2i) != 0:
        [in_ex, in_in] = split_ex_in(in2i)
        if len(in_ex) != 0:
            [plastic, non_plastic] = split_plastic(in_ex)
            if len(plastic) != 0:
                p.Projection(bandit_model, inhib, p.FromListConnector(plastic),
                             receptor_type='excitatory', synapse_type=stdp_model)
            if len(non_plastic) != 0:
                p.Projection(bandit_model, inhib, p.FromListConnector(non_plastic),
                             receptor_type='excitatory')
        if len(in_in) != 0:
            [plastic, non_plastic] = split_plastic(in_in)
            if len(plastic) != 0:
                p.Projection(bandit_model, inhib, p.FromListConnector(plastic),
                             receptor_type='inhibitory', synapse_type=stdp_model)
            if len(non_plastic) != 0:
                p.Projection(bandit_model, inhib, p.FromListConnector(non_plastic),
                             receptor_type='inhibitory')
    if len(in2out) != 0:
        [in_ex, in_in] = split_ex_in(in2out)
        if len(in_ex) != 0:
            [plastic, non_plastic] = split_plastic(in_ex)
            if len(plastic) != 0:
                connect_to_arms(bandit_model, plastic, input_arms, 'excitatory',
                                True, stdp_model=stdp_model)
            if len(non_plastic) != 0:
                connect_to_arms(bandit_model, non_plastic, input_arms, 'excitatory',
                                False, stdp_model=stdp_model)
        if len(in_in) != 0:
            [plastic, non_plastic] = split_plastic(in_in)
            if len(plastic) != 0:
                connect_to_arms(bandit_model, plastic, input_arms, 'inhibitory',
                                True, stdp_model=stdp_model)
            if len(non_plastic) != 0:
                connect_to_arms(bandit_model, non_plastic, input_arms, 'inhibitory',
                                False, stdp_model=stdp_model)
    if len(e2e) != 0:
        [plastic, non_plastic] = split_plastic(e2e)
        if len(plastic) != 0:
            p.Projection(excite, excite, p.FromListConnector(plastic),
                         receptor_type='excitatory', synapse_type=stdp_model)
        if len(non_plastic) != 0:
            p.Projection(excite, excite, p.FromListConnector(non_plastic),
                         receptor_type='excitatory')
    if len(e2i) != 0:
        [plastic, non_plastic] = split_plastic(e2i)
        if len(plastic) != 0:
            p.Projection(excite, inhib, p.FromListConnector(plastic),
                         receptor_type='excitatory', synapse_type=stdp_model)
        if len(non_plastic) != 0:
            p.Projection(excite, inhib, p.FromListConnector(non_plastic),
                         receptor_type='excitatory')
    if len(i2e) != 0:
        [plastic, non_plastic] = split_plastic(i2e)
        if len(plastic) != 0:
            p.Projection(inhib, excite, p.FromListConnector(plastic),
                         receptor_type='inhibitory', synapse_type=stdp_model)
        if len(non_plastic) != 0:
            p.Projection(inhib, excite, p.FromListConnector(non_plastic),
                         receptor_type='inhibitory')
    if len(i2i) != 0:
        [plastic, non_plastic] = split_plastic(i2i)
        if len(plastic) != 0:
            p.Projection(inhib, inhib, p.FromListConnector(plastic),
                         receptor_type='inhibitory', synapse_type=stdp_model)
        if len(non_plastic) != 0:
            p.Projection(inhib, inhib, p.FromListConnector(non_plastic),
                         receptor_type='inhibitory')
    if len(e2out) != 0:
        [plastic, non_plastic] = split_plastic(e2out)
        if len(plastic) != 0:
            connect_to_arms(excite, plastic, input_arms, 'excitatory',
                            True, stdp_model=stdp_model)
        if len(non_plastic) != 0:
            connect_to_arms(excite, non_plastic, input_arms, 'excitatory',
                            False, stdp_model=stdp_model)
    if len(i2out) != 0:
        [plastic, non_plastic] = split_plastic(i2out)
        if len(plastic) != 0:
            connect_to_arms(inhib, plastic, input_arms, 'inhibitory',
                            True, stdp_model=stdp_model)
        if len(non_plastic) != 0:
            connect_to_arms(inhib, non_plastic, input_arms, 'inhibitory',
                            False, stdp_model=stdp_model)

    simulator = get_simulator()
    p.run(runtime)

    scores = get_scores(game_pop=bandit_model, simulator=simulator)
    print scores
    print arm

    e_spikes = excite.get_data('spikes').segments[0].spiketrains
    i_spikes = inhib.get_data('spikes').segments[0].spiketrains
    # v = receive_pop[i].get_data('v').segments[0].filter(name='v')[0]
    plt.figure("[{}, {}] - {}".format(arm1, arm2, scores))
    Figure(
        Panel(e_spikes, xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(i_spikes, xlabel="Time (ms)", ylabel="nID", xticks=True)
    )
    plt.show()

    p.end()

# file = 'best agent 348: score(206), score bandit reward_shape:True, reward:0, noise r-w:100-0.01, arms:[1, 0]-2-0, max_d10, size:False, spikes:False, w_max0.1.csv'
# file = 'best agent 152: score(204), score bandit reward_shape:True, reward:0, noise r-w:0-0.01, arms:[1, 0]-2-0, max_d10, size:False, spikes:False, w_max0.1.csv'
# file = 'best agent 257: score(784), score bandit reward_shape:True, reward:0, noise r-w:0-0.01, arms:[0.1, 0.9]-8-0, max_d7, size:False, spikes:False, w_max0.1.csv'
# file = 'best agent 2: score(352), score bandit reward_shape:True, reward:0, noise r-w:100-0.01, arms:[0.4, 0.6]-4-0, max_d10, size:False, spikes:False, w_max0.1.csv'
# file = 'best agent 282: score(760), fitness bandit reward_shape:True, reward:0, noise r-w:0-0.01, arms:[0.2, 0.8]-8-0, max_d4, size:False, spikes:False, w_max0.1.csv'
file = 'best agent 138 - score(1391) good down to 0.7 .csv'

arms = [0.9, 0.1]
# arms = [0.3, 0.7]
# arms = [0, 1]
# arms = [1, 0]
# arms = [[0.1, 0.9], [0.9, 0.1]]
# arms = [[1, 0], [0, 1]]
# arms = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.15, 0.85], [0.85, 0.15], [1, 0], [0, 1], [0.05, 0.95], [0.95, 0.05]]
# arms = [[0.2, 0.8], [0.8, 0.2]]
# arms = [[0.4, 0.6], [0.6, 0.4]]
# arms = [[0.4, 0.6], [0.6, 0.4], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
arms = [[0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]]
number_of_arms = 2
split = 1

inputs = 2
outputs = 2

reward_shape = True
reward = 0
noise_rate = 0
noise_weight = 0.01
random_arms = 0

runtime = 40000
exposure_time = 200

read_agent(True)

thread_bandit()