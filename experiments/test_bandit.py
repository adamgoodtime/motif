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
        arms_and = [[arm, 0] for arm in arms]

        pool = pathos.multiprocessing.Pool(processes=len(arms_and))

        pool.map(func=helper, iterable=arms_and)
    else:
        test_agent(arms)

def test_agent(arm, empty):

    print "arm = ", arm

    connections = read_agent()

    p.setup(timestep=1.0, min_delay=1, max_delay=127)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
    bandit = p.Population(len(arm), Bandit(arm, exposure_time, reward_based=reward, label='bandit_pop'))
    output = p.Population(len(arms), p.IF_cond_exp(), label='output')
    p.Projection(output, bandit, p.AllToAllConnector(), p.StaticSynapse())

    [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
     i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out] = connections
    if e_size > 0:
        excite = p.Population(e_size, p.IF_cond_exp(), label='excite_pop')
        excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
        p.Projection(excite_noise, excite, p.OneToOneConnector(),
                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
        excite.record('spikes')
    if i_size > 0:
        inhib = p.Population(i_size, p.IF_cond_exp(), label='inhib_pop')
        inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
        p.Projection(inhib_noise, inhib, p.OneToOneConnector(),
                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
        inhib.record('spikes')
    if len(in2e) != 0:
        [in_ex, in_in] = split_ex_in(in2e)
        if len(in_ex) != 0:
            p.Projection(bandit, excite, p.FromListConnector(in_ex),
                         receptor_type='excitatory')
        if len(in_in) != 0:
            p.Projection(bandit, excite, p.FromListConnector(in_in),
                         receptor_type='inhibitory')
    if len(in2i) != 0:
        [in_ex, in_in] = split_ex_in(in2i)
        if len(in_ex) != 0:
            p.Projection(bandit, inhib, p.FromListConnector(in_ex),
                         receptor_type='excitatory')
        if len(in_in) != 0:
            p.Projection(bandit, inhib, p.FromListConnector(in_in),
                         receptor_type='inhibitory')
    if len(in2in) != 0:
        [in_ex, in_in] = split_ex_in(in2in)
        if len(in_ex) != 0:
            p.Projection(bandit, bandit, p.FromListConnector(in_ex),
                         receptor_type='excitatory')
        if len(in_in) != 0:
            p.Projection(bandit, bandit, p.FromListConnector(in_in),
                         receptor_type='inhibitory')
    if len(in2out) != 0:
        [in_ex, in_in] = split_ex_in(in2out)
        if len(in_ex) != 0:
            p.Projection(bandit, output, p.FromListConnector(in_ex),
                         receptor_type='excitatory')
        if len(in_in) != 0:
            p.Projection(bandit, output, p.FromListConnector(in_in),
                         receptor_type='inhibitory')
    if len(e2in) != 0:
        p.Projection(excite, bandit, p.FromListConnector(e2in),
                     receptor_type='excitatory')
    if len(i2in) != 0:
        p.Projection(inhib, bandit, p.FromListConnector(i2in),
                     receptor_type='inhibitory')
    if len(e2e) != 0:
        p.Projection(excite, excite, p.FromListConnector(e2e),
                     receptor_type='excitatory')
    if len(e2i) != 0:
        p.Projection(excite, inhib, p.FromListConnector(e2i),
                     receptor_type='excitatory')
    if len(i2e) != 0:
        p.Projection(inhib, excite, p.FromListConnector(i2e),
                     receptor_type='inhibitory')
    if len(i2i) != 0:
        p.Projection(inhib, inhib, p.FromListConnector(i2i),
                     receptor_type='inhibitory')
    if len(e2out) != 0:
        p.Projection(excite, output, p.FromListConnector(e2out),
                     receptor_type='excitatory')
    if len(i2out) != 0:
        p.Projection(inhib, output, p.FromListConnector(i2out),
                     receptor_type='inhibitory')
    if len(out2e) != 0:
        [out_ex, out_in] = split_ex_in(out2e)
        if len(out_ex) != 0:
            p.Projection(output, excite, p.FromListConnector(out_ex),
                         receptor_type='excitatory')
        if len(out_in) != 0:
            p.Projection(output, excite, p.FromListConnector(out_in),
                         receptor_type='inhibitory')
    if len(out2i) != 0:
        [out_ex, out_in] = split_ex_in(out2i)
        if len(out_ex) != 0:
            p.Projection(output, inhib, p.FromListConnector(out_ex),
                         receptor_type='excitatory')
        if len(out_in) != 0:
            p.Projection(output, inhib, p.FromListConnector(out_in),
                         receptor_type='inhibitory')
    if len(out2in) != 0:
        [out_ex, out_in] = split_ex_in(out2in)
        if len(out_ex) != 0:
            p.Projection(output, bandit, p.FromListConnector(out_ex),
                         receptor_type='excitatory')
        if len(out_in) != 0:
            p.Projection(output, bandit, p.FromListConnector(out_in),
                         receptor_type='inhibitory')
    if len(out2out) != 0:
        [out_ex, out_in] = split_ex_in(out2out)
        if len(out_ex) != 0:
            p.Projection(output, output, p.FromListConnector(out_ex),
                         receptor_type='excitatory')
        if len(out_in) != 0:
            p.Projection(output, output, p.FromListConnector(out_in),
                         receptor_type='inhibitory')
    if len(e2out) != 0:
        p.Projection(excite, output, p.FromListConnector(e2out),
                     receptor_type='excitatory')
    if len(i2out) != 0:
        p.Projection(inhib, output, p.FromListConnector(i2out),
                     receptor_type='inhibitory')

    simulator = get_simulator()
    p.run(runtime)

    scores = get_scores(game_pop=bandit, simulator=simulator)
    print scores
    print arm

    e_spikes = excite.get_data('spikes').segments[0].spiketrains
    i_spikes = inhib.get_data('spikes').segments[0].spiketrains
    # v = receive_pop[i].get_data('v').segments[0].filter(name='v')[0]
    plt.figure("[{}, {}] - {}".format(arm[0], arm[1], scores))
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
file = 'best agent 68: score(456), fitness bandit reward_shape:False, reward:1, noise r-w:0-0.01, arms:[0.8, 0.2]-8-0, max_d5, size:False, spikes:False, w_max0.1, rents0.1, seeded.csv'

arm_1 = 0.8
arm_2 = 0.2
arm_len = 1
arms = []
for i in range(arm_len):
    arms.append([arm_1, arm_2])
    arms.append([arm_2, arm_1])
# arms = [0.9, 0.1]
# arms = [0.1, 0.9]
# arms = [0, 1]
# arms = [1, 0]
# arms = [[0.15, 0.85], [0.85, 0.15], [1, 0], [0, 1], [0.05, 0.95], [0.95, 0.05]]
arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.2, 0.8], [0.8, 0.2]]
# arms = [[0.4, 0.6], [0.6, 0.4]]
# arms = [[0.4, 0.6], [0.6, 0.4], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
number_of_arms = 2
split = 1

reward_shape = True
reward = 0
noise_rate = 0
noise_weight = 0.01
random_arms = 0

runtime = 40000
exposure_time = 200

read_agent(True)

thread_bandit()