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

def get_scores(game_pop, simulator):
    g_vertex = game_pop._vertex
    scores = g_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
    return scores.tolist()

def read_agent():
    agent_connections = []
    with open(file) as from_file:
        csvFile = csv.reader(from_file)
        for row in csvFile:
            print row
            print "0", row[0]
            if row[0] == 'fitness':
                break
            agent_connections.append(literal_eval(row[0]))
    return agent_connections

def test_agent(arm):
    connections = read_agent()

    p.setup(timestep=1.0, min_delay=1, max_delay=127)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
    bandit = p.Population(len(arm), Bandit(arm, exposure_time, reward_based=reward, label='bandit_pop'))
    [in2e, in2i, e_size, e2e, e2i, i_size, i2e, i2i, e2out, i2out] = connections
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
        p.Projection(bandit, excite, p.FromListConnector(in2e),
                     receptor_type='excitatory')
        # p.Projection(starting_pistol, excite, p.FromListConnector(in2e),
        #              receptor_type='excitatory')
    if len(in2i) != 0:
        p.Projection(bandit, inhib, p.FromListConnector(in2i),
                     receptor_type='excitatory')
        # p.Projection(starting_pistol, inhib, p.FromListConnector(in2i),
        #              receptor_type='excitatory')
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
        p.Projection(excite, bandit, p.FromListConnector(e2out),
                     receptor_type='excitatory')
    if len(i2out) != 0:
        p.Projection(inhib, bandit, p.FromListConnector(i2out),
                     receptor_type='inhibitory')

    simulator = get_simulator()
    p.run(runtime)

    scores = get_scores(game_pop=bandit, simulator=simulator)
    print scores
    p.end()

file = 'best agent 348: score(206), score bandit reward_shape:True, reward:0, noise r-w:100-0.01, arms:[1, 0]-2-0, max_d10, size:False, spikes:False, w_max0.1.csv'

# arms = [0.9, 0.1]
arms = [0, 0.9]
# arms = [[0.1, 0.9], [0.9, 0.1]]
# arms = [[1, 0], [0, 1]]
# arms = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.2, 0.8], [0.8, 0.2]]
# arms = [[0.4, 0.6], [0.6, 0.4]]
# arms = [[0.4, 0.6], [0.6, 0.4], [0.1, 0.9], [0.9, 0.1]]
number_of_arms = 2
split = 1

reward_shape = True
reward = 0
noise_rate = 100
noise_weight = 0.01
random_arms = 0

runtime = 40000
exposure_time = 200

test_agent(arms)