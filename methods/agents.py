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
import math
import itertools
from copy import deepcopy
import operator
from spinn_front_end_common.utilities.globals_variables import get_simulator
import traceback

class agent_pop(object):
    def __init__(self,
                 motifs,
                 max_motif_size=4,
                 min_motif_size=2,
                 population_size=200,
                 static_population=True,
                 population_seed=None,
                 read_entire_population=False,
                 discrete_params=True,
                 weights=True,
                 weight_range=(0, 0.1),
                 no_weight_bins=7,
                 initial_weight=0,
                 weight_stdev=0.02,
                 delays=True,
                 delay_range=(1.0, 25.0),
                 no_delay_bins=7,
                 delay_stdev=3.0,
                 initial_hierarchy_depth=1,
                 max_hierarchy_depth=4,
                 selection_metric='fitness',  # fixed, population based, fitness based
                 # starting_weight='uniform',
                 neuron_types=['excitatory', 'inhibitory'],
                 io_config='fixed',  # fixed, dynamic/coded probabilistic, uniform
                 global_io=('highest', 'seeded'), # highest, seeded, random, average
                 multi_synapse=False):

        self.max_motif_size = max_motif_size
        self.min_motif_size = min_motif_size
        self.population_size = population_size
        self.static_population = static_population
        self.population_seed = population_seed
        # self.read_entire_population = read_entire_population
        self.discrete_params = discrete_params
        self.weights = weights
        self.weight_range = weight_range
        self.no_weight_bins = no_weight_bins
        weight_bin_range = self.weight_range[1] - self.weight_range[0]
        self.weight_bin_width = weight_bin_range / (self.no_weight_bins - 1)
        self.initial_weight = initial_weight
        self.weight_stdev = weight_stdev
        self.delays = delays
        self.delay_range = delay_range
        self.no_delay_bins = no_delay_bins
        delay_bin_range = self.delay_range[1] - self.delay_range[0]
        self.delay_bin_width = delay_bin_range / (self.no_delay_bins - 1)
        self.delay_stdev = delay_stdev
        self.initial_hierarchy_depth = initial_hierarchy_depth
        self.max_hierarchy_depth = max_hierarchy_depth
        self.selection_metric = selection_metric
        self.neuron_types = neuron_types
        self.io_config = io_config
        self.global_io = global_io
        self.multi_synapse = multi_synapse

        self.motif_configs = {}  # Tuple of tuples(node types, node i/o P(), connections, selection weight)
        # for types in self.neuron_types:
        #     self.motif_configs[types] = {}
        #     self.motif_configs[types]['depth'] = 0
        self.motifs_generated = 0
        self.total_weight = 0
        self.agent_pop = []
        self.agent_nets = {}


    def new_individual(self):
        motifs.

    def construct_connections(self, agent_connections, seed, inputs, outputs):
        indexed_ex = []
        indexed_in = []
        input_count = {}
        output_count = {}
        e2e = []
        e2i = []
        i2e = []
        i2i = []
        for conn in agent_connections:
            if conn[0][0] == 'excitatory':
                pre_ex = True
                try:
                    pre_index = indexed_ex.index(conn[0])
                    try:
                        output_count['{}'.format(conn[0])] += 1
                    except:
                        output_count['{}'.format(conn[0])] = 1
                except:
                    indexed_ex.append(conn[0])
                    output_count['{}'.format(conn[0])] = 1
                    pre_index = indexed_ex.index(conn[0])
            else:
                pre_ex = False
                try:
                    pre_index = indexed_in.index(conn[0])
                    try:
                        output_count['{}'.format(conn[0])] += 1
                    except:
                        output_count['{}'.format(conn[0])] = 1
                except:
                    indexed_in.append(conn[0])
                    pre_index = indexed_in.index(conn[0])
                    output_count['{}'.format(conn[0])] = 1
            if conn[1][0] == 'excitatory':
                post_ex = True
                try:
                    post_index = indexed_ex.index(conn[1])
                    try:
                        input_count['{}'.format(conn[1])] += 1
                    except:
                        input_count['{}'.format(conn[1])] = 1
                except:
                    indexed_ex.append(conn[1])
                    post_index = indexed_ex.index(conn[1])
                    input_count['{}'.format(conn[1])] = 1
            else:
                post_ex = False
                try:
                    post_index = indexed_in.index(conn[1])
                    try:
                        input_count['{}'.format(conn[1])] += 1
                    except:
                        input_count['{}'.format(conn[1])] = 1
                except:
                    indexed_in.append(conn[1])
                    post_index = indexed_in.index(conn[1])
                    input_count['{}'.format(conn[1])] = 1
            if pre_ex and post_ex:
                e2e.append((pre_index, post_index, conn[2], conn[3]))
            elif pre_ex and not post_ex:
                e2i.append((pre_index, post_index, conn[2], conn[3]))
            elif not pre_ex and post_ex:
                i2e.append((pre_index, post_index, conn[2], conn[3]))
            elif not pre_ex and not post_ex:
                i2i.append((pre_index, post_index, conn[2], conn[3]))
        pre_ex_count = [[0, 'e', j] for j in range(len(indexed_ex))]
        post_ex_count = [[0, 'e', j] for j in range(len(indexed_ex))]
        pre_in_count = [[0, 'i', j] for j in range(len(indexed_in))]
        post_in_count = [[0, 'i', j] for j in range(len(indexed_in))]
        for e in e2e:
            pre_ex_count[e[0]][0] += 1
            post_ex_count[e[1]][0] += 1
        for e in e2i:
            pre_ex_count[e[0]][0] += 1
            post_in_count[e[1]][0] += 1
        for e in i2e:
            pre_in_count[e[0]][0] += 1
            post_ex_count[e[1]][0] += 1
        for e in i2i:
            pre_in_count[e[0]][0] += 1
            post_in_count[e[1]][0] += 1
        pre_ex_count.sort(reverse=True)
        pre_in_count.sort(key=lambda x: x[0], reverse=True)
        post_ex_count.sort(reverse=True)
        post_in_count.sort(reverse=True)
        pre_count = pre_ex_count + pre_in_count
        pre_count.sort(reverse=True)
        post_count = post_in_count + post_ex_count
        post_count.sort(reverse=True)
        in2e = []
        in2i = []
        e2out = []
        i2out = []
        if self.global_io[0] == 'highest':
            if self.global_io[1] == 'seeded':
                np.random.seed(seed)
                input_order = range(inputs)
                np.random.shuffle(input_order)
                output_order = range(outputs)
                np.random.shuffle(output_order)
                io_index = 0
                for node in pre_count:
                    if node[1] == 'e':
                        in2e.append((io_index, node[2], 0.1, 1))
                    else:
                        in2i.append((io_index, node[2], 0.1, 1))
                    io_index += 1
                    if io_index == inputs:
                        break
                io_index = 0
                for node in post_count:
                    if node[1] == 'e':
                        e2out.append((node[2], io_index, 0.1, 1))
                    else:
                        i2out.append((node[2], io_index, 0.1, 1))
                    io_index += 1
                    if io_index == outputs:
                        break

        return in2e, in2i, len(indexed_ex), e2e, e2i, len(indexed_in), i2e, i2i, e2out, i2out


    def convert_population(self, inputs, outputs):
        agent_connections = []
        for agent in self.agent_pop:
            # agent_connections.append(self.read_motif(agent))
            agent_conn = self.read_motif(agent[0])
            spinn_conn = \
                self.construct_connections(agent_conn, agent[1], inputs, outputs)
            agent_connections.append(spinn_conn)
        print "return all the from_list food"
        return agent_connections


    def get_scores(self, game_pop, simulator):
        g_vertex = game_pop._vertex
        scores = g_vertex.get_data(
            'score', simulator.no_machine_time_steps, simulator.placements,
            simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)

        return scores.tolist()


    def bandit_test(self, connections, arms, runtime=2000, exposure_time=200, noise_rate=1, noise_weight=1):
        max_attempts = 5
        try_except = 0
        while try_except < max_attempts:
            bandit = []
            bandit_count = -1
            excite = []
            excite_count = -1
            inhib = []
            inhib_count = -1
            failures = []
            p.setup(timestep=1.0, min_delay=self.delay_range[0], max_delay=self.delay_range[1])
            p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
            for i in range(len(connections)):
                [in2e, in2i, e_size, e2e, e2i, i_size, i2e, i2i, e2out, i2out] = connections[i]
                if (len(in2e) == 0 and len(in2i) == 0) or (len(e2out) == 0 and len(i2out) == 0):
                    failures.append(i)
                    print "agent {} was not properly connected to the game".format(i)
                else:
                    bandit_count += 1
                    bandit.append(
                        p.Population(1, p.Bandit(arms, exposure_time, label='bandit_pop_{}-{}'.format(bandit_count, i))))
                    if e_size > 0:
                        excite_count += 1
                        excite.append(
                            p.Population(e_size, p.IF_cond_exp(), label='excite_pop_{}-{}'.format(excite_count, i)))
                        excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(excite_noise, excite[excite_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if i_size > 0:
                        inhib_count += 1
                        inhib.append(p.Population(i_size, p.IF_cond_exp(), label='inhib_pop_{}-{}'.format(inhib_count, i)))
                        inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(inhib_noise, inhib[inhib_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if len(in2e) != 0:
                        p.Projection(bandit[bandit_count], excite[excite_count], p.FromListConnector(in2e),
                                     receptor_type='excitatory')
                    if len(in2i) != 0:
                        p.Projection(bandit[bandit_count], inhib[inhib_count], p.FromListConnector(in2i),
                                     receptor_type='excitatory')
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
                        p.Projection(excite[excite_count], bandit[bandit_count], p.FromListConnector(e2out),
                                     receptor_type='excitatory')
                    if len(i2out) != 0:
                        p.Projection(inhib[inhib_count], bandit[bandit_count], p.FromListConnector(i2out),
                                     receptor_type='inhibitory')

            simulator = get_simulator()
            try:
                p.run(runtime)
                try_except = max_attempts
                break
            except:
                traceback.print_exc()
                try_except += 1
                print "failed to run on attempt ", try_except, "\n"  # . total fails: ", all_fails, "\n"

        scores = []
        agent_fitness = []
        fails = 0
        for i in range(len(connections)):
            if i in failures:
                fails += 1
                scores.append(-100000)
                agent_fitness.append(scores[i])
                print "worst score for the failure"
            else:
                scores.append(self.get_scores(game_pop=bandit[i - fails], simulator=simulator))
                # pop[i].stats = {'fitness': scores[i][len(scores[i]) - 1][0]}  # , 'steps': 0}
                agent_fitness.append(scores[i][len(scores[i]) - 1][0])

        return agent_fitness