# import spynnaker8 as p
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


# import random


class motif_population(object):
    def __init__(self,
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
                 delay_range=(0.0, 25.0),
                 no_delay_bins=7,
                 delay_stdev=3.0,
                 initial_hierarchy_depth=1,
                 max_hierarchy_depth=4,
                 selection_metric='fitness',  # fixed, population based, fitness based
                 # starting_weight='uniform',
                 neuron_types=['excitatory', 'inhibitory'],
                 io_config='fixed',  # fixed, dynamic/coded probabilistic, uniform
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
        self.multi_synapse = multi_synapse

        self.motif_configs = {}  # Tuple of tuples(node types, node i/o P(), connections, selection weight)
        # for types in self.neuron_types:
        #     self.motif_configs[types] = {}
        #     self.motif_configs[types]['depth'] = 0
        self.motifs_generated = 0
        self.total_weight = 0
        self.agent_pop = []
        self.agent_nets = {}

        true_or_false = [True, False]

        if not read_entire_population:
            print "generating population"
            if self.population_seed is not None:
                for seed in self.population_seed:
                    # or just make this an insert and start = motifs_gened
                    self.motif_configs.update(seed)
                start_point = len(self.population_seed)
            else:
                start_point = 0
            if self.discrete_params and not self.multi_synapse:
                maximum_number_of_motifs = 1
                for i in range(self.min_motif_size, self.max_motif_size + 1):
                    maximum_number_of_motifs *= math.pow(4, i)  # possible connections
                if self.weights:
                    maximum_number_of_motifs *= self.no_weight_bins
                if self.delays:
                    maximum_number_of_motifs *= self.no_delay_bins
                if self.io_config == 'fixed':
                    maximum_number_of_motifs *= math.pow(4, i)  # possible io configs
                maximum_number_of_motifs *= math.pow(2, i)  # exit/inhib
            if self.population_size > maximum_number_of_motifs:
                print "\nPopulation size is bigger than the full spectrum of possible motifs.\n" \
                      "Repeats will be allowed during generation.\n"
                repeats = True
            else:
                repeats = False
            i = start_point
            while i < population_size:
                motif = {}
                node_types = []
                io_properties = []
                synapses = []
                number_of_neurons = np.random.randint(self.min_motif_size, self.max_motif_size + 1)
                for j in range(number_of_neurons):
                    node_types.append(np.random.choice(self.neuron_types))
                    if self.io_config == 'fixed':
                        io_properties.append((np.random.choice(true_or_false), np.random.choice(true_or_false)))
                    else:
                        print "incompatible io config"
                        # todo figure out how to error out
                    for k in range(number_of_neurons):
                        if np.random.choice(true_or_false):
                            if self.discrete_params:
                                conn = []
                                conn.append(j)
                                conn.append(k)
                                bin = np.random.randint(0, self.no_weight_bins)
                                conn.append(weight_range[0] + (bin * self.weight_bin_width))
                                bin = np.random.randint(0, self.no_delay_bins)
                                conn.append(delay_range[0] + (bin * self.delay_bin_width))
                                synapses.append(conn)
                motif['node'] = node_types
                motif['io'] = io_properties
                motif['conn'] = synapses
                motif['depth'] = 1
                # motif['id'] = self.motifs_generated
                if self.selection_metric == 'fitness':
                    weight = 1
                # if not repeats:
                if not self.id_check(motif):
                    # print self.id_check(motif)
                    self.insert_motif(motif, weight, False)
                else:
                    print "repeated ", i, self.id_check(motif)
                    i -= 1
                # else:
                #     self.insert_motif(motif)
                i += 1

        else:
            print "reading from file"
        print "done generating motif pop"

    def permutations(self, motif):
        motif_size = len(motif['node'])
        connections = len(motif['conn'])
        permutations = list(itertools.permutations(range(len(motif['node']))))
        motif_array = []
        for i in range(len(permutations)):
            motif_array.append(deepcopy(motif))
            for j in range(motif_size):
                motif_array[i]['node'][j] = motif['node'][permutations[i][j]]
                motif_array[i]['io'][j] = motif['io'][permutations[i][j]]
            for j in range(connections):
                motif_array[i]['conn'][j][0] = permutations[i][motif_array[i]['conn'][j][0]]
                motif_array[i]['conn'][j][1] = permutations[i][motif_array[i]['conn'][j][1]]
        return motif_array

    def id_check(self, motif):
        motif_id = False
        motif_array = self.permutations(motif)
        for config in self.motif_configs:
            if len(self.motif_configs[config]['node']) == len(motif['node']):
                if len(self.motif_configs[config]['conn']) == len(motif['conn']):
                    # for orig_node in self.motif_configs[config]['node']:
                    #     for check_node in motif['node']:
                    for isomer in motif_array:
                        # compare each element and return rotations with which they are similar
                        if self.motif_configs[config]['node'] == isomer['node'] and \
                                        self.motif_configs[config]['io'] == isomer['io'] and \
                                        self.motif_configs[config]['conn'] == isomer['conn']:
                            motif_id = config
                            break
                    if motif_id:
                        break
        return motif_id

    def select_motif(self):
        if self.total_weight == 0:
            for motif in self.motif_configs:
                self.total_weight += self.motif_configs[motif]['weight']
        choice = np.random.uniform(0, self.total_weight)
        for motif in self.motif_configs:
            choice -= self.motif_configs[motif]['weight']
            if choice < 0:
                break
        # return self.motif_configs[motif]
        return motif

    def insert_motif(self, motif, weight=0, check=True):
        if check == True:
            check = self.id_check(motif)
        if not check:
            motif_id = self.motifs_generated
            self.motif_configs['{}'.format(motif_id)] = motif
            self.motif_configs['{}'.format(motif_id)]['weight'] = weight
            self.motif_configs['{}'.format(motif_id)]['id'] = motif_id
            self.motifs_generated += 1
            # print motif_id
            return '{}'.format(self.motifs_generated - 1)
        else:
            return check

    def motif_of_motif(self, motif_id, config, max_depth, current_depth=0):
        # add layer at lowest level
        # add layer at specific level
        # add motif with a certain probability
        # possibly combine with a mutate operation?
        motif = deepcopy(self.motif_configs[motif_id])
        if current_depth < max_depth:
            i = 0
            layer = 0
            if config == 'lowest':
                None
            elif config <= 1:
                increased_depth = False
                picked_bigger = False
                for node in motif['node']:
                    if node == 'excitatory' or node == 'inhibitory':
                        if np.random.random() < config:
                            motif['node'][i] = self.select_motif()
                            increased_depth = True
                    else:
                        sub_motif = self.motif_of_motif(node, config, max_depth, current_depth + 1)
                        # sub_motif_id = self.insert_motif(sub_motif)
                        # motif['node'][i] = sub_motif[sub_motif_id]
                        motif['node'][i] = self.insert_motif(sub_motif)
                        new_depth = sub_motif['depth']
                        if new_depth >= motif['depth']:
                            motif['depth'] = new_depth + 1
                            picked_bigger = True
                    i += 1
                if increased_depth and not picked_bigger:
                    motif['depth'] += 1
            else:
                # go to a certain depth
                None
        return motif

    def generate_agents(self, inputs, outputs, pop_size=200, connfig=1, start_small=False, max_depth=2):
        print "constructing population of agents"
        self.agent_pop = []
        for i in range(pop_size):
            # select depth of the agent
            if not start_small:  # this is broke af
                depth = np.random.randint(self.initial_hierarchy_depth, max_depth + 1)
            else:
                depth = self.initial_hierarchy_depth
            # motif = None
            motif = self.select_motif()
            if motif is None:
                motif = self.select_motif()
            else:
                for i in range(depth):
                    motif = self.motif_of_motif(motif, connfig, depth, i)
                    motif = self.insert_motif(motif)
            self.agent_pop.append(motif)
        return self.agent_pop

    def collect_IO(self, node, prepost, upper, layer, node_array=[]):
        motif = self.motif_configs[node]
        node_count = 0
        for io in motif['io']:
            local_upper = deepcopy(upper)
            if prepost == 'pre' and io[1]:
                pre_node = motif['node'][node_count]
                if pre_node == 'excitatory' or pre_node == 'inhibitory':
                    node_array.append([pre_node, node_count, local_upper, layer])
                else:
                    local_upper.append(node_count)
                    self.collect_IO(pre_node, prepost, local_upper, layer + 1, node_array)
            if prepost == 'post' and io[0]:
                post_node = motif['node'][node_count]
                if post_node == 'excitatory' or post_node == 'inhibitory':
                    node_array.append([post_node, node_count, local_upper, layer])
                else:
                    local_upper.append(node_count)
                    self.collect_IO(post_node, prepost, local_upper, layer + 1, node_array)
            node_count += 1
        return node_array

    def connect_nodes(self, pre_node, pre_count, post_node, post_count, layer, upper, weight,
                      delay):
        pre_ids = []
        post_ids = []
        connections = []
        if pre_node == 'excitatory' or pre_node == 'inhibitory':
            pre_ids.append([pre_node, pre_count, upper, layer])
        else:
            new_pre_count = upper + [pre_count]
            self.collect_IO(pre_node, 'pre', new_pre_count, layer + 1, pre_ids)
        if post_node == 'excitatory' or post_node == 'inhibitory':
            post_ids.append([post_node, post_count, upper, layer])
        else:
            new_post_count = upper + [post_count]
            self.collect_IO(post_node, 'post', new_post_count, layer + 1, post_ids)
        for pre in pre_ids:
            for post in post_ids:
                connections.append([pre, post, weight, delay])
        return connections

    def read_motif(self, motif_id, layer=0, upper=[]):
        motif = self.motif_configs[motif_id]
        all_connections = []
        for conn in motif['conn']:
            pre = conn[0]
            pre_up = []
            pre_up.append(pre)
            post = conn[1]
            post_up = []
            post_up.append(post)
            weight = conn[2]
            delay = conn[3]
            pre_node = motif['node'][pre]
            post_node = motif['node'][post]
            all_connections += self.connect_nodes(pre_node, pre, post_node, post, layer,
                                                  upper, weight, delay)
        node_count = 0
        for node in motif['node']:
            if node != 'excitatory' and node != 'inhibitory':
                local_upper = deepcopy(upper)
                local_upper.append(node_count)
                all_connections += self.read_motif(node, layer + 1, local_upper)
            node_count += 1
        return all_connections

    def construct_connections(self, agent_connections):
        indexed_ex = []
        indexed_in = []
        e2e = []
        e2i = []
        i2e = []
        i2i = []
        pre_ex = True
        post_ex = True
        for conn in agent_connections:
            if conn[0][0] == 'excitatory':
                pre_ex = True
                try:
                    pre_index = indexed_ex.index(conn[0])
                except:
                    indexed_ex.append(conn[0])
                    pre_index = indexed_ex.index(conn[0])
            else:
                pre_ex = False
                try:
                    pre_index = indexed_in.index(conn[0])
                except:
                    indexed_in.append(conn[0])
                    pre_index = indexed_in.index(conn[0])
            if conn[1][0] == 'excitatory':
                post_ex = True
                try:
                    post_index = indexed_ex.index(conn[1])
                except:
                    indexed_ex.append(conn[1])
                    post_index = indexed_ex.index(conn[1])
            else:
                post_ex = False
                try:
                    post_index = indexed_in.index(conn[1])
                except:
                    indexed_in.append(conn[1])
                    post_index = indexed_in.index(conn[1])
            if pre_ex and post_ex:
                e2e.append((pre_index, post_index, conn[2], conn[3]))
            elif pre_ex and not post_ex:
                e2i.append((pre_index, post_index, conn[2], conn[3]))
            elif not pre_ex and post_ex:
                i2e.append((pre_index, post_index, conn[2], conn[3]))
            elif not pre_ex and not post_ex:
                i2i.append((pre_index, post_index, conn[2], conn[3]))
        return e2e, e2i, i2e, i2i

    def convert_population(self):
        agent_connections = []
        for agent in self.agent_pop:
            # agent_connections.append(self.read_motif(agent))
            agent_conn = self.read_motif(agent)
            [e2e, e2i, i2e, i2i] = self.construct_connections(agent_conn)
            print "convert the connections into a spinn_net"


class species(object):
    def __init__(self, initial_member):
        self.members = [initial_member]
        self.representative = initial_member
        self.offspring = 0
        self.age = 0
        self.avg_fitness = 0.
        self.max_fitness = 0.
        self.max_fitness_prev = 0.
        self.no_improvement_age = 0
        self.has_best = False
