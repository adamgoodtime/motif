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
                 weight_range=(-0.1, 0.1),
                 no_weight_bins=7,
                 initial_weight=0,
                 weight_stdev=0.02,
                 delays=True,
                 delay_range=(0.0, 25.0),
                 no_delay_bins=7,
                 delay_stdev=3.0,
                 initial_hierarchy_depth=2,
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
        self.motifs_generated = 0
        self.total_weight = 0

        true_or_false = [True, False]

        if not read_entire_population:
            print "generating population"
            if self.population_seed is not None:
                for seed in self.population_seed:
                    self.motif_configs.update(seed)
                start_point = len(self.population_seed)
            else:
                start_point = 0
            if self.discrete_params and not self.multi_synapse:
                maximum_number_of_motifs = 0
                for i in range(self.min_motif_size, self.max_motif_size + 1):
                    maximum_number_of_motifs += math.pow(i, 2)  # possible connections
                if self.weights:
                    maximum_number_of_motifs *= self.no_weight_bins
                if self.delays:
                    maximum_number_of_motifs *= self.no_delay_bins
                if self.io_config == 'fixed':
                    maximum_number_of_motifs *= i * 4  # possible io configs
                maximum_number_of_motifs *= i * 2
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
                                conn.append(bin * self.weight_bin_width)
                                bin = np.random.randint(0, self.no_delay_bins)
                                conn.append(bin * self.delay_bin_width)
                                synapses.append(conn)
                motif['node'] = node_types
                motif['io'] = io_properties
                motif['conn'] = conn
                motif['depth'] = 1
                motif['id'] = self.motifs_generated
                if self.selection_metric == 'fitness':
                    motif['weight'] = 1
                if not repeats:
                    done = 0
                    for config in self.motif_configs:
                        if self.motif_configs[config] == motif:
                            done += 1
                            print "found a samey", i, "done:", done
                    if done == 0:
                        self.insert_motif(motif)
                    else:
                        i -= 1
                else:
                    self.insert_motif(motif)
                i += 1

        else:
            print "reading from file"

    def select_motif(self):
        if self.total_weight == 0:
            for motif in self.motif_configs:
                self.total_weight += self.motif_configs[motif]['weight']
        choice = np.random.uniform(0, self.total_weight)
        for motif in self.motif_configs:
            choice -= self.motif_configs[motif]['weight']
            if choice < 0:
                break
        return self.motif_configs[motif]

    def insert_motif(self, motif):
        self.motif_configs['{}'.format(self.motifs_generated)] = motif
        self.motifs_generated += 1

    def motif_of_motif(self, motif, min_layer, max_layer):
        i = 0
        layer = 0
        for node in motif['node']:
            if node == 'excitatory' or node == 'inhibitory':
                selected_motif = self.select_motif()
                motif['node'][i] = selected_motif['id']
                motif['depth'] += 1
                i += 1

    def generate_agents(self,
                        pop_size=200,
                        start_small=False):
        print "constructing population of agents"
        for i in range(pop_size):
            # select depth of the agent
            if not start_small:
                depth = np.random.randint(self.initial_hierarchy_depth, self.max_hierarchy_depth)
            else:
                depth = self.initial_hierarchy_depth
            motif = None
            # generate the agent
            for j in range(depth):
                # check if it's the first iteration
                if motif is None:
                    motif = self.select_motif()
                else:
                    # if motif['depth'] >= depth-j:
                    #     agent = motif['id']
                    # else:

                    # select a motif and iterate through the desired number of times to generate agent depth
                    index = 0
                    for node in motif['node']:
                        if node == 'excitatory' or node == 'inhibitory':
                            selected_motif = self.select_motif()
                            motif['node'][index] = selected_motif['id']
                        else:
                            if motif['depth']+j > depth:
                                None
                            else:
                                for


        return 0



