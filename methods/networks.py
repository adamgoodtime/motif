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
import csv
from ast import literal_eval
import fnmatch

# import random


class motif_population(object):
    def __init__(self,
                 neurons,
                 max_motif_size=4,
                 min_motif_size=2,
                 population_size=200,
                 static_population=True,
                 population_seed=None,
                 read_entire_population=False,
                 keep_reading=0,
                 discrete_params=True,
                 plasticity=True,
                 weights=True,
                 weight_range=(0, 0.1),
                 no_weight_bins=7,
                 initial_weight=0,
                 weight_stdev=0.02,
                 delays=True,
                 delay_range=(1.0, 25.0),
                 no_delay_bins=7,
                 delay_stdev=3.0,
                 constant_delays=0,
                 initial_hierarchy_depth=1,
                 max_hierarchy_depth=4,
                 selection_metric='fitness',  # fixed, population based, fitness based
                 # starting_weight='uniform',
                 neuron_types=['excitatory', 'inhibitory'],
                 io_config='fixed',  # fixed, dynamic/coded probabilistic, uniform
                 global_io=('highest', 'unseeded'), # highest, seeded, random, average
                 multi_synapse=False):

        self.neurons = neurons

        self.max_motif_size = max_motif_size
        self.min_motif_size = min_motif_size
        self.population_size = population_size
        self.static_population = static_population
        self.population_seed = population_seed
        self.read_entire_population = read_entire_population
        self.keep_reading = keep_reading
        self.discrete_params = discrete_params
        self.plasticity = plasticity
        self.weights = weights
        self.weight_range = weight_range
        if isinstance(no_weight_bins, list):
            self.no_weight_bins = no_weight_bins[0]
        else:
            self.no_weight_bins = no_weight_bins
        weight_bin_range = self.weight_range[1] - self.weight_range[0]
        self.weight_bin_width = weight_bin_range / (self.no_weight_bins - 1)
        self.initial_weight = initial_weight
        self.weight_stdev = weight_stdev
        self.delays = delays
        self.delay_range = delay_range
        if isinstance(no_delay_bins, list):
            self.no_delay_bins = no_delay_bins[0]
        else:
            self.no_delay_bins = no_delay_bins
        delay_bin_range = self.delay_range[1] - self.delay_range[0]
        self.delay_bin_width = delay_bin_range / (self.no_delay_bins - 1)
        self.delay_stdev = delay_stdev
        self.constant_delays = constant_delays
        self.initial_hierarchy_depth = initial_hierarchy_depth
        self.max_hierarchy_depth = max_hierarchy_depth
        self.selection_metric = selection_metric
        self.inputs = self.neurons.inputs
        self.outputs = self.neurons.outputs
        self.io_config = io_config
        self.global_io = global_io
        self.multi_synapse = multi_synapse

        self.motif_configs = {}  # Tuple of tuples(node types, node i/o P(), connections, selection weight)
        self.motifs_generated = 0
        self.total_weight = 0

        true_or_false = [True, False]

        '''Generate the motif population, either by reading the entire population from a file, seeding the population
        with a population of motifs and generating further motifs to fill the population or creating the entire 
        population from scratch
        '''
        if not read_entire_population:
            print "generating the population"
            if self.population_seed is not None:
                for seed in self.population_seed:
                    # or just make this an insert and start = motifs_gened
                    self.motif_configs.update(seed)
                start_point = len(self.population_seed)
            else:
                start_point = 0
            # attempt to calculate the maximum number of motifs although seems redundant now as max size is very large
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
            # create the remaining population
            while i < population_size:
                self.generate_motif()
                i += 1

        else:
            self.read_population()

        print "done generating motif pop"

    '''Reads a motif population from a csv file and inputs each motif into the set'''
    def read_population(self):
        with open(self.read_entire_population) as from_file:
            csvFile = csv.reader(from_file)
            motif = False
            for row in csvFile:
                temp = row
                if temp[0] == 'node':
                    if motif:
                        self.insert_motif(deepcopy(motif), weight=motif['weight'], read=True)
                    motif = {}
                atribute = temp[0]
                del temp[0]
                if atribute == 'depth':
                    temp = int(temp[0])
                elif atribute == 'weight':
                    temp = literal_eval(temp[0])
                elif atribute == 'conn' or atribute == 'io':
                    for i in range(len(temp)):
                        temp[i] = literal_eval(temp[i])
                elif atribute == 'id':
                    temp = temp[0]
                motif['{}'.format(atribute)] = temp

            self.insert_motif(deepcopy(motif), weight=motif['weight'], read=True)

        # print "done generating motif pop"

    def set_delay_bins(self, bins, iteration, max_iterations):
        if isinstance(bins, list):
            bins_range = bins[1] - bins[0]
            bin_width = max_iterations / float(bins_range + 1)
            self.no_delay_bins = bins[0] + int(iteration / bin_width)
        else:
            self.no_delay_bins = bins
        delay_bin_range = self.delay_range[1] - self.delay_range[0]
        self.delay_bin_width = delay_bin_range / (self.no_delay_bins - 1)

    def set_weight_bins(self, bins, iteration, max_iterations):
        if isinstance(bins, list):
            bins_range = bins[1] - bins[0]
            bin_width = max_iterations / float(bins_range + 1)
            self.no_weight_bins = bins[0] + int(iteration / bin_width)
        else:
            self.no_weight_bins = bins
        weight_bin_range = self.weight_range[1] - self.weight_range[0]
        self.weight_bin_width = weight_bin_range / (self.no_weight_bins - 1)

    '''Generates a random motif within the allowable configurations and attempts to enter it into the population. If it
    already exists within the population the function will be called again until a novel motif is added.'''
    def generate_motif(self, weight=None):
        true_or_false = [True, False]
        motif = {}
        node_types = []
        io_properties = []
        synapses = []
        # selects motif size randomly
        number_of_neurons = np.random.randint(self.min_motif_size, self.max_motif_size + 1)
        for j in range(number_of_neurons):
            node_types.append(self.neurons.choose_neuron())
            # sets the input/output dynamics of each neuron with 50% P() todo set this with a certain probability
            if self.io_config == 'fixed':
                io_properties.append((np.random.choice(true_or_false), np.random.choice(true_or_false)))
            else:
                print "incompatible io config"
                # todo enable erroring out
            # connects the neurons with a 50% chance and sets random weights/delays within the allowed discrete range
            for k in range(number_of_neurons):
                if np.random.choice(true_or_false):
                    if self.discrete_params:
                        conn = []
                        conn.append(j)
                        conn.append(k)
                        bin = np.random.randint(0, self.no_weight_bins)
                        conn.append(self.weight_range[0] + (bin * self.weight_bin_width))
                        bin = np.random.randint(0, self.no_delay_bins)
                        conn.append(self.delay_range[0] + (bin * self.delay_bin_width))
                        if np.random.choice(true_or_false) and self.plasticity:
                            conn.append('plastic')
                        else:
                            conn.append('non-plastic')
                        if conn[2] != 0:
                            synapses.append(conn)
        # moves the generated motifs properties into a dict representing the motif
        motif['node'] = node_types
        motif['io'] = io_properties
        motif['conn'] = synapses
        motif['depth'] = 1
        # gives the motifs a particular weight altering it's chance of being chosen later
        if self.selection_metric == 'fitness' and weight is None:
            weight = 1
        else:
            weight = weight
        # attempts to insert the motif
        if not self.id_check(motif):
            # print self.id_check(motif)
            id = self.insert_motif(motif, weight, False)
        else:
            id = self.generate_motif(weight)
        return id

    '''Creates an array of all possible orientations of a motifs which have the same dynamic and structural properties 
    to allow the correct checking of whether a motif is already within the population'''
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

    '''Compare a motif with all other motifs in the population and return the id of the similar motif or return false if
    there are no identical motifs'''
    def id_check(self, motif):
        motif_id = False
        # acquire the array of all similar permuation of the motif in question
        motif_array = self.permutations(motif)
        for config in self.motif_configs:
            # do a simple check of motif dimesions first to skip checking through the whole array if possible
            if len(self.motif_configs[config]['node']) == len(motif['node']) and \
                    len(self.motif_configs[config]['conn']) == len(motif['conn']):
                # if it passes initial check compare it with a possible permutations
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

    '''This function returns a random motif with probability relative to it's weight and the population total weight. If
    the total population weight is set to 0 elsewhere in the program, indicating a unplanned change, the new weight is 
    determined before selecting a motif'''
    def select_motif(self):
        # ensure total weight is set
        if self.total_weight == 0:
            for motif in self.motif_configs:
                self.total_weight += self.motif_configs[motif]['weight']
        # select a random possition within the population weight range
        choice = np.random.uniform(0, self.total_weight)
        # cycle through the motifs subtracting the motif weight until the value drops below 0 indicating the selection
        for motif in self.motif_configs:
            choice -= self.motif_configs[motif]['weight']
            if choice < 0:
                break
        return motif

    '''Inserts the motif into the population. First however the motif is compared with other motifs to ensure it does 
    not already exist with the population. Return the new id if it is successful or false if not.'''
    def insert_motif(self, motif, weight=0, check=True, read=False):
        # check for repeats
        if check == True:
            check = self.id_check(motif)
        # if it does not exist the insert the motif into the population
        if not check:
            self.total_weight += weight
            if read:
                motif_id = motif['id']
            else:
                does_it_exist = True
                while does_it_exist:
                    try:
                        does_it_exist = self.motif_configs['{}'.format(self.motifs_generated)]
                        print self.motifs_generated, "existed"
                        self.motifs_generated += 1
                    except:
                        # traceback.print_exc()
                        motif_id = self.motifs_generated
                        does_it_exist = False

            self.motif_configs['{}'.format(motif_id)] = motif
            self.motif_configs['{}'.format(motif_id)]['weight'] = weight
            self.motif_configs['{}'.format(motif_id)]['id'] = motif_id
            self.motifs_generated += 1
            # print motif_id
            if read:
                return motif_id
            else:
                return '{}'.format(self.motifs_generated - 1)
        else:
            return check

    '''Creates a hierachical motif of specified depth. Mainly used in agent generation. Selects a motif and replaces 
    each node in the motif with a pointer to another motif. Each motifs and submotif etc is inserted into the population'''
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
            # Check config is set to a P() of a new connection
            elif config <= 1:
                increased_depth = False
                picked_bigger = False
                for node in motif['node']:
                    # if it is a base node replace it with a motif
                    if node in self.neurons.neuron_configs:
                        # with a certain P() add a new motif
                        if np.random.random() < config:
                            selected_motif = self.select_motif()
                            if self.motif_configs[selected_motif]['depth'] + current_depth < max_depth:
                                motif['node'][i] = selected_motif
                                increased_depth = True
                    # go another layer down if it's not a base node yet
                    else:
                        sub_motif = self.motif_of_motif(node, config, max_depth, current_depth + 1)
                        # sub_motif_id = self.insert_motif(sub_motif)
                        # motif['node'][i] = sub_motif[sub_motif_id]
                        motif['node'][i] = self.insert_motif(sub_motif)
                        new_depth = sub_motif['depth']
                        # if the depth is bigger adjust it
                        if new_depth >= motif['depth']:
                            motif['depth'] = new_depth + 1
                            picked_bigger = True
                    i += 1
                # check to make sure the depth is correct
                if increased_depth and not picked_bigger:
                    motif['depth'] += 1
            else:
                # go to a certain depth
                None
        return motif

    '''generates a individual which is a motif of motifs up to a certain depth. motif_of_motif is recursively called 
    until the required depth is reached.'''
    def generate_individual(self, connfig=1, start_small=False, max_depth=2):
        # select depth of the agent
        if not start_small:  # this is super dysfunctional
            depth = np.random.randint(self.initial_hierarchy_depth, max_depth + 1)
        else:
            depth = self.initial_hierarchy_depth
        # select an initial motif
        motif = self.select_motif()
        for i in range(depth):
            motif = self.motif_of_motif(motif, connfig, depth, i)
            motif = self.insert_motif(motif)
        return [motif, np.random.randint(200)]

    def shift_io(self, in_or_out, motif_id, direction='random', shift='linear'):
        if direction == 'random':
            if shift == 'linear':
                if in_or_out == 'in':
                    direction = np.random.choice(range(self.inputs - 1)) + 1
                else:
                    direction = np.random.choice(range(self.outputs - 1)) + 1
        if motif_id in self.neurons.neuron_configs:
            return self.neurons.shift_io(motif_id, in_or_out, direction)
        motif = self.motif_configs[motif_id]
        motif_copy = deepcopy(motif)
        for i in range(len(motif['node'])):
            if motif_copy['node'][i] not in self.neurons.neuron_configs:
                motif_copy['node'][i] = self.shift_io(in_or_out, motif_copy['node'][i], direction, shift)
            else:
                motif_copy['node'][i] = self.neurons.shift_io(motif_copy['node'][i], in_or_out, direction)
        if motif_copy != motif:
            new_id = self.insert_motif(motif_copy)
        else:
            new_id = motif_id
        return new_id

    def list_neurons(self, list, motif_id, repeat=True):
        motif = self.motif_configs[motif_id]
        for node in motif['node']:
            if node in self.neurons.neuron_configs:
                if node not in list or repeat:
                    list.append(node)
            else:
                list = self.list_neurons(list, node)
        return list


    # def generate_agents(self, pop_size=200, connfig=1, start_small=False, max_depth=2):
    #     print "constructing population of agents"
    #     self.agent_pop = []
    #     for i in range(pop_size):
    #         # select depth of the agent
    #         if not start_small:  # this is broke af
    #             depth = np.random.randint(self.initial_hierarchy_depth, max_depth + 1)
    #         else:
    #             depth = self.initial_hierarchy_depth
    #         # motif = None
    #         motif = self.select_motif()
    #         if motif is None:
    #             motif = self.select_motif()
    #         else:
    #             for i in range(depth):
    #                 motif = self.motif_of_motif(motif, connfig, depth, i)
    #                 motif = self.insert_motif(motif)
    #         self.agent_pop.append((motif, np.random.randint(pop_size)))
    #     return self.agent_pop

    def collect_IO(self, node, prepost, upper, layer, node_array=[]):
        try:
            motif = self.motif_configs[str(node)]
        except:
            traceback.print_exc()
            print "mate I dunno"
        node_count = 0
        for io in motif['io']:
            local_upper = deepcopy(upper)
            if prepost == 'pre' and io[1]:
                pre_node = motif['node'][node_count]
                if pre_node in self.neurons.neuron_configs:
                    node_array.append([pre_node, node_count, local_upper, layer])
                else:
                    local_upper.append(node_count)
                    self.collect_IO(pre_node, prepost, local_upper, layer + 1, node_array)
            if prepost == 'post' and io[0]:
                post_node = motif['node'][node_count]
                if post_node in self.neurons.neuron_configs:
                    node_array.append([post_node, node_count, local_upper, layer])
                else:
                    local_upper.append(node_count)
                    self.collect_IO(post_node, prepost, local_upper, layer + 1, node_array)
            node_count += 1
        return node_array

    def connect_nodes(self, pre_node, pre_count, post_node, post_count, layer, upper, weight, delay, plasticity):
        pre_ids = []
        post_ids = []
        connections = []
        if pre_node in self.neurons.neuron_configs:
            pre_ids.append([pre_node, pre_count, upper, layer])
        else:
            new_pre_count = upper + [pre_count]
            self.collect_IO(pre_node, 'pre', new_pre_count, layer + 1, pre_ids)
        if post_node in self.neurons.neuron_configs:
            post_ids.append([post_node, post_count, upper, layer])
        else:
            new_post_count = upper + [post_count]
            self.collect_IO(post_node, 'post', new_post_count, layer + 1, post_ids)
        for pre in pre_ids:
            for post in post_ids:
                connections.append([pre, post, weight, delay, plasticity])
        return connections

    def read_motif(self, motif_id, layer=0, upper=[]):
        motif = self.motif_configs[motif_id]
        all_connections = []
        for conn in motif['conn']:
            pre = conn[0]
            post = conn[1]
            weight = conn[2]
            delay = conn[3]
            plasticity = conn[4]
            pre_node = motif['node'][pre]
            post_node = motif['node'][post]
            all_connections += self.connect_nodes(pre_node, pre, post_node, post, layer,
                                                  upper, weight, delay, plasticity)
        node_count = 0
        for node in motif['node']:
            if node not in self.neurons.neuron_configs:
                local_upper = deepcopy(upper)
                local_upper.append(node_count)
                all_connections += self.read_motif(node, layer + 1, local_upper)
            node_count += 1
        return all_connections

    def construct_io(self, agent_connections, inputs, outputs):
        indexed_ex = []
        indexed_in = []
        e2e = []
        e2i = []
        i2e = []
        i2i = []
        in2e = []
        in2i = []
        in2in = []
        in2out = []
        e2out = []
        i2out = []
        e2in = []
        i2in = []
        out2e = []
        out2i = []
        out2in = []
        out2out = []
        for conn in agent_connections:
            out_pre = -1
            in_pre = -1
            out_post = -1
            in_post = -1
            if self.neurons.neuron_configs[conn[0][0]]['type'] == 'excitatory':
                pre_ex = True
                pre_in = False
                try:
                    pre_index = indexed_ex.index(conn[0])
                except:
                    indexed_ex.append(conn[0])
                    pre_index = indexed_ex.index(conn[0])
            elif self.neurons.neuron_configs[conn[0][0]]['type'] == 'inhibitory':
                pre_ex = False
                pre_in = True
                try:
                    pre_index = indexed_in.index(conn[0])
                except:
                    indexed_in.append(conn[0])
                    pre_index = indexed_in.index(conn[0])
            else:
                pre_ex = False
                pre_in = False
                if self.neurons.neuron_configs[conn[0][0]]['type'] == 'output':
                    out_pre = self.neurons.neuron_configs[conn[0][0]]['io']
                    pre_index = out_pre
                else:
                    in_pre = self.neurons.neuron_configs[conn[0][0]]['io']
                    pre_index = in_pre
            if self.neurons.neuron_configs[conn[1][0]]['type'] == 'excitatory':
                post_ex = True
                post_in = False
                try:
                    post_index = indexed_ex.index(conn[1])
                except:
                    indexed_ex.append(conn[1])
                    post_index = indexed_ex.index(conn[1])
            elif self.neurons.neuron_configs[conn[1][0]]['type'] == 'inhibitory':
                post_ex = False
                post_in = True
                try:
                    post_index = indexed_in.index(conn[1])
                except:
                    indexed_in.append(conn[1])
                    post_index = indexed_in.index(conn[1])
            else:
                post_ex = False
                post_in = False
                if self.neurons.neuron_configs[conn[1][0]]['type'] == 'output':
                    out_post = self.neurons.neuron_configs[conn[1][0]]['io']
                    post_index = out_post
                else:
                    in_post = self.neurons.neuron_configs[conn[1][0]]['io']
                    post_index = in_post
            if self.constant_delays:
                conn[4] = self.constant_delays
            if pre_ex and post_ex:
                e2e.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif pre_ex and post_in:
                e2i.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif pre_in and post_ex:
                i2e.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif pre_in and post_in:
                i2i.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif pre_ex and in_post >= 0:
                e2in.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif pre_ex and out_post >= 0:
                e2out.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif pre_in and in_post >= 0:
                i2in.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif pre_in and out_post >= 0:
                i2out.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif in_pre >= 0 and post_ex:
                in2e.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif in_pre >= 0 and post_in:
                in2i.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif in_pre >= 0 and in_post >= 0:
                in2in.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif in_pre >= 0 and out_post >= 0:
                in2out.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif out_pre >= 0 and post_ex:
                out2e.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif out_pre >= 0 and post_in:
                out2i.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif out_pre >= 0 and in_post >= 0:
                out2in.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            elif out_pre >= 0 and out_post >= 0:
                out2out.append((pre_index, post_index, conn[2], conn[3], conn[4]))
            else:
                print "somethin fucky"
        excite_params = {}
        for excite_neuron in indexed_ex:
            neuron = excite_neuron[0]
            neuron = self.neurons.neuron_configs[neuron]
            for param in neuron['params']:
                try:
                    excite_params[param]
                except:
                    excite_params[param] = []
                excite_params[param].append(neuron['params'][param])
        inhib_params = {}
        for inhib_neuron in indexed_in:
            neuron = inhib_neuron[0]
            neuron = self.neurons.neuron_configs[neuron]
            for param in neuron['params']:
                try:
                    inhib_params[param]
                except:
                    inhib_params[param] = []
                inhib_params[param].append(neuron['params'][param])
        return in2e, in2i, in2in, in2out, e2in, i2in, len(indexed_ex), e2e, e2i, len(indexed_in), \
               i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out, excite_params, inhib_params

    def convert_individual(self, agent, inputs, outputs):
        if isinstance(agent, list):
            agent_conn = self.read_motif(agent[0])
        else:
            agent_conn = self.read_motif(agent)
        spinn_conn = self.construct_io(agent_conn, inputs, outputs)
        return spinn_conn


    def list_motifs(self, motif_id, list=[]):
        list.append(motif_id)
        if isinstance(motif_id, int):
            motif_id = '{}'.format(motif_id)
        motif = self.motif_configs[motif_id]
        for node in motif['node']:
            if node not in self.neurons.neuron_configs:
                # if node in list:
                #     list = False
                #     break
                # local_list = deepcopy(list)
                self.list_motifs(node, list)
        return list

    def recurse_check(self, motif_id, list=[]):
        check = True
        list.append(motif_id)
        if isinstance(motif_id, int):
            motif_id = '{}'.format(motif_id)
        motif = self.motif_configs[motif_id]
        for node in motif['node']:
            if node not in self.neurons.neuron_configs:
                if node in list:
                    return False
                if not self.recurse_check(node, list):
                    return False
        del list[list.index(motif_id)]
        return check

    def return_motif(self, motif_id):
        return self.motif_configs[motif_id]

    def reset_weights(self, full_reset=True):
        if full_reset == True:
            self.total_weight = 0
            for motif_id in self.motif_configs:
                self.motif_configs['{}'.format(motif_id)]['weight'] = 0
            self.neurons.reset_weights()

    def update_weight(self, motif_ids, weight):
        for motif_id in motif_ids:
            self.motif_configs['{}'.format(motif_id)]['weight'] += weight

    def delete_motif(self, motif_id):
        del self.motif_configs['{}'.format(motif_id)]

    def depth_read(self, motif_id, best_depth=0, current_depth=1):
        if current_depth > best_depth:
            best_depth = current_depth
        motif = self.motif_configs[motif_id]['node']
        for node in self.motif_configs[motif_id]['node']:
            if node not in self.neurons.neuron_configs:
                best_depth = self.depth_read(node, best_depth, current_depth+1)
        return best_depth

    def depth_fix(self):
        for motif_id in self.motif_configs:
            self.motif_configs[motif_id]['depth'] = self.depth_read(motif_id)

    def shape_check(self, motif_id):
        base_motif = True
        motif = self.motif_configs[motif_id]
        motif_id = False
        motif_array = self.permutations(motif)
        for config in self.motif_configs:
            if int(config) != motif['id'] and len(self.motif_configs[config]['node']) == len(motif['node']) and \
                    len(self.motif_configs[config]['conn']) == len(motif['conn']):
                for node in self.motif_configs[config]['node']:
                    if node not in self.neurons.neuron_configs:
                        base_motif = False
                        break
                    else:
                        base_motif = True
                if base_motif:
                    for isomer in motif_array:
                        # compare each element and return rotations with which they are similar
                        if self.motif_configs[config]['io'] == isomer['io'] and \
                                        self.motif_configs[config]['conn'] == isomer['conn']:
                            motif_id = config
                            break
                if motif_id:
                    break
        return motif_id

    def reward_shape(self):
        for motif_id in self.motif_configs:
            motif = self.motif_configs[motif_id]
            if self.depth_read(motif_id) > 1:
                id = self.shape_check(motif_id)
                if id:
                    self.motif_configs[id]['weight'] += self.motif_configs[motif_id]['weight']


    def clean_population(self, reward_shape):
        if reward_shape:
            self.reward_shape()
        to_be_deleted = []
        for motif_id in self.motif_configs:
            if self.motif_configs['{}'.format(motif_id)]['weight'] == 0:
                to_be_deleted.append(motif_id)
        for id in to_be_deleted:
            self.delete_motif(id)
        self.depth_fix()

    def save_motifs(self, iteration, config):
        with open('Motif population {}: {}.csv'.format(iteration, config), 'w') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for motif_id in self.motif_configs:
                motif = self.motif_configs[motif_id]
                for atribute in motif:
                    line = []
                    line.append(atribute)
                    if isinstance(motif[atribute], list):
                        for var in motif[atribute]:
                            line.append(var)
                    else:
                        line.append(motif[atribute])
                    writer.writerow(line)
            file.close()

    def average_weights(self, motif_counts):
        for motif_id in motif_counts:
            # self.motif_configs['{}'.format(motif_id)]['weight'] += weight
            self.motif_configs[motif_id]['weight'] /= motif_counts[motif_id]

    def adjust_weights(self, agents, clean=True, fitness_shaping=True, reward_shape=True, average=True, iteration=0):
        self.reset_weights()
        motif_count = {}
        neuron_count = {}
        if fitness_shaping:
            agents.sort(key=lambda x: x[2])#, reverse=True)
            i = 1
            for agent in agents:
                component_motifs = []
                component_motifs = self.list_motifs(agent[0], component_motifs)
                component_neurons = []
                component_neurons = self.list_neurons(component_neurons, agent[0])
                self.update_weight(component_motifs, agent[2])
                self.neurons.update_weights(component_neurons, agent[2])
                if average:
                    for component in component_motifs:
                        if component in motif_count:
                            motif_count[component] += 1
                        else:
                            motif_count[component] = 1
                    for component in component_neurons:
                        if component in neuron_count:
                            neuron_count[component] += 1
                        else:
                            neuron_count[component] = 1
                i += 1
        if average:
            self.average_weights(motif_count)
            self.neurons.average_weights(neuron_count)
        if clean:
            self.clean_population(reward_shape)
            self.neurons.clean_population()
        if iteration < self.keep_reading and self.read_entire_population:
            self.read_population()
        self.total_weight = 0


