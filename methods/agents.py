import sys, os
import time
import socket
import numpy as np
import math
import itertools
from copy import deepcopy
# import operator
# from spinn_front_end_common.utilities.globals_variables import get_simulator
# import traceback
import math
from methods.networks import motif_population
import traceback
import csv
import networkx as nx
# import threading
# import pathos.multiprocessing
# from spinn_front_end_common.utilities import globals_variables
# from ast import literal_eval

# max_fail_score = 0

class agent_population(object):
    def __init__(self,
                 motif,
                 conn_weight=0.5,
                 # motif_weight=0.5,
                 crossover=0.5,
                 elitism=0.1,
                 viable_parents=0.1,
                 strict_io=True,
                 force_i2o=True,
                 sexuality=[1./4., 1./4., 1./4., 1./4.],  # asexual (just mutate), sexual, sexual + mutate, fresh motif construction
                 conn_param_mutate=0.1,
                 param_mutate_stdev=0.15,
                 base_mutate=0,
                 conn_add=0.015,
                 conn_gone=0.015,
                 io_mutate=0.015,
                 input_shift=0.015,
                 output_shift=0.015,
                 node_mutate=0.015,
                 node_switch=0.015,
                 motif_add=0.015,
                 motif_gone=0.015,
                 motif_switch=0.015,
                 new_motif=0.015,
                 switch_plasticity=0.015,
                 multiple_mutates=True,
                 maximum_depth=5,
                 similarity_threshold=0.4,
                 stagnation_age=25,
                 inputs=2,
                 outputs=2,
                 pop_size=100):

        self.motifs = motif
        self.pop_size = pop_size
        self.conn_weight = conn_weight
        self.motif_weight = 1 - conn_weight
        self.crossover = crossover
        self.elitism = elitism
        self.viable_parents = viable_parents
        self.strict_io = strict_io
        self.force_i2o = force_i2o
        # self.asexual = asexual
        self.sexuality = {'asexual': sexuality[0], 'sexual': sexuality[1], 'both': sexuality[2], 'fresh': sexuality[3]}
        self.conn_param_mutate = conn_param_mutate
        self.param_mutate_stdev = param_mutate_stdev
        if base_mutate:
            self.conn_add = base_mutate
        else:
            self.conn_add = conn_add
        if base_mutate:
            self.conn_gone = base_mutate
        else:
            self.conn_gone = conn_gone
        if base_mutate:
            self.io_mutate = base_mutate
        else:
            self.io_mutate = io_mutate
        if base_mutate:
            self.input_shift = base_mutate
        else:
            self.input_shift = input_shift
        if base_mutate:
            self.output_shift = base_mutate
        else:
            self.output_shift = output_shift
        if base_mutate:
            self.node_mutate = base_mutate
        else:
            self.node_mutate = node_mutate
        if base_mutate:
            self.node_switch = base_mutate
        else:
            self.node_switch = node_switch
        if base_mutate:
            self.motif_add = base_mutate
        else:
            self.motif_add = motif_add
        if base_mutate:
            self.motif_gone = base_mutate
        else:
            self.motif_gone = motif_gone
        if base_mutate:
            self.motif_switch = base_mutate
        else:
            self.motif_switch = motif_switch
        if base_mutate:
            self.new_motif = base_mutate
        else:
            self.new_motif = new_motif
        if base_mutate:
            self.switch_plasticity = base_mutate
        else:
            self.switch_plasticity = switch_plasticity
        self.multiple_mutates = multiple_mutates
        if isinstance(maximum_depth, list):
            self.maximum_depth = maximum_depth[0]
        else:
            self.maximum_depth = maximum_depth
        self.similarity_threshold = similarity_threshold
        self.stagnation_age = stagnation_age
        self.inputs = inputs
        self.outputs = outputs

        self.species = []
        self.agent_pop = []
        self.agent_mutate_keys = {}

        self.max_fitness = []
        self.average_fitness = []
        self.min_fitness = []
        self.max_score = []
        self.average_score = []
        self.min_score = []
        self.total_average = []

        self.min_hidden_neurons = []
        self.average_hidden_neurons = []
        self.max_hidden_neurons = []
        self.weighted_hidden_score = []
        self.weighted_hidden_fitness = []
        self.best_score_hidden = []
        self.best_fitness_hidden = []
        self.min_io_neurons = []
        self.average_io_neurons = []
        self.max_io_neurons = []
        self.weighted_io_score = []
        self.weighted_io_fitness = []
        self.best_score_io = []
        self.best_fitness_io = []
        self.min_connections = []
        self.average_connections = []
        self.max_connections = []
        self.weighted_conn_score = []
        self.weighted_conn_fitness = []
        self.best_score_conn = []
        self.best_fitness_conn = []
        self.min_pl_ratio = []
        self.average_pl_ratio = []
        self.max_pl_ratio = []
        self.weighted_pl_ratio_score = []
        self.weighted_pl_ratio_fitness = []
        self.best_score_pl_ratio = []
        self.best_fitness_pl_ratio = []
        self.min_m_depth = []
        self.average_m_depth = []
        self.max_m_depth = []
        self.weighted_m_depth_score = []
        self.weighted_m_depth_fitness = []
        self.best_score_m_depth = []
        self.best_fitness_m_depth = []

    '''sets the maximum depth allowed by the agents to progressively change depth as the algorithm progresses'''
    def set_max_d(self, depth, iteration, max_iterations):
        if isinstance(depth, list):
            depth_range = depth[1] - depth[0]
            depth_width = max_iterations / float(depth_range + 1)
            self.maximum_depth = depth[0] + int(iteration / depth_width)
        else:
            self.maximum_depth = depth

    '''creates the spinnaker fromlist connections either from a list of agents or will generate agents if required'''
    def generate_spinn_nets(self, input=None, output=None, create=True, max_depth=2):
        if input is None:
            input = self.inputs
        if output is None:
            output = self.outputs
        agent_connections = []
        spinn_conns = []
        if create:
            if create == 'reset':
                self.agent_pop = []
            self.generate_population(max_depth)
        for agent in self.agent_pop:
            agent_setup = self.convert_agent_tf(agent)
            agent_connections.append(agent_setup)

        return agent_connections

    '''creates the population'''
    def generate_population(self, max_depth):
        for i in range(self.pop_size):
            self.agent_pop.append(self.new_individual(max_depth))
            print("created agent ", i + 1, "of", self.pop_size)

    '''creates a new individual'''
    def new_individual(self, max_depth):
        agent = False
        while agent == False:
            agent = self.motifs.generate_individual(max_depth=max_depth)
            agent = self.valid_net(agent)
        return agent

    '''converts an agent into a list of connections'''
    def convert_agent(self, agent):
        agent_setup = self.motifs.convert_individual(agent)
        return agent_setup

    '''convert agent into connection matrix'''
    def convert_agent_tf(self, agent):
        conn_matrix, delay_matrix, indexed_i, indexed_o, neuron_params = self.motifs.convert_individual(agent)

        return conn_matrix, delay_matrix, indexed_i, indexed_o, neuron_params

    def convert_connections_to_matrix(self, conn_list, in_size, out_size, inhibitory=False,
                                      in_offset=0, out_offset=0, delaying=False):
        connections = np.zeros([in_size, out_size])
        delays = np.zeros([in_size, out_size])
        conn_count = np.ones([in_size, out_size])
        for conn in conn_list:
            conn_count[conn[0] + in_offset][conn[1] + out_offset] += 1
            if inhibitory:
                connections[conn[0] + in_offset][conn[1] + out_offset] += conn[2] * -1.
            else:
                connections[conn[0] + in_offset][conn[1] + out_offset] += conn[2]
            # connections[conn[0]+in_offset][conn[1]+out_offset] += conn[2]
            delays[conn[0]+in_offset][conn[1]+out_offset] += conn[3]

        for i in range(len(conn_count)):
            for j in range(len(conn_count[0])):
                connections[i][j] /= conn_count[i][j]
                delays[i][j] /= conn_count[i][j]
                delays[i][j] = int(round(delays[i][j]))
        if delaying:
            return delays
        else:
            return connections

    '''shapes the fitness generated by a population to make them relative to each other not absolute scores'''
    def fitness_shape(self, fitnesses, max_fail_score, spike_weighting):
        # if multiple fitness scores are passed back index and shape them all
        if isinstance(fitnesses[0], list):
            shaped_fitnesses = [0 for i in range(len(fitnesses[0]))]
            indexed_fitness = []
            # labels the fitness with the agent id
            for i in range(len(fitnesses)):
                new_indexes = []
                for j in range(len(fitnesses[i])):
                    new_indexes.append([fitnesses[i][j], j])
                new_indexes.sort()
                indexed_fitness.append(new_indexes)
            # ranks the fitnesses relative to each other
            for metric, weight in zip(indexed_fitness, spike_weighting):
                current_shape = 0
                for i in range(len(metric)):
                    # if it comepletely fails don't allow it a fitness removing the option of passing on any genes
                    if metric[i][0] == max_fail_score or metric[i][0] == 'fail':
                        shaped_fitnesses[metric[i][1]] += 0
                    else:
                        if i > 0:
                            # if the fitness is the same as the previous agent give it the same score otherwise increase
                            # the increment to the current rank
                            if metric[i][0] != metric[i-1][0]:
                                current_shape = i
                        shaped_fitnesses[metric[i][1]] += current_shape * weight
                    if i > len(metric) - (len(metric) * self.viable_parents):
                        shaped_fitnesses[metric[i][1]] += 0.00001  # just to ensure elite aren't erased accidentally
        else:
            # the same as above but with only one fitness metric
            shaped_fitnesses = [0 for i in range(len(fitnesses))]
            new_indexes = []
            for i in range(len(fitnesses)):
                new_indexes.append([fitnesses[i], i])
            new_indexes.sort()
            current_shape = 0
            for i in range(len(fitnesses)):
                if new_indexes[i][0] == max_fail_score or new_indexes[i][0] == 'fail':
                    shaped_fitnesses[new_indexes[i][1]] += 0
                else:
                    if new_indexes[i][0] != new_indexes[i-1][0]:
                        current_shape = i
                    shaped_fitnesses[new_indexes[i][1]] += current_shape
                    if i > len(new_indexes) - (len(new_indexes) * self.viable_parents):
                        shaped_fitnesses[new_indexes[i][1]] += 0.00001  # just to ensure elite aren't erased accidentally
        return shaped_fitnesses

    '''either shape the fitnesses or not, then pass then pass the score to the agents for later processing'''
    def pass_fitnesses(self, fitnesses, max_fail_score, spike_weighting, fitness_shaping=True):
        if fitness_shaping:
            processed_fitnesses = self.fitness_shape(fitnesses, max_fail_score, spike_weighting)
        else:
            # todo figure out what to do about fitness less than 0
            # if isinstance(fitnesses[0], list):
            #     processed_fitnesses = []
            #     for i in range(len(fitnesses[0])):
            #         summed_f = 0
            #         for j in range(len(fitnesses)):
            #             if fitnesses[j][i] == 'fail':
            #                 summed_f = max_fail_score
            #                 break
            #             summed_f += fitnesses[j][i]
            #         processed_fitnesses.append(summed_f)
            processed_fitnesses = fitnesses

        for i in range(len(self.agent_pop)):
            self.agent_pop[i].append(processed_fitnesses[i])

    '''reset species, an unused function'''
    def reset(self):
        for specie in self.species:
            specie.has_best = False

    '''calculates the similarity between two motifs'''
    def similarity(self, a, b):
        if isinstance(a, list):
            a = a[0]
        if isinstance(b, list):
            b = b[0]
        a_list = []
        self.motifs.list_motifs(a, a_list)
        a_conn = self.motifs.read_motif(a)
        b_list = []
        self.motifs.list_motifs(b, b_list)
        b_conn = self.motifs.read_motif(b)

        list_similarity = [0, 0]
        copy_a = deepcopy(a_list)
        for motif in b_list:
            if motif in copy_a:
                list_similarity[0] += 1
                del b_list[b_list.index(motif)]
            list_similarity[1] += 1
        for motif in copy_a:
            list_similarity[1] += 1
        list_similarity = list_similarity[0] / list_similarity[1]

        conn_similarity = [0, 0]
        copy_a = deepcopy(a_conn)
        for conn in b_conn:
            if conn in copy_a:
                conn_similarity[0] += 1
                del b_conn[b_conn.index(conn)]
            conn_similarity[1] += 1
        for conn in copy_a:
            conn_similarity[1] += 1

        conn_similarity = conn_similarity[0] / conn_similarity[1]

        similarity = (self.conn_weight * conn_similarity) + (self.motif_weight * list_similarity)
        return 1 - similarity

    '''forms the species and quantifies them and their separation from each other'''
    def iterate_species(self):
        self.reset()
        print("evolve them here")
        for agent in self.agent_pop:
            belongs = False
            for specie in self.species:
                if self.similarity(specie.representative, agent) < self.similarity_threshold:
                    specie.members.append(agent)
                    belongs = True
            if not belongs:
                self.species.append(agent_species(agent))

        highest_fitness = None
        for specie in self.species:
            specie.calc_metrics()
            if highest_fitness is not None:
                if specie.max_fitness > highest_fitness:
                    highest_fitness = specie.max_fitness
                    best_specie = specie
            else:
                highest_fitness = specie.max_fitness
                best_specie = specie
        self.species[self.species.index(best_specie)].has_best = True

        # remove old species unless they're the best
        self.species = [s for s in self.species if s.no_improvement_age < self.stagnation_age or s.has_best]

        print("species are formed and quantified, now to add young and old age modifiers to quantify the amount of offspring generated")

    '''performs one step of evolution'''
    def evolve(self, species=True):
        if species:
            self.iterate_species()
        else:
            self.agent_pop, self.agent_mutate_keys = self.generate_children(self.agent_pop, len(self.agent_pop))
            print("")

    '''Takes in a parent motif and mutates it in various ways, all sub-motifs are added with the final motif structure/
    id being returned. A key is kept of each mutation used to generate a child for later analysis of appropriate 
    mutation operators'''
    def mutate(self, parent, mutate_key={}):
        # initialise mutation key
        if mutate_key == {}:
            mutate_key['motif'] = 0
            mutate_key['new'] = 0
            mutate_key['node_s'] = 0
            mutate_key['node_g'] = 0
            mutate_key['io'] = 0
            mutate_key['in_shift'] = 0
            mutate_key['out_shift'] = 0
            mutate_key['m_add'] = 0
            mutate_key['m_gone'] = 0
            mutate_key['c_add'] = 0
            mutate_key['c_gone'] = 0
            mutate_key['param_w'] = 0
            mutate_key['param_d'] = 0
            mutate_key['plasticity'] = 0
            mutate_key['sex'] = 0
        # acquire the motif of parent and copy it to avoid messing with both memory locations
        if isinstance(parent, list):
            motif_config = self.motifs.return_motif(parent[0])
            if len(parent) > 3 and mutate_key == {}:
                mutate_key['mum'] = [parent[2], parent[3]]
                mutate_key['dad'] = [parent[2], parent[3]]
            # elif len(parent) > 2:
            #     mutate_key['mum'] = [parent[2], parent[3]]
            #     mutate_key['dad'] = [parent[2], parent[3]]
        else:
            motif_config = self.motifs.return_motif(parent)
        config_copy = deepcopy(motif_config)
        # add a node to the motif
        if np.random.random() < self.motif_add and len(config_copy['node']) <= self.motifs.max_motif_size:
            config_copy = self.motifs.add_motif(config_copy)
            mutate_key['m_add'] += 1
        # remove a node from the motif
        if np.random.random() < self.motif_gone \
                and len(config_copy['node']) >= self.motifs.min_motif_size and len(config_copy['node']) > 1:
            config_copy = self.motifs.remove_motif(config_copy)
            mutate_key['m_gone'] += 1
        motif_size = len(config_copy['node'])
        # loop through each node and randomly mutate
        for i in range(motif_size):
            prob_resize_factor = 1
            # switch with a randomly selected motif
            if np.random.random() * prob_resize_factor < self.motif_switch:
                selected = False
                while not selected:
                    selected_motif = self.motifs.select_motif()
                    selected = self.motifs.recurse_check(selected_motif, [])
                config_copy['node'][i] = selected_motif
                new_depth = self.motifs.motif_configs[config_copy['node'][i]]['depth']
                if new_depth >= config_copy['depth']:
                    config_copy['depth'] = new_depth + 1
                mutate_key['motif'] += 1
                if not self.multiple_mutates:
                    continue
            elif not self.multiple_mutates:
                prob_resize_factor *= 1 - self.motif_switch
            # switch with a completely novel motif todo maybe add or make this a motif of motifs of w/e depth
            if np.random.random() * prob_resize_factor < self.new_motif:
                config_copy['node'][i] = self.motifs.generate_motif(weight=0)
                new_depth = self.motifs.motif_configs[config_copy['node'][i]]['depth']
                if new_depth >= config_copy['depth']:
                    config_copy['depth'] = new_depth + 1
                mutate_key['new'] += 1
                continue
            else:
                prob_resize_factor *= 1 - self.new_motif
            # change the IO configurations
            if np.random.random() * prob_resize_factor < self.io_mutate:
                old_io = config_copy['io'][i]
                while config_copy['io'][i] == old_io:
                    new_io = (np.random.choice((True, False)), np.random.choice((True, False)))
                    config_copy['io'][i] = new_io
                mutate_key['io'] += 1
                if not self.multiple_mutates:
                    continue
            elif not self.multiple_mutates:
                prob_resize_factor *= 1 - self.io_mutate
            # mutate the base node if it's a base node
            if np.random.random() * prob_resize_factor < self.node_mutate and \
                    config_copy['node'][i] in self.motifs.neurons.neuron_configs:
                mutate_key['node_g'] += 1
                new_node = config_copy['node'][i]
                while config_copy['node'][i] == new_node:
                    new_node = self.motifs.neurons.generate_neuron()
                config_copy['node'][i] = new_node
                if not self.multiple_mutates:
                    continue
            elif config_copy['node'][i] in self.motifs.neurons.neuron_configs and not self.multiple_mutates:
                prob_resize_factor *= 1 - self.node_mutate
            # switch the base node if it's a base node
            if np.random.random() * prob_resize_factor < self.node_switch and \
                    config_copy['node'][i] in self.motifs.neurons.neuron_configs:
                mutate_key['node_s'] += 1
                new_node = config_copy['node'][i]
                while config_copy['node'][i] == new_node:
                    new_node = self.motifs.neurons.choose_neuron()
                config_copy['node'][i] = new_node
                if not self.multiple_mutates:
                    continue
            elif config_copy['node'][i] in self.motifs.neurons.neuron_configs and not self.multiple_mutates:
                prob_resize_factor *= 1 - self.node_mutate
            # shift all input nodes of the motif by the same amount
            if np.random.random() * prob_resize_factor < self.input_shift:
                new_node = self.motifs.shift_io('in', config_copy['node'][i])
                if new_node != config_copy['node'][i]:
                    config_copy['node'][i] = new_node
                    mutate_key['in_shift'] += 1
                if not self.multiple_mutates:
                    continue
            elif not self.multiple_mutates:
                prob_resize_factor *= 1 - self.input_shift
            # shift all output nodes of the motif by the same amount
            if np.random.random() * prob_resize_factor < self.output_shift:
                new_node = self.motifs.shift_io('out', config_copy['node'][i])
                if new_node != config_copy['node'][i]:
                    config_copy['node'][i] = new_node
                    mutate_key['out_shift'] += 1
        # delete a connection
        if np.random.random() < self.conn_gone and len(config_copy['conn']) > 0:
            del config_copy['conn'][np.random.randint(len(config_copy['conn']))]
            mutate_key['c_gone'] += 1
        # add a connection
        if np.random.random() < self.conn_add:
            new_conn = False
            while not new_conn:
                conn = []
                conn.append(np.random.randint(motif_size))
                conn.append(np.random.randint(motif_size))
                bin = np.random.randint(0, self.motifs.no_weight_bins)
                conn.append(self.motifs.weight_range[0] + (bin * self.motifs.weight_bin_width))
                bin = np.random.randint(0, self.motifs.no_delay_bins)
                conn.append(self.motifs.delay_range[0] + (bin * self.motifs.delay_bin_width))
                if self.motifs.plasticity and self.motifs.structural:
                    choice = np.random.random()
                    if choice < 1.0/3.0:
                        conn.append('stdp')
                    elif choice < 2.0/3.0:
                        conn.append('non-plastic')
                    else:
                        conn.append('structural')
                if np.random.random() < 0.5 and self.motifs.plasticity and not self.motifs.structural:
                    conn.append('stdp')
                else:
                    conn.append('non-plastic')
                if np.random.random() < 0.5 and self.motifs.structural and not self.motifs.plasticity:
                    conn.append('structural')
                else:
                    conn.append('non-plastic')
                if conn[2]:
                    new_conn = True
            config_copy['conn'].append(conn)
            mutate_key['c_add'] += 1
        # mutate the weight or delay of a connection
        for i in range(len(config_copy['conn'])):
            # weight
            if np.random.random() < self.conn_param_mutate:
                old_weight = config_copy['conn'][i][2]
                while old_weight == config_copy['conn'][i][2]:
                    change = np.random.normal(0, self.param_mutate_stdev)
                    change *= (self.motifs.weight_range[1] - self.motifs.weight_range[0])
                    bin_change = int(round(change / self.motifs.weight_bin_width))
                    bin = bin_change + ((old_weight - self.motifs.weight_range[0]) / self.motifs.weight_bin_width)
                    bin %= (self.motifs.no_weight_bins - 1)
                    new_weight = self.motifs.weight_range[0] + (bin * self.motifs.weight_bin_width)
                    config_copy['conn'][i][2] = new_weight
                mutate_key['param_w'] += 1
            # delay
            if np.random.random() < self.conn_param_mutate:
                old_delay = config_copy['conn'][i][3]
                while old_delay == config_copy['conn'][i][3]:
                    change = np.random.normal(0, self.param_mutate_stdev)
                    change *= (self.motifs.delay_range[1] - self.motifs.delay_range[0])
                    bin_change = int(round(change / self.motifs.delay_bin_width))
                    bin = bin_change + ((old_delay - self.motifs.delay_range[0]) / self.motifs.delay_bin_width)
                    bin %= (self.motifs.no_delay_bins - 1)
                    new_delay = self.motifs.delay_range[0] + (bin * self.motifs.delay_bin_width)
                    config_copy['conn'][i][3] = new_delay
                mutate_key['param_d'] += 1
            # plasticity
            if np.random.random() < self.switch_plasticity:
                if self.motifs.plasticity and not self.motifs.structural:
                    if config_copy['conn'][i][4] == 'stdp':
                        config_copy['conn'][i][4] = 'non-plastic'
                    else:
                        config_copy['conn'][i][4] = 'stdp'
                elif not self.motifs.plasticity and self.motifs.structural:
                    if config_copy['conn'][i][4] == 'structural':
                        config_copy['conn'][i][4] = 'non-plastic'
                    else:
                        config_copy['conn'][i][4] = 'structural'
                elif self.motifs.plasticity and self.motifs.structural:
                    if config_copy['conn'][i][4] == 'structural':
                        if np.random.random() < 0.5:
                            config_copy['conn'][i][4] = 'non-plastic'
                        else:
                            config_copy['conn'][i][4] = 'stdp'
                    elif config_copy['conn'][i][4] == 'stdp':
                        if np.random.random() < 0.5:
                            config_copy['conn'][i][4] = 'non-plastic'
                        else:
                            config_copy['conn'][i][4] = 'structural'
                    else:
                        if np.random.random() < 0.5:
                            config_copy['conn'][i][4] = 'stdp'
                        else:
                            config_copy['conn'][i][4] = 'structural'
                mutate_key['plasticity'] += 1
        # insert the new motif and then go through the nodes and mutate them
        motif_id = self.motifs.insert_motif(config_copy)
        if self.motifs.recurse_check(motif_id, []):
            copy_copy = deepcopy(config_copy)
        else:
            self.motifs.delete_motif(motif_id)
            print("That just tried to recurse")
            copy_copy = deepcopy(motif_config)
        node_count = 0
        for node in config_copy['node']:
            if node not in self.motifs.neurons.neuron_configs:
                try:
                    copy_copy['node'][node_count] = self.mutate([node], mutate_key)
                except RuntimeError as e:
                    traceback.print_exc()
                    print(mutate_key)
                    if e.args[0] != 'maximum recursion depth exceeded':
                        raise
                    else:
                        print("\nTried to mutate too many times\n")
                        return node
                except:
                    traceback.print_exc()
                    print(mutate_key)
                    print("\nNot an RTE\n")
                    raise
            node_count += 1
        if copy_copy != config_copy:
            motif_id = self.motifs.insert_motif(copy_copy)
        return motif_id

    '''mates 2 agents by iteration through the mother and probabilistically replacing a motif with a motif randomly
    selected from a list of dad's motifs'''
    def mate(self, mum, dad, mutate_key):
        # maybe the crossover should be more than just random, incorporating depth or some other dad decision metric
        # maybe take the seed randomly from mum or dad?
        # normally distributed around similar depth as a decision metric
        # swap with comparable motif in terms of IO on node or count of sensory motor neurons
        if mutate_key == {}:
            mutate_key['motif'] = 0
            mutate_key['new'] = 0
            mutate_key['node_s'] = 0
            mutate_key['node_g'] = 0
            mutate_key['io'] = 0
            mutate_key['in_shift'] = 0
            mutate_key['out_shift'] = 0
            mutate_key['m_add'] = 0
            mutate_key['m_gone'] = 0
            mutate_key['c_add'] = 0
            mutate_key['c_gone'] = 0
            mutate_key['param_w'] = 0
            mutate_key['param_d'] = 0
            mutate_key['mum'] = [mum[2], mum[2]] # was [3] but removed as no secondary metric as of this re-write
            mutate_key['dad'] = [dad[2], dad[2]]
            mutate_key['plasticity'] = 0
            mutate_key['sex'] = 1
        child_id = mum[0]
        mum_motif = deepcopy(self.motifs.motif_configs[mum[0]])
        dad_list = []
        dad_list = self.motifs.list_motifs(dad[0], dad_list)
        for i in range(len(mum_motif['node'])):
            if np.random.random() < self.crossover:
                mum_motif['node'][i] = np.random.choice(dad_list)
            elif mum_motif['node'][i] not in self.motifs.neurons.neuron_configs:
                mum_motif['node'][i] = self.mate([mum_motif['node'][i]], dad, mutate_key)
        if self.motifs.motif_configs[mum[0]] != mum_motif:
            child_id = self.motifs.insert_motif(mum_motif)
        return child_id

    '''creates a child from motifs of motifs'''
    def fresh_child(self, mutate_key):
        mutate_key['motif'] = 0
        mutate_key['new'] = 0
        mutate_key['node_s'] = 0
        mutate_key['node_g'] = 0
        mutate_key['io'] = 0
        mutate_key['in_shift'] = 0
        mutate_key['out_shift'] = 0
        mutate_key['m_add'] = 0
        mutate_key['m_gone'] = 0
        mutate_key['c_add'] = 0
        mutate_key['c_gone'] = 0
        mutate_key['param_w'] = 0
        mutate_key['param_d'] = 0
        mutate_key['mum'] = [self.average_fitness[len(self.average_fitness)-1], self.average_score[len(self.average_fitness)-1]]
        mutate_key['dad'] = [self.average_fitness[len(self.average_fitness)-1], self.average_score[len(self.average_fitness)-1]]
        mutate_key['plasticity'] = 0
        mutate_key['sex'] = 3
        child = self.motifs.select_motif()
        # starting_depth = child['depth']
        new_depth = np.random.randint(self.motifs.initial_hierarchy_depth, self.maximum_depth)
        while self.motifs.motif_configs[child]['depth'] < new_depth:
            child = self.motifs.motif_of_motif(child, 1, new_depth, 0)
        # for i in range(new_depth):
        #     child = self.motifs.motif_of_motif(child, 1, new_depth, i)
        #     # child = self.motifs.insert_motif(child)
        #     if child['depth'] >= new_depth:
        #         break
        return child

    # def check_connections_per_node(self, connections, max_synpase_count=255):
    #     [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size, i2e, i2i, e2out, i2out, out2e, out2i, out2in,
    #      out2out, excite_params, inhib_params] = connections
    #     connections = [in2e, in2i, in2out, e2e, e2i, i2e, i2i, e2out, i2out, out2e, out2i, out2in,
    #      out2out]
    #     in_pre = [in2e, in2i, in2out]
    #     e_pre = [e2e, e2i, e2out]
    #     i_pre = [i2e, i2i, i2out]
    #     out_pre = [out2e, out2i, out2out]
    #     pre_connections = [in_pre, e_pre, i_pre]
    #     in_post = [out2in]
    #     e_post = [in2e, e2e, i2e, out2e]
    #     i_post = [in2i, e2i, i2i, out2i]
    #     out_post = [in2out, e2out, i2out, out2out]
    #     post_connections = [e_post, i_post, out_post]
    #     max_pop_size = np.max([self.inputs, self.outputs, e_size, i_size])
    #     for pre_connection in pre_connections:
    #         index_count = [0 for i in range(max_pop_size)]
    #         for connection in pre_connection:
    #             for conn in connection:
    #                 index_count[conn[0]] += 1
    #         if np.max(index_count) > max_synpase_count:
    #             return False
    #     index_count = [0 for i in range(max_pop_size)]
    #     for connection in out_pre:
    #         for conn in connection:
    #             index_count[conn[0]] += 1
    #     if np.max(index_count) > max_synpase_count - self.outputs:
    #         return False
    #     for post_connection in post_connections:
    #         index_count = [0 for i in range(max_pop_size)]
    #         for connection in post_connection:
    #             for conn in connection:
    #                 index_count[conn[1]] += 1
    #         if np.max(index_count) > max_synpase_count:
    #             return False
    #     index_count = [0 for i in range(max_pop_size)]
    #     for conn in out2in:
    #         index_count[conn[1]] += 1
    #     if np.max(index_count) > max_synpase_count:
    #         return False
    #     return True
    
    def valid_net(self, child):
        [conn_matrix, delay_matrix, indexed_i, indexed_o, neuron_params] = self.convert_agent(child)

        if self.motifs.neurons.inputs + self.motifs.neurons.outputs == 0:
            if len(np.nonzero(np.array(conn_matrix))[0]) == 0:
                print("bad agent")
                return False
            else:
                return child
        if len(indexed_i) == 0:
            print("in bad agent")
            return False
        if len(indexed_o) == 0:
            print("out bad agent")
            return False
        # if not self.check_connections_per_node(connections):
        #     print "too many pre-post"
        #     return False
        if self.strict_io:
            checked_inputs = [0 for i in range(self.inputs)]
            for i in indexed_i:
                checked_inputs[indexed_i[i]] += 1
            checked_outputs = [0 for i in range(self.outputs)]
            for o in indexed_o:
                checked_outputs[indexed_o[o]] += 1
            if not np.count_nonzero(checked_inputs) == self.inputs or \
                    not np.count_nonzero(checked_outputs) == self.outputs:
                print("not all io")
                return False

        if self.force_i2o:
            graph = nx.convert_matrix.from_numpy_matrix(np.asmatrix(conn_matrix), create_using=nx.DiGraph)
            '''
            import networkx as nx
            import matplotlib.pyplot as plt
            graph = nx.convert_matrix.from_numpy_matrix(np.asmatrix(agent[0]), create_using=nx.DiGraph)
            nx.draw(graph)
            plt.show()
            '''
            shortest_paths = nx.all_pairs_shortest_path_length(graph)
            for n_paths in shortest_paths:
                neuron = n_paths[0]
                paths = n_paths[1]
                i_connected = True
                if neuron in indexed_i:
                    i_connected = False
                    for o in indexed_o:
                        if o in paths:
                            i_connected = True
                            break
                if not i_connected:
                    print("bad io")
                    return False
        print("good agent")
        return child

    '''here is where the children (the next generation of the population) are created for both a species and for the 
    entire population if required'''
    def generate_children(self, pop, birthing, fitness_shaping=True):
        parents = deepcopy(pop)
        children = []
        mumate_dict = {}
        # the best agents in the population (the elite) are automatically added to the next generation
        elite = int(math.ceil(len(pop) * self.elitism))
        parents.sort(key=lambda x: x[2], reverse=True)
        for i in range(elite):
            children.append([parents[i][0], parents[i][1]])
        i = elite
        # keep creating children until the population is full
        while i < birthing:
            birthing_type = np.random.random()
            # create a child by mutating a selected parent
            if birthing_type - self.sexuality['asexual'] < 0:
                if fitness_shaping:
                    parent = parents[self.select_parents(parents)]
                else:
                    print("use a function to determine the parent based on fitness")
                mutate_key = {}
                child = self.mutate(parent, mutate_key)
                # if the child created is beyond the maximum depth allowed for an agent restart and try again
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print("as3x d")
                if child:
                    if not self.valid_net(child):
                        child = False
                        print("as3x v")
            # create a child by mating 2 selected parents
            elif birthing_type - self.sexuality['asexual'] - self.sexuality['sexual'] < 0:
                if fitness_shaping:
                    mum = parents[self.select_parents(parents)]
                    dad = parents[self.select_parents(parents)]
                else:
                    print("use a function to determine the parent based on fitness")
                mutate_key = {}
                child = self.mate(mum, dad, mutate_key)
                # if the child created is beyond the maximum depth allowed for an agent restart and try again
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print("mate d")
                if child:
                    if not self.valid_net(child):
                        child = False
                        print("mate v")
            # create a child by first mating 2 parents then mutating the offspring
            elif birthing_type - self.sexuality['asexual'] - self.sexuality['sexual'] - self.sexuality['both'] < 0:
                if fitness_shaping:
                    mum = parents[self.select_parents(parents)]
                    dad = parents[self.select_parents(parents)]
                else:
                    print("use a function to determine the parent based on fitness")
                mutate_key = {}
                child = self.mate(mum, dad, mutate_key)
                # if the child created is beyond the maximum depth allowed for an agent restart and try again
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print("both mate d")
                if child:
                    child = self.mutate(child, mutate_key)
                    # if the child created is beyond the maximum depth allowed for an agent restart and try again
                    if self.motifs.depth_read(child) > self.maximum_depth:
                        child = False
                        print("both as3x d")
                if child:
                    if not self.valid_net(child):
                        child = False
                        print("both v")
                mutate_key['sex'] = 2
            # create a child but creating motifs of motifs
            else:
                mutate_key = {}
                child = self.fresh_child(mutate_key)
                # if the child created is beyond the maximum depth allowed for an agent restart and try again
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print("fresh d")
                if child:
                    if not self.valid_net(child):
                        child = False
                        print("fresh v")
            # if a child is created give it a random seed which is used to seed the random selection of inputs and
            # outputs, now a redundant fucntion giving the current mapping of IO
            if child:
                children.append([child, np.random.randint(200)])
                mumate_dict[child] = mutate_key
                i += 1
                print("created child", i, "of", birthing)
            else:
                print("not a valid child")
        return children, mumate_dict

    ''''''
    def select_parents(self, parents, best_first=True):
        if self.viable_parents == 0:
            return self.select_shaped(len(parents), best_first=best_first)
        else:
            allowed_to_mate = int(math.ceil(len(parents) * self.viable_parents))
            total_fitness = 0.0
            for i in range(allowed_to_mate):
                if best_first:
                    total_fitness += parents[i][2]
                else:
                    total_fitness += parents[len(parents) - 1 - i][2]
            selection = np.random.uniform(low=0, high=total_fitness)
            for i in range(allowed_to_mate):
                if best_first:
                    selection -= parents[i][2]
                else:
                    selection -= parents[len(parents) - 1 - i][2]
                if selection < 0:
                    if best_first:
                        return i
                    else:
                        return len(parents) - 1 - i

    def select_shaped(self, list_size, best_first=True):
        list_total = 0
        for i in range(list_size):
            list_total += i
        selection = np.random.randint(list_total)
        for i in range(list_size):
            if best_first:
                selection -= (list_size - i)
            else:
                selection -= i
            if selection < 0:
                break
        return i

    def save_agents(self, iteration, config):
        with open('Agent population {}: {}.csv'.format(iteration, config), 'w') as agent_file:
            writer = csv.writer(agent_file, delimiter=',', lineterminator='\n')
            for agent in self.agent_pop:
                writer.writerow(agent)
            agent_file.close()

    def save_agent_connections(self, agent, iteration, config):
        conn_and_result = []
        conn_and_result.append(self.convert_agent(agent))
        conn_and_result.append("fitness:".format(agent[2]))
        conn_and_result.append("score:".format(agent[3]))
        np.save('best agent {}: score({}), {}.npy'.format(iteration, agent[3], config), conn_and_result)
        # with open('best agent {}: score({}), {}.csv'.format(iteration, agent[3], config), 'w') as conn_file:
        #     writer = csv.writer(conn_file, delimiter=',', lineterminator='\n')
        #     for thing in connections:
        #         writer.writerow([thing])
        #     writer.writerow(["fitness", agent[2]])
        #     writer.writerow(["score", agent[3]])
        #     conn_file.close()

    def save_status(self, config, iteration, best_performance_score, best_performance_fitness):
        with open('status for {}.csv'.format(config), 'w') as status_file:
            writer = csv.writer(status_file, delimiter=',', lineterminator='\n')
            writer.writerow([time.localtime()])
            writer.writerow([config])
            writer.writerow(['on iteration: {}'.format(iteration)])
            writer.writerow(['maximum score'])
            writer.writerow(self.max_score)
            if best_performance_score:
                writer.writerow(['best performance score:', max(best_performance_score), "at iteration", best_performance_score.index(max(best_performance_score))])
            else:
                writer.writerow(['best performance score'])
            writer.writerow(best_performance_score)
            if best_performance_fitness:
                writer.writerow(['best performance fitness:', max(best_performance_fitness), "at iteration", best_performance_fitness.index(max(best_performance_fitness))])
            else:
                writer.writerow(['best performance fitness'])
            writer.writerow(best_performance_fitness)
            writer.writerow(['average score'])
            writer.writerow(self.average_score)
            writer.writerow(['minimum score'])
            writer.writerow(self.min_score)
            status_file.close()

    def save_mutate_keys(self, iteration, config):
        with open('mutate keys for {}: {}.csv'.format(iteration, config), 'w') as key_file:
            writer = csv.writer(key_file, delimiter=',', lineterminator='\n')
            for agent in self.agent_pop:
                try:
                    mutate_key = self.agent_mutate_keys[agent[0]]
                    writer.writerow([agent[0], agent[2], agent[3]])
                    for attribute in mutate_key:
                        writer.writerow([attribute, mutate_key[attribute]])
                except:
                    None
                    # print "no key for agent ", agent[0]
            key_file.close()

    def new_status_update(self, agent_data, iteration, config, connections, best_performance_score, best_performance_fitness):
        if isinstance(agent_data[0], list):
            fitnesses = agent_data[0]
            scores = agent_data[1]
        else:
            fitnesses = agent_data
            scores = agent_data
        print("Agent fitnesses", fitnesses)
        best_agent_f = fitnesses.index(max(fitnesses))
        worst_agent_f = fitnesses.index(min(fitnesses))
        print("Agent scores", scores)
        best_agent_s = scores.index(max(scores))
        worst_agent_s = scores.index(min(scores))
        best_fitness = fitnesses[best_agent_f]
        best_score = scores[best_agent_s]
        worst_fitness = fitnesses[worst_agent_f]
        worst_score = scores[worst_agent_s]

        self.max_score.append(round(best_score, 2))
        self.min_score.append(round(worst_score, 2))
        self.average_score.append(round(np.average(fitnesses), 2))
        self.max_fitness.append(round(best_fitness, 2))
        self.min_fitness.append(round(worst_fitness, 2))
        self.average_fitness.append(round(np.average(fitnesses), 2))

        self.track_networks(connections, config)
        print("\n\nBest fitness agent:", best_agent_f, "with performance", best_fitness, scores[best_agent_f])
        print("Best score agent:", best_agent_s, "with perfromance", fitnesses[best_agent_s], best_score)
        self.print_agent_net(connections[best_agent_s])
        if best_performance_fitness:
            print("max best performance fitness:", max(best_performance_fitness), "at iteration ", best_performance_fitness.index(max(best_performance_fitness)))
        print("best performance fitness:", best_performance_fitness)
        print("maximum fitness:", self.max_fitness)
        # print "average fitness:", self.total_average
        print("minimum fitness:", self.min_fitness)
        best_scores = '{:3}'.format(scores[best_agent_s])
        print("best score was ", fitnesses[best_agent_s], " by agent:", best_agent_s, "with a score of: ", fitnesses[best_agent_s], "--->", best_scores)
        if best_performance_score:
            print("max best performance score:", max(best_performance_score), "at iteration ", best_performance_score.index(max(best_performance_score)))
        print("best performance score:", best_performance_score)
        print("maximum score:", self.max_score)
        print("average score:", self.average_score)
        print("minimum score:", self.min_score)
        print("\n\nBest fitness agent:", best_agent_f, "with performance", best_fitness, scores[best_agent_f])
        print("Best score agent:", best_agent_s, "with perfromance", fitnesses[best_agent_s], best_score)
        self.print_agent_net(connections[best_agent_s])
        # if config != 'test':
        #     self.save_agent_connections(self.agent_pop[best_agent], iteration, 'score '+config)
        #     self.save_agent_connections(self.agent_pop[best_agent_s], iteration, 'fitness '+config)
        #     self.save_status(config, iteration, best_performance_score, best_performance_fitness)
        #     self.save_mutate_keys(iteration, config)
        best_score_connections = self.convert_agent_tf(self.agent_pop[best_agent_s])[0]
        best_fitness_connections = self.convert_agent_tf(self.agent_pop[best_agent_f])[0]
        return best_score_connections, best_fitness_connections

    def status_update(self, combined_fitnesses, iteration, config, num_tests, connections, best_performance_score, best_performance_fitness):
        total_scores = [0 for i in range(len(combined_fitnesses))]
        average_fitness = 0
        worst_score = 100000000
        worst_agent = 'need to higher worst score'
        best_score = -100000000
        best_agent = 'need to lower best score'
        worst_fitness = 100000000
        worst_agent_s = 'need to higher worst score'
        best_fitness = -100000000
        best_agent_s = 'need to lower best score'
        for j in range(len(self.agent_pop)):
            scores = '|'
            for i in range(len(combined_fitnesses)):
                scores += '{:8}'.format(combined_fitnesses[i][j])
            print('{:3}'.format(j), scores)
            if self.agent_pop[j][2] > best_fitness:
                best_fitness = self.agent_pop[j][2]
                best_agent = j
            if self.agent_pop[j][2] < worst_fitness:
                worst_fitness = self.agent_pop[j][2]
                worst_agent = j
            average_fitness += self.agent_pop[j][2]
            combined_score = 0
            for i in range(num_tests):
                if combined_fitnesses[i][j] != 'fail':
                    combined_score += combined_fitnesses[i][j]
                    total_scores[i] += combined_fitnesses[i][j]
            if combined_score > best_score:
                best_score = combined_score
                best_agent_s = j
            if combined_score < worst_score:
                worst_score = combined_score
                worst_agent_s = j
            self.agent_pop[j].append(combined_score)
        best_scores = '{:3}'.format(combined_fitnesses[0][best_agent])
        for i in range(1, len(combined_fitnesses)):
            best_scores += ', {:3}'.format(combined_fitnesses[i][best_agent])
        self.track_networks(connections, config)
        best_fitness_score = sum(np.take(combined_fitnesses[:num_tests], best_agent, axis=1))
        worst_fitness_score = sum(np.take(combined_fitnesses[:num_tests], worst_agent, axis=1))
        print("At iteration: ", iteration, "\n")
        print("best fitness was ", best_fitness, " by agent:", best_agent, \
            "with a score of: ", best_fitness_score, "--->", best_scores)
        self.print_agent_net(connections[best_agent])
        self.max_score.append(round(best_score, 2))
        self.min_score.append(round(worst_score, 2))
        total_average = 0
        for i in range(num_tests):
            total_average += total_scores[i]
        total_average /= len(self.agent_pop)
        self.average_score.append(round(total_average, 2))
        # self.average_score.append(round(total_scores, 2))
        # self.max_fitness.append(round(best_fitness, 2))
        # self.min_fitness.append(round(worst_fitness, 2))
        self.max_fitness.append(round(best_fitness_score, 2))
        self.min_fitness.append(round(worst_fitness_score, 2))
        self.average_fitness.append(round(average_fitness / len(self.agent_pop), 2))
        if best_performance_fitness:
            print("max best performance fitness:", max(best_performance_fitness), "at iteration ", best_performance_fitness.index(max(best_performance_fitness)))
        print("best performance fitness:", best_performance_fitness)
        print("maximum fitness:", self.max_fitness)
        # print "average fitness:", self.total_average
        print("minimum fitness:", self.min_fitness)
        best_scores = '{:3}'.format(combined_fitnesses[0][best_agent_s])
        for i in range(1, len(combined_fitnesses)):
            best_scores += ', {:3}'.format(combined_fitnesses[i][best_agent_s])
        print("best score was ", best_score, " by agent:", best_agent_s, \
            "with a score of: ", round(best_score, 2), "--->", best_scores)
        if best_performance_score:
            print("max best performance score:", max(best_performance_score), "at iteration ", best_performance_score.index(max(best_performance_score)))
        print("best performance score:", best_performance_score)
        print("maximum score:", self.max_score)
        print("average score:", self.average_score)
        print("minimum score:", self.min_score)
        print("At iteration: ", iteration, "\n")
        print("best fitness was ", best_fitness, " by agent:", best_agent, \
            "with a score of: ", best_fitness_score, "--->", best_scores)
        self.print_agent_net(connections[best_agent])
        if config != 'test':
            self.save_agent_connections(self.agent_pop[best_agent], iteration, 'score '+config)
            self.save_agent_connections(self.agent_pop[best_agent_s], iteration, 'fitness '+config)
            self.save_status(config, iteration, best_performance_score, best_performance_fitness)
            self.save_mutate_keys(iteration, config)
        best_score_connections = self.convert_agent(self.agent_pop[best_agent_s])
        best_fitness_connections = self.convert_agent(self.agent_pop[best_agent])
        return best_score_connections, best_fitness_connections

    def print_agent_net(self, agent, draw_graph=False):
        if isinstance(agent, str):
            agent = self.convert_agent_tf(agent)
        conn_matrix = agent[0]
        indexed_i = agent[2]
        indexed_o = agent[3]
        print("Agent has:")
        print(len(conn_matrix) - len(indexed_i) - len(indexed_o), "hidden neurons")
        print(len(indexed_i), "input neurons")
        print(len(indexed_o), "output neurons")
        print(np.count_nonzero(conn_matrix), "connections")

        if draw_graph:
            import networkx as nx
            import matplotlib.pyplot as plt
            graph = nx.convert_matrix.from_numpy_matrix(np.asmatrix(conn_matrix, dtype=[('weight', float)]), create_using=nx.DiGraph)
            edge_labs = [w for u, v, w in graph.edges(data="weight")]
            labels = {}
            for i in range(len(conn_matrix)):
                if i in indexed_i:
                    labels[i] = 'i{}'.format(indexed_i[i])
                elif i in indexed_o:
                    labels[i] = 'o{}'.format(indexed_o[i])
                else:
                    labels[i] = 'h'
            pos = [[10.*np.cos(2.*np.pi*float(i) / float(len(conn_matrix))), 10.*np.sin(2.*np.pi*float(i) / float(len(conn_matrix)))]
                   for i in range(len(conn_matrix))]
            nx.draw_networkx_edges(graph, pos=pos, width=edge_labs, edge_color=edge_labs)
            nx.draw(graph, pos=pos, with_labels=True, labels=labels)
            plt.show()


    def track_networks(self, connections, config):
        hidden_sizes = []
        io_sizes = []
        number_of_connections = []
        plasticity_ratio = []
        depths = []
        scores_list = []
        fitness_list = []
        for i in range(self.pop_size):
            conn_matrix = connections[i][0]
            indexed_i = connections[i][2]
            indexed_o = connections[i][3]
            hidden_sizes.append(len(conn_matrix) - len(indexed_i) - len(indexed_o))
            io_sizes.append(len(indexed_i) + len(indexed_o))
            pl_ratio = 0#pl_count / (pl_count + non_pl_count)
            number_of_connections.append(np.count_nonzero(conn_matrix))
            plasticity_ratio.append(round(pl_ratio, 2))
            scores_list.append(self.agent_pop[i][2]) # was [3] as before there was score too with other fitness metrics
            fitness_list.append(self.agent_pop[i][2])
            depths.append(self.motifs.motif_configs[self.agent_pop[i][0]]['depth'])
        best_score_index = scores_list.index(np.max(scores_list))
        best_fitness_index = fitness_list.index(np.max(fitness_list))

        self.min_hidden_neurons.append(np.min(hidden_sizes))
        self.average_hidden_neurons.append(round(np.average(hidden_sizes), 2))
        self.max_hidden_neurons.append(np.max(hidden_sizes))
        try:
            self.weighted_hidden_score.append(round(np.average(hidden_sizes, weights=scores_list), 2))
        except:
            self.weighted_hidden_score.append(round(np.average(hidden_sizes), 2))
        self.weighted_hidden_fitness.append(round(np.average(hidden_sizes, weights=fitness_list), 2))
        self.best_score_hidden.append(hidden_sizes[best_score_index])
        self.best_fitness_hidden.append(hidden_sizes[best_fitness_index])

        self.min_io_neurons.append(np.min(io_sizes))
        self.average_io_neurons.append(round(np.average(io_sizes), 2))
        self.max_io_neurons.append(np.max(io_sizes))
        try:
            self.weighted_io_score.append(round(np.average(io_sizes, weights=scores_list), 2))
        except:
            self.weighted_io_score.append(round(np.average(io_sizes), 2))
        self.weighted_io_fitness.append(round(np.average(io_sizes, weights=fitness_list), 2))
        self.best_score_io.append(io_sizes[best_score_index])
        self.best_fitness_io.append(io_sizes[best_fitness_index])

        self.min_connections.append(np.min(number_of_connections))
        self.average_connections.append(round(np.average(number_of_connections), 2))
        self.max_connections.append(np.max(number_of_connections))
        try:
            self.weighted_conn_score.append(round(np.average(number_of_connections, weights=scores_list), 2))
        except:
            self.weighted_conn_score.append(round(np.average(number_of_connections), 2))
        self.weighted_conn_fitness.append(round(np.average(number_of_connections, weights=fitness_list), 2))
        self.best_score_conn.append(number_of_connections[best_score_index])
        self.best_fitness_conn.append(number_of_connections[best_fitness_index])

        self.min_pl_ratio.append(np.min(plasticity_ratio))
        self.average_pl_ratio.append(round(np.average(plasticity_ratio), 2))
        self.max_pl_ratio.append(np.max(plasticity_ratio))
        try:
            self.weighted_pl_ratio_score.append(round(np.average(plasticity_ratio, weights=scores_list), 2))
        except:
            self.weighted_pl_ratio_score.append(round(np.average(plasticity_ratio), 2))
        self.weighted_pl_ratio_fitness.append(round(np.average(plasticity_ratio, weights=fitness_list), 2))
        self.best_score_pl_ratio.append(plasticity_ratio[best_score_index])
        self.best_fitness_pl_ratio.append(plasticity_ratio[best_fitness_index])

        self.min_m_depth.append(np.min(depths))
        self.average_m_depth.append(round(np.average(depths), 2))
        self.max_m_depth.append(np.max(depths))
        try:
            self.weighted_m_depth_score.append(round(np.average(depths, weights=scores_list), 2))
        except:
            self.weighted_m_depth_score.append(round(np.average(depths), 2))
        self.weighted_m_depth_fitness.append(round(np.average(depths, weights=fitness_list), 2))
        self.best_score_m_depth.append(depths[best_score_index])
        self.best_fitness_m_depth.append(depths[best_fitness_index])

        if config != 'test':
            self.print_tracking(config)

    def print_tracking(self, config):
        print("\nbest score hidden", self.best_score_hidden)
        print("best score io", self.best_score_io)
        print("best score depth", self.best_score_m_depth)
        print("best score conn", self.best_score_conn)
        print("best score pl", self.best_score_pl_ratio)

        print("\nbest fitness hidden", self.best_fitness_hidden)
        print("best fitness io", self.best_fitness_io)
        print("best fitness depth", self.best_fitness_m_depth)
        print("best fitness conn", self.best_fitness_conn)
        print("best fitness pl", self.best_fitness_pl_ratio)

        print("\naverage hidden", self.average_hidden_neurons)
        print("average io", self.average_io_neurons)
        print("average depth", self.average_m_depth)
        print("average conn", self.average_connections)
        print("average pl", self.average_pl_ratio)

        print("\nscore ave hidden", self.weighted_hidden_score)
        print("score ave io", self.weighted_io_score)
        print("score ave depth", self.weighted_m_depth_score)
        print("score ave conn", self.weighted_conn_score)
        print("score ave pl", self.weighted_pl_ratio_score)

        print("\nfitness ave hidden", self.weighted_hidden_fitness)
        print("fitness ave io", self.weighted_io_fitness)
        print("fitness ave depth", self.weighted_m_depth_fitness)
        print("fitness ave conn", self.weighted_conn_fitness)
        print("fitness ave pl", self.weighted_pl_ratio_fitness)

        print("\nmin hidden", self.min_hidden_neurons)
        print("min io", self.min_io_neurons)
        print("min depth", self.min_m_depth)
        print("min conn", self.min_connections)
        print("min pl", self.min_pl_ratio)

        print("\nmax hidden", self.max_hidden_neurons)
        print("max io", self.max_io_neurons)
        print("max depth", self.max_m_depth)
        print("max conn", self.max_connections)
        print("max pl", self.max_pl_ratio, "\n")

        # with open('network tracking {}.csv'.format(config), 'w') as network_file:
        #     writer = csv.writer(network_file, delimiter=',', lineterminator='\n')
        #
        #     writer.writerow([config])
        #     writer.writerow(["\nbest score hidden", self.best_score_hidden])
        #     writer.writerow(["best score io", self.best_score_io])
        #     writer.writerow(["best score depth", self.best_score_m_depth])
        #     writer.writerow(["best score conn", self.best_score_conn])
        #     writer.writerow(["best score pl", self.best_score_pl_ratio])
        #
        #     writer.writerow(["\nbest fitness hidden", self.best_fitness_hidden])
        #     writer.writerow(["best fitness io", self.best_fitness_io])
        #     writer.writerow(["best fitness depth", self.best_fitness_m_depth])
        #     writer.writerow(["best fitness conn", self.best_fitness_conn])
        #     writer.writerow(["best fitness pl", self.best_fitness_pl_ratio])
        #
        #     writer.writerow(["\naverage hidden", self.average_hidden_neurons])
        #     writer.writerow(["average io", self.average_io_neurons])
        #     writer.writerow(["average depth", self.average_m_depth])
        #     writer.writerow(["average conn", self.average_connections])
        #     writer.writerow(["average pl", self.average_pl_ratio])
        #
        #     writer.writerow(["\nscore ave hidden", self.weighted_hidden_score])
        #     writer.writerow(["score ave io", self.weighted_io_score])
        #     writer.writerow(["score ave depth", self.weighted_m_depth_score])
        #     writer.writerow(["score ave conn", self.weighted_conn_score])
        #     writer.writerow(["score ave pl", self.weighted_pl_ratio_score])
        #
        #     writer.writerow(["\nfitness ave hidden", self.weighted_hidden_fitness])
        #     writer.writerow(["fitness ave io", self.weighted_io_fitness])
        #     writer.writerow(["fitness ave depth", self.weighted_m_depth_fitness])
        #     writer.writerow(["fitness ave conn", self.weighted_conn_fitness])
        #     writer.writerow(["fitness ave pl", self.weighted_pl_ratio_fitness])
        #
        #     writer.writerow(["\nmin hidden", self.min_hidden_neurons])
        #     writer.writerow(["min io", self.min_io_neurons])
        #     writer.writerow(["min depth", self.min_m_depth])
        #     writer.writerow(["min conn", self.min_connections])
        #     writer.writerow(["min pl", self.min_pl_ratio])
        #
        #     writer.writerow(["\nmax hidden", self.max_hidden_neurons])
        #     writer.writerow(["max io", self.max_io_neurons])
        #     writer.writerow(["max depth", self.max_m_depth])
        #     writer.writerow(["max conn", self.max_connections])
        #     writer.writerow(["max pl", self.max_pl_ratio, "\n"])
        #
        #     # writer.writerow(["min hidden", self.min_hidden_neurons])
        #     # writer.writerow(["average hidden", self.average_hidden_neurons])
        #     # writer.writerow(["max hidden", self.max_hidden_neurons])
        #     # writer.writerow(["score ave hidden", self.weighted_hidden_score])
        #     # writer.writerow(["fitness ave hidden", self.weighted_hidden_fitness])
        #     # writer.writerow(["best score hidden", self.best_score_hidden])
        #     # writer.writerow(["best fitness hidden", self.best_fitness_hidden])
        #     # writer.writerow([""])
        #     #
        #     # writer.writerow(["min io", self.min_io_neurons])
        #     # writer.writerow(["average io", self.average_io_neurons])
        #     # writer.writerow(["max io", self.max_io_neurons])
        #     # writer.writerow(["score ave io", self.weighted_io_score])
        #     # writer.writerow(["fitness ave io", self.weighted_io_fitness])
        #     # writer.writerow(["best score io", self.best_score_io])
        #     # writer.writerow(["best fitness io", self.best_fitness_io])
        #     # writer.writerow([""])
        #     #
        #     # writer.writerow(["min conn", self.min_connections])
        #     # writer.writerow(["average conn", self.average_connections])
        #     # writer.writerow(["max conn", self.max_connections])
        #     # writer.writerow(["score ave conn", self.weighted_conn_score])
        #     # writer.writerow(["fitness ave conn", self.weighted_conn_fitness])
        #     # writer.writerow(["best score conn", self.best_score_conn])
        #     # writer.writerow(["best fitness conn", self.best_fitness_conn])
        #     # writer.writerow([""])
        #     #
        #     # writer.writerow(["min pl", self.min_pl_ratio])
        #     # writer.writerow(["average pl", self.average_pl_ratio])
        #     # writer.writerow(["max pl", self.max_pl_ratio])
        #     # writer.writerow(["score ave pl", self.weighted_pl_ratio_score])
        #     # writer.writerow(["fitness ave pl", self.weighted_pl_ratio_fitness])
        #     # writer.writerow(["best score pl", self.best_score_pl_ratio])
        #     # writer.writerow(["best fitness pl", self.best_fitness_pl_ratio])
        #
        #     network_file.close()


    def read_fitnesses(self, config, worst_score, make_action):
        if config != 'test':
            fitnesses = np.load('fitnesses {}.npy'.format(config))
            # os.remove('fitnesses {}.npy'.format(config))
            fitnesses = fitnesses.tolist()
            processed_fitness = []
            for fitness in fitnesses:
                processed_score = []
                for score in fitness:
                    if score == 'fail':
                        if make_action:
                            processed_score.append([worst_score, -10000001, -10000001])
                        else:
                            processed_score.append(worst_score)
                    else:
                        if make_action:
                            if score[2] == 0:
                                processed_score.append([worst_score, score[1], score[2]])
                            else:
                                processed_score.append(score)
                        else:
                            processed_score.append(score)
                processed_fitness.append(processed_score)
            return processed_fitness
        else:
            processed_fitness = []
            for i in range(5):
                processed_score = []
                for j in range(self.pop_size):
                    processed_score.append([np.random.random() * 10000, np.random.random() * 10000, np.random.random() * 10000])
                processed_fitness.append(processed_score)
            return processed_fitness

class agent_species(object):
    # todo species could work on a fitness metric in isolation with a shared motif pool
    def __init__(self, initial_member):
        self.members = [initial_member]
        self.representative = initial_member
        self.offspring = 0
        self.age = 0
        self.avg_fitness = None
        self.max_fitness = None
        self.max_fitness_prev = None
        self.no_improvement_age = 0
        self.has_best = False

    def calc_metrics(self):
        self.max_fitness_prev = self.max_fitness
        total_fitness = 0
        max_fitness = None
        for member in self.members:
            fitness = member[2]
            if max_fitness is None:
                max_fitness = fitness
            else:
                if fitness > max_fitness:
                    max_fitness = fitness
            total_fitness += fitness
        self.avg_fitness = total_fitness / len(self.members)
        self.max_fitness = max_fitness
        if self.max_fitness_prev is not None:
            if self.max_fitness > self.max_fitness_prev:
                self.no_improvement_age = 0
            else:
                self.no_improvement_age += 1