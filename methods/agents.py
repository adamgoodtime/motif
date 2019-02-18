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

max_fail_score = -1000000

class agent_population(object):
    def __init__(self,
                 motif,
                 conn_weight=0.5,
                 # motif_weight=0.5,
                 crossover=0.5,
                 elitism=0.1,
                 viable_parents=0.1,
                 strict_io=True,
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
        if self.motifs.io_weight[2] == 0:
            self.input_shift = 0
            self.output_shift = 0
        else:
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
        if create:
            if create == 'reset':
                self.agent_pop = []
            self.generate_population(max_depth)
        for agent in self.agent_pop:
            agent_connections.append(self.convert_agent(agent, input, output))

        return agent_connections

    '''creates the population'''
    def generate_population(self, max_depth):
        for i in range(self.pop_size):
            self.agent_pop.append(self.new_individual(max_depth))

    '''creates a new individual'''
    def new_individual(self, max_depth):
        agent = False
        while agent == False:
            agent = self.motifs.generate_individual(max_depth=max_depth)
            agent = self.valid_net(agent)
        return agent

    '''converts an agent into a list of connections'''
    def convert_agent(self, agent, inputs, outputs):
        SpiNN_connections = self.motifs.convert_individual(agent, inputs, outputs)
        return SpiNN_connections

    '''shapes the fitness generated by a population to make them relative to each other not absolute scores'''
    def fitness_shape(self, fitnesses):
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
            for metric in indexed_fitness:
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
                        shaped_fitnesses[metric[i][1]] += current_shape  # maybe add some weighting here but I dunno
        else:
            # the same as above but with only one fitness metric
            shaped_fitnesses = [0 for i in range(len(fitnesses))]
            new_indexes = []
            for i in range(len(fitnesses)):
                new_indexes.append([fitnesses[i], i])
            new_indexes.sort()
            for i in range(len(fitnesses)):
                if new_indexes[i][0] == max_fail_score or new_indexes[i][0] == 'fail':
                    shaped_fitnesses[new_indexes[i][1]] += 0
                else:
                    shaped_fitnesses[new_indexes[i][1]] += i
        return shaped_fitnesses

    '''either shape the fitnesses or not, then pass then pass the score to the agents for later processing'''
    def pass_fitnesses(self, fitnesses, fitness_shaping=True):
        if fitness_shaping:
            processed_fitnesses = self.fitness_shape(fitnesses)
        else:
            # todo figure out what to do about fitness less than 0
            if isinstance(fitnesses[0], list):
                processed_fitnesses = []
                for i in range(len(fitnesses[0])):
                    summed_f = 0
                    for j in range(len(fitnesses)):
                        if fitnesses[j][i] == 'fail':
                            summed_f = max_fail_score
                            break
                        summed_f += fitnesses[j][i]
                    processed_fitnesses.append(summed_f)

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
        print "evolve them here"
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
        self.species = filter(lambda s: s.no_improvement_age < self.stagnation_age or s.has_best, self.species)

        print "species are formed and quantified, now to add young and old age modifiers to quantify the amount of offspring generated"

    '''performs one step of evolution'''
    def evolve(self, species=True):
        if species:
            self.iterate_species()
        else:
            self.agent_pop, self.agent_mutate_keys = self.generate_children(self.agent_pop, len(self.agent_pop))

    '''Takes in a parent motif and mutates it in various ways, all sub-motifs are added with the final motif structure/
    id being returned. A key is kept of each mutation used to generate a child for later analysis of appropriate 
    mutation operators'''
    def mutate(self, parent, mutate_key={}):
        # initialise mutation key
        if mutate_key == {}:
            mutate_key['motif'] = 0
            mutate_key['new'] = 0
            mutate_key['node'] = 0
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
        motif_size = len(config_copy['node'])
        # loop through each node and randomly mutate
        for i in range(motif_size):
            prob_resize_factor = 1
            # switch with a randomly selected motif
            if np.random.random() * prob_resize_factor < self.motif_switch:
                selected = False
                while not selected:
                    selected_motif = self.motifs.select_motif()
                    selected = self.motifs.recurse_check(selected_motif)
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
            # switch the base node if it's a base node
            if np.random.random() * prob_resize_factor < self.node_mutate and config_copy['node'][i] in self.motifs.neuron_types:
                mutate_key['node'] += 1
                new_node = config_copy['node'][i]
                while config_copy['node'][i] == new_node:
                    new_node = np.random.choice(self.motifs.neuron_types)
                config_copy['node'][i] = new_node
                if not self.multiple_mutates:
                    continue
            elif config_copy['node'][i] in self.motifs.neuron_types and not self.multiple_mutates:
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
        # ad a connection
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
                if np.random.random() < 0.5:
                    conn.append('plastic')
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
            if np.random.random() < self.switch_plasticity and self.motifs.plasticity:
                if config_copy['conn'][i][4] == 'plastic':
                    config_copy['conn'][i][4] = 'non-plastic'
                else:
                    config_copy['conn'][i][4] = 'plastic'
                mutate_key['plasticity'] += 1
        # insert the new motif and then go through the nodes and mutate them
        motif_id = self.motifs.insert_motif(config_copy)
        if self.motifs.recurse_check(motif_id):
            copy_copy = deepcopy(config_copy)
        else:
            self.motifs.delete_motif(motif_id)
            print "That just tried to recurse"
            copy_copy = deepcopy(motif_config)
        node_count = 0
        for node in config_copy['node']:
            if node not in self.motifs.neuron_types:
                try:
                    copy_copy['node'][node_count] = self.mutate([node], mutate_key)
                except:
                    traceback.print_exc()
                    print "\nTried to mutate too many times\n"
                    return node
            node_count += 1
        if copy_copy != config_copy:
            motif_id = self.motifs.insert_motif(copy_copy)
        return motif_id

    '''mates 2 agents by iteration through the mother and probabilistically replacing a motif with a motif randomly
    selected from a list of dad's motifs'''
    def mate(self, mum, dad, mutate_key):
        # maybe the crossover should be more than just random, incorporating depth or some other dad decision metric
        # maybe take the seed randomly from mum or dad?
        if mutate_key == {}:
            mutate_key['motif'] = 0
            mutate_key['new'] = 0
            mutate_key['node'] = 0
            mutate_key['io'] = 0
            mutate_key['in_shift'] = 0
            mutate_key['out_shift'] = 0
            mutate_key['m_add'] = 0
            mutate_key['m_gone'] = 0
            mutate_key['c_add'] = 0
            mutate_key['c_gone'] = 0
            mutate_key['param_w'] = 0
            mutate_key['param_d'] = 0
            mutate_key['mum'] = [mum[2], mum[3]]
            mutate_key['dad'] = [dad[2], dad[3]]
            mutate_key['plasticity'] = 0
            mutate_key['sex'] = 1
        child_id = mum[0]
        mum_motif = deepcopy(self.motifs.motif_configs[mum[0]])
        dad_list = []
        dad_list = self.motifs.list_motifs(dad[0], dad_list)
        for i in range(len(mum_motif['node'])):
            if np.random.random() < self.crossover:
                mum_motif['node'][i] = np.random.choice(dad_list)
            elif mum_motif['node'][i] not in self.motifs.neuron_types:
                mum_motif['node'][i] = self.mate([mum_motif['node'][i]], dad, mutate_key)
        if self.motifs.motif_configs[mum[0]] != mum_motif:
            child_id = self.motifs.insert_motif(mum_motif)
        return child_id

    '''creates a child from motifs of motifs'''
    def fresh_child(self, mutate_key):
        mutate_key['motif'] = 0
        mutate_key['new'] = 0
        mutate_key['node'] = 0
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
        for i in range(self.maximum_depth):
            child = self.motifs.motif_of_motif(child, 1, self.maximum_depth, i)
            child = self.motifs.insert_motif(child)
        return child
    
    def valid_net(self, child):
        [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
         i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out] = self.convert_agent(child, self.inputs, self.outputs)
        if len(in2e) == 0 and len(in2i) == 0 and len(in2out) == 0:
            print "in bad agent"
            return False
        else:
            if self.strict_io:
                if len(e2out) == 0 and len(i2out) == 0 and len(in2out) == 0:
                    print "out bad agent"
                    return False
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
                    print "use a function to determine the parent based on fitness"
                mutate_key = {}
                child = self.mutate(parent, mutate_key)
                # if the child created is beyond the maximum depth allowed for an agent restart and try again
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print "as3x d"
                if child:
                    if not self.valid_net(child):
                        child = False
                        print "as3x v"
            # create a child by mating 2 selected parents
            elif birthing_type - self.sexuality['asexual'] - self.sexuality['sexual'] < 0:
                if fitness_shaping:
                    mum = parents[self.select_parents(parents)]
                    dad = parents[self.select_parents(parents)]
                else:
                    print "use a function to determine the parent based on fitness"
                mutate_key = {}
                child = self.mate(mum, dad, mutate_key)
                # if the child created is beyond the maximum depth allowed for an agent restart and try again
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print "mate d"
                if child:
                    if not self.valid_net(child):
                        child = False
                        print "mate v"
            # create a child by first mating 2 parents then mutating the offspring
            elif birthing_type - self.sexuality['asexual'] - self.sexuality['sexual'] - self.sexuality['both'] < 0:
                if fitness_shaping:
                    mum = parents[self.select_parents(parents)]
                    dad = parents[self.select_parents(parents)]
                else:
                    print "use a function to determine the parent based on fitness"
                mutate_key = {}
                child = self.mate(mum, dad, mutate_key)
                # if the child created is beyond the maximum depth allowed for an agent restart and try again
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print "both mate d"
                if child:
                    child = self.mutate(child, mutate_key)
                    # if the child created is beyond the maximum depth allowed for an agent restart and try again
                    if self.motifs.depth_read(child) > self.maximum_depth:
                        child = False
                        print "both as3x d"
                if child:
                    if not self.valid_net(child):
                        child = False
                        print "both v"
                mutate_key['sex'] = 2
            # create a child but creating motifs of motifs
            else:
                mutate_key = {}
                child = self.fresh_child(mutate_key)
                # if the child created is beyond the maximum depth allowed for an agent restart and try again
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print "fresh d"
                if child:
                    if not self.valid_net(child):
                        child = False
                        print "fresh v"
            # if a child is created give it a random seed which is used to seed the random selection of inputs and
            # outputs, now a redundant fucntion giving the current mapping of IO
            if child:
                children.append([child, np.random.randint(200)])
                mumate_dict[child] = mutate_key
                i += 1
            else:
                print "not a valid child"
        return children, mumate_dict

    ''''''
    def select_parents(self, parents, best_first=True):
        if self.viable_parents == 0:
            return self.select_shaped(len(parents), best_first=best_first)
        else:
            allowed_to_mate = int(math.ceil(len(parents) * self.viable_parents))
            total_fitness = 0
            for i in range(allowed_to_mate):
                if best_first:
                    total_fitness += parents[i][2]
                else:
                    total_fitness += parents[len(parents) - 1 - i][2]
            selection = np.random.uniform(total_fitness)
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
        connections = self.convert_agent(agent, self.inputs, self.outputs)
        with open('best agent {}: score({}), {}.csv'.format(iteration, agent[3], config), 'w') as conn_file:
            writer = csv.writer(conn_file, delimiter=',', lineterminator='\n')
            for thing in connections:
                writer.writerow([thing])
            writer.writerow(["fitness", agent[2]])
            writer.writerow(["score", agent[3]])
            conn_file.close()

    def save_status(self, config, iteration):
        with open('status for {}.csv'.format(config), 'w') as status_file:
            writer = csv.writer(status_file, delimiter=',', lineterminator='\n')
            writer.writerow(['on iteration: {}'.format(iteration)])
            writer.writerow(['maximum score'])
            writer.writerow(self.max_score)
            writer.writerow(['average score'])
            writer.writerow(self.average_score)
            writer.writerow(['minimum score'])
            writer.writerow(self.min_score)
            writer.writerow([''])
            writer.writerow([time.localtime()])
            writer.writerow([''])
            writer.writerow([config])
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
                    print "no key for agent ", agent[0]
            key_file.close()

    def status_update(self, combined_fitnesses, iteration, config, len_arms):
        total_scores = [0 for i in range(len(combined_fitnesses))]
        average_fitness = 0
        worst_score = 1000000
        worst_agent = 'need to higher worst score'
        best_score = -1000000
        best_agent = 'need to lower best score'
        worst_fitness = 1000000
        worst_agent_s = 'need to higher worst score'
        best_fitness = -1000000
        best_agent_s = 'need to lower best score'
        for j in range(len(self.agent_pop)):
            scores = '|'
            for i in range(len(combined_fitnesses)):
                scores += '{:8}'.format(combined_fitnesses[i][j])
            print '{:3}'.format(j), scores
            if self.agent_pop[j][2] > best_fitness:
                best_fitness = self.agent_pop[j][2]
                best_agent = j
            if self.agent_pop[j][2] < worst_fitness:
                worst_fitness = self.agent_pop[j][2]
                worst_agent = j
            average_fitness += self.agent_pop[j][2]
            combined_score = 0
            for i in range(len_arms):
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
        print "At iteration: ", iteration
        print "best fitness was ", best_fitness, " by agent:", best_agent, \
            "with a score of: ", best_scores
        self.max_score.append(best_score)
        self.min_score.append(worst_score)
        total_average = 0
        for i in range(len_arms):
            total_average += total_scores[i]
        total_average /= len(self.agent_pop)
        self.average_score.append(total_average)
        # self.average_score.append(total_scores)
        self.max_fitness.append(best_fitness)
        self.min_fitness.append(worst_fitness)
        self.average_fitness.append(average_fitness / len(self.agent_pop))
        print "maximum fitness:", self.max_fitness
        # print "average fitness:", self.total_average
        print "minimum fitness:", self.min_fitness
        best_scores = '{:3}'.format(combined_fitnesses[0][best_agent_s])
        for i in range(1, len(combined_fitnesses)):
            best_scores += ', {:3}'.format(combined_fitnesses[i][best_agent_s])
        print "best score was ", best_score, " by agent:", best_agent_s, \
            "with a score of: ", best_scores
        print "maximum score:", self.max_score
        print "average score:", self.average_score
        print "minimum score:", self.min_score
        if config != 'test':
            self.save_agent_connections(self.agent_pop[best_agent], iteration, 'score '+config)
            self.save_agent_connections(self.agent_pop[best_agent_s], iteration, 'fitness '+config)
            self.save_status(config, iteration)
            self.save_mutate_keys(iteration, config)

    def read_fitnesses(self, config, worst_score):
        #todo check if this handles fails properly
        fitnesses = np.load('fitnesses {}.npy'.format(config))
        os.remove('fitnesses {}.npy'.format(config))
        # file_name = 'fitnesses {}.csv'.format(config)
        # with open(file_name) as from_file:
        #     csvFile = csv.reader(from_file)
        #     for row in csvFile:
        #         metric = []
        #         for thing in row:
        #             if thing == 'fail':
        #                 metric.append(worst_score)
        #             else:
        #                 metric.append(literal_eval(thing))
        #         fitnesses.append(metric)
        return fitnesses.tolist()

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