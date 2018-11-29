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
                 asexual=0.5,
                 conn_param_mutate=0.1,
                 conn_add=0.03,
                 conn_gone=0.03,
                 io_mutate=0.03,
                 node_mutate=0.03,
                 motif_add=0.03,
                 motif_gone=0.03,
                 motif_switch=0.03,
                 new_motif=0.03,
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
        self.asexual = asexual
        self.conn_param_mutate = conn_param_mutate
        self.conn_add = conn_add
        self.conn_gone = conn_gone
        self.io_mutate = io_mutate
        self.node_mutate = node_mutate
        self.motif_add = motif_add
        self.motif_gone = motif_gone
        self.motif_switch = motif_switch
        self.new_motif = new_motif
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

    def generate_spinn_nets(self, input=None, output=None, create=True, max_depth=2):
        if input is None:
            input = self.inputs
        if output is None:
            output = self.outputs
        agent_connections = []
        if create:
            self.generate_population(max_depth)
        for agent in self.agent_pop:
            agent_connections.append(self.convert_agent(agent, input, output))

        return agent_connections

    def generate_population(self, max_depth):
        for i in range(self.pop_size):
            self.agent_pop.append(self.new_individual(max_depth))


    def new_individual(self, max_depth):
        agent = self.motifs.generate_individual(max_depth=max_depth)
        return agent


    def convert_agent(self, agent, inputs, outputs):
        SpiNN_connections = self.motifs.convert_individual(agent, inputs, outputs)
        return SpiNN_connections

    def fitness_shape(self, fitnesses):
        if isinstance(fitnesses[0], list):
            shaped_fitnesses = [0 for i in range(len(fitnesses[0]))]
            indexed_fitness = []
            for i in range(len(fitnesses)):
                new_indexes = []
                for j in range(len(fitnesses[i])):
                    new_indexes.append([fitnesses[i][j], j])
                new_indexes.sort()
                indexed_fitness.append(new_indexes)
            for metric in indexed_fitness:
                current_shape = 0
                for i in range(len(metric)):
                    if metric[i][0] == max_fail_score or metric[i][0] == 'fail':
                        shaped_fitnesses[metric[i][1]] += 0
                    else:
                        if i > 0:
                            if metric[i][0] != metric[i-1][0]:
                                current_shape = i
                        shaped_fitnesses[metric[i][1]] += current_shape  # maybe add some weighting here but I dunno
        else:
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


    def reset(self):
        for specie in self.species:
            specie.has_best = False

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
            mutate_key['m_add'] = 0
            mutate_key['m_gone'] = 0
            mutate_key['c_add'] = 0
            mutate_key['c_gone'] = 0
            mutate_key['param_w'] = 0
            mutate_key['param_d'] = 0
            mutate_key['mum'] = [parent[2], parent[3]]
            mutate_key['dad'] = [parent[2], parent[3]]
            mutate_key['sex'] = 0
        # acquire the motif of parent and copy it to avoid messing with both memory locations
        motif_config = self.motifs.return_motif(parent[0])
        config_copy = deepcopy(motif_config)
        motif_size = len(config_copy['node'])
        # loop through each node and randomly mutate
        for i in range(motif_size):
            # switch with a randomly selected motif
            if np.random.random() < self.motif_switch:
                selected = False
                while not selected:
                    selected_motif = self.motifs.select_motif()
                    selected = self.motifs.recurse_check(selected_motif)
                config_copy['node'][i] = selected_motif
                new_depth = self.motifs.motif_configs[config_copy['node'][i]]['depth']
                if new_depth >= config_copy['depth']:
                    config_copy['depth'] = new_depth + 1
                mutate_key['motif'] += 1
            # switch with a completely novel motif todo maybe add or make this a motif of motifs of w/e depth
            elif np.random.random() < self.new_motif:
                config_copy['node'][i] = self.motifs.generate_motif(weight=0)
                new_depth = self.motifs.motif_configs[config_copy['node'][i]]['depth']
                if new_depth >= config_copy['depth']:
                    config_copy['depth'] = new_depth + 1
                mutate_key['new'] += 1
            # change the IO configurations
            elif np.random.random() < self.io_mutate:
                old_io = config_copy['io'][i]
                while config_copy['io'][i] == old_io:
                    new_io = (np.random.choice((True, False)), np.random.choice((True, False)))
                    config_copy['io'][i] = new_io
                mutate_key['io'] += 1
            else:
                # switch the base node if it's a base node
                if np.random.random() < self.node_mutate:
                    mutate_key['node'] += 1
                    if config_copy['node'][i] == 'excitatory':
                        config_copy['node'][i] = 'inhibitory'
                    elif config_copy['node'][i] == 'inhibitory':
                        config_copy['node'][i] = 'excitatory'
                    else:
                        mutate_key['node'] -= 0.0001
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
                    bin = np.random.randint(0, self.motifs.no_weight_bins)
                    new_weight = self.motifs.weight_range[0] + (bin * self.motifs.weight_bin_width)
                    config_copy['conn'][i][2] = new_weight
                mutate_key['param_w'] += 1
            # delay
            if np.random.random() < self.conn_param_mutate:
                old_delay = config_copy['conn'][i][3]
                while old_delay == config_copy['conn'][i][3]:
                    bin = np.random.randint(0, self.motifs.no_weight_bins)
                    new_delay = self.motifs.delay_range[0] + (bin * self.motifs.delay_bin_width)
                    config_copy['conn'][i][3] = new_delay
                mutate_key['param_d'] += 1
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
                copy_copy['node'][node_count] = self.mutate([node], mutate_key)
            node_count += 1
        if copy_copy != config_copy:
            motif_id = self.motifs.insert_motif(copy_copy)
        # todo do something with mutate key
        return motif_id

    def mate(self, mum, dad, mutate_key):
        # maybe the crossover should be more than just random, incorporating depth or some other dad decision metric
        # maybe take the seed randomly from mum or dad?
        if mutate_key == {}:
            mutate_key['motif'] = 0
            mutate_key['new'] = 0
            mutate_key['node'] = 0
            mutate_key['io'] = 0
            mutate_key['m_add'] = 0
            mutate_key['m_gone'] = 0
            mutate_key['c_add'] = 0
            mutate_key['c_gone'] = 0
            mutate_key['param_w'] = 0
            mutate_key['param_d'] = 0
            mutate_key['mum'] = [mum[2], mum[3]]
            mutate_key['dad'] = [dad[2], dad[3]]
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

    # here is where the children are created for both a species and for the entire population if required
    def generate_children(self, pop, birthing, fitness_shaping=True):
        parents = deepcopy(pop)
        children = []
        mumate_dict = {}
        elite = int(math.ceil(len(pop) * self.elitism))
        parents.sort(key=lambda x: x[2], reverse=True)
        for i in range(elite):
            children.append([parents[i][0], parents[i][1]])
        i = elite
        while i < birthing:
            if np.random.random() < self.asexual:
                if fitness_shaping:
                    parent = parents[self.select_parents(parents)]
                else:
                    print "use a function to determine the parent based on fitness"
                mutate_key = {}
                child = self.mutate(parent, mutate_key)
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print "as3x d"
            else:
                if fitness_shaping:
                    mum = parents[self.select_parents(parents)]
                    dad = parents[self.select_parents(parents)]
                else:
                    print "use a function to determine the parent based on fitness"
                mutate_key = {}
                child = self.mate(mum, dad, mutate_key)
                if self.motifs.depth_read(child) > self.maximum_depth:
                    child = False
                    print "mate d"
            if child:
                children.append([child, np.random.randint(200)])
                mumate_dict[child] = mutate_key
                i += 1
            else:
                print "went over the maximum depth"
        return children, mumate_dict

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

    def save_status(self, config):
        with open('status for {}.csv'.format(config), 'w') as status_file:
            writer = csv.writer(status_file, delimiter=',', lineterminator='\n')
            writer.writerow(['maximum score'])
            writer.writerow(self.max_score)
            writer.writerow(['average score'])
            writer.writerow(self.average_score)
            writer.writerow(['minimum score'])
            writer.writerow(self.min_score)
            writer.writerow([time.localtime()])
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
            total_scores[i] /= len(self.agent_pop)
            total_average += total_scores[i]
        # self.average_score.append(total_scores)
        self.max_fitness.append(best_fitness)
        self.average_score.append(total_average)
        self.min_fitness.append(worst_fitness)
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
        self.save_agent_connections(self.agent_pop[best_agent], iteration, 'score '+config)
        self.save_agent_connections(self.agent_pop[best_agent_s], iteration, 'fitness '+config)
        self.save_status(config)
        self.save_mutate_keys(iteration, config)

    def get_scores(self, game_pop, simulator):
        g_vertex = game_pop._vertex
        scores = g_vertex.get_data(
            'score', simulator.no_machine_time_steps, simulator.placements,
            simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
        return scores.tolist()

    def thread_bandit_test_test(self, connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100,
                           noise_weight=0.01, reward=0, spike_f=False, seed=0):

        np.random.seed(seed)
        if np.random.random() < 0.5:
            result = []
            for i in range(len(connections)):
                result.append([np.random.random(), np.random.randint(100)])
        else:
            result = 'fail'

        # if len(connections) > 0:
        #     self.thread_bandit_test(connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100,
        #                    noise_weight=0.01, reward=0)
        # else:
        #     self.bandit_test(connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100,
        #                    noise_weight=0.01, reward=0)

        return result

    def thread_bandit(self, connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01, reward=0, size_f=False, spike_f=False, seed=0, top=True):

        def helper(args):
            return self.bandit_test(*args)
            # return self.thread_bandit_test_test(*args)

        step_size = len(connections) / split
        if step_size == 0:
            step_size = 1
        if isinstance(arms[0], list):
            connection_threads = []
            all_configs = [[[connections[x:x + step_size], arm, split, runtime, exposure_time, noise_rate, noise_weight,
                             reward, spike_f] for x in xrange(0, len(connections), step_size)] for arm in arms]
            # all_configs = [[[connections[x:x + step_size], arm, split, runtime, exposure_time, noise_rate, noise_weight,
            #                  reward, spike_f, np.random.randint(100)] for x in xrange(0, len(connections), step_size)] for arm in arms]
            for arm in all_configs:
                for config in arm:
                    connection_threads.append(config)
        else:
            connection_threads = [[connections[x:x + step_size], arms, split, runtime, exposure_time, noise_rate,
                                   noise_weight, reward, spike_f] for x in xrange(0, len(connections), step_size)]
            # connection_threads = [[connections[x:x + step_size], arms, split, runtime, exposure_time, noise_rate,
            #                        noise_weight, reward, spike_f, np.random.randint(100)] for x in xrange(0, len(connections), step_size)]
        pool = pathos.multiprocessing.Pool(processes=len(connection_threads))

        pool_result = pool.map(func=helper, iterable=connection_threads)

        for i in range(len(pool_result)):
            new_split = 4
            if pool_result[i] == 'fail' and len(connection_threads[i][0]) > 1:
                print "splitting ", len(connection_threads[i][0]), " into ", new_split, " pieces"
                problem_arms = connection_threads[i][1]
                pool_result[i] = self.thread_bandit(connection_threads[i][0], problem_arms, new_split, runtime, exposure_time, noise_rate, noise_weight, reward, spike_f, seed=i, top=False)

        agent_fitness = []
        for thread in pool_result:
            if isinstance(thread, list):
                for result in thread:
                    agent_fitness.append(result)
            else:
                agent_fitness.append(thread)

        if isinstance(arms[0], list) and top:
            copy_fitness = deepcopy(agent_fitness)
            agent_fitness = []
            for i in range(len(arms)):
                arm_results = []
                for j in range(self.pop_size):
                    arm_results.append(copy_fitness[(i*self.pop_size) + j])
                agent_fitness.append(arm_results)
            if size_f:
                arm_results = []
                for i in range(self.pop_size):
                    arm_results.append(connections[i][2] + connections[i][5])
                agent_fitness.append(arm_results)
        return agent_fitness


    def bandit_test(self, connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01, reward=0, spike_f=False):
        max_attempts = 2
        try_except = 0
        while try_except < max_attempts:
            bandit = []
            bandit_count = -1
            excite = []
            excite_count = -1
            excite_marker = []
            inhib = []
            inhib_count = -1
            inhib_marker = []
            failures = []
            # p.setup(timestep=1.0, min_delay=self.delay_range[0], max_delay=self.delay_range[1])
            p.setup(timestep=1.0, min_delay=1, max_delay=127)
            p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
            # starting_pistol = p.Population(len(arms), p.SpikeSourceArray(spike_times=[0]))
            for i in range(len(connections)):
                [in2e, in2i, e_size, e2e, e2i, i_size, i2e, i2i, e2out, i2out] = connections[i]
                if (len(in2e) == 0 and len(in2i) == 0) or (len(e2out) == 0 and len(i2out) == 0):
                    failures.append(i)
                    print "agent {} was not properly connected to the game".format(i)
                else:
                    bandit_count += 1
                    bandit.append(
                        p.Population(len(arms), Bandit(arms, exposure_time, reward_based=reward, label='bandit_pop_{}-{}'.format(bandit_count, i))))
                    if e_size > 0:
                        excite_count += 1
                        excite.append(
                            p.Population(e_size, p.IF_cond_exp(), label='excite_pop_{}-{}'.format(excite_count, i)))
                        excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(excite_noise, excite[excite_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                        excite[excite_count].record('spikes')
                        excite_marker.append(i)
                    if i_size > 0:
                        inhib_count += 1
                        inhib.append(p.Population(i_size, p.IF_cond_exp(), label='inhib_pop_{}-{}'.format(inhib_count, i)))
                        inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(inhib_noise, inhib[inhib_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                        inhib[inhib_count].record('spikes')
                        inhib_marker.append(i)
                    if len(in2e) != 0:
                        p.Projection(bandit[bandit_count], excite[excite_count], p.FromListConnector(in2e),
                                     receptor_type='excitatory')
                        # p.Projection(starting_pistol, excite[excite_count], p.FromListConnector(in2e),
                        #              receptor_type='excitatory')
                    if len(in2i) != 0:
                        p.Projection(bandit[bandit_count], inhib[inhib_count], p.FromListConnector(in2i),
                                     receptor_type='excitatory')
                        # p.Projection(starting_pistol, inhib[inhib_count], p.FromListConnector(in2i),
                        #              receptor_type='excitatory')
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
                try:
                    globals_variables.unset_simulator()
                    print "end was necessary"
                except:
                    traceback.print_exc()
                    print "end wasn't necessary"
                try_except += 1
                print "failed to run on attempt ", try_except, "\n"  # . total fails: ", all_fails, "\n"
                if try_except >= max_attempts:
                    print "calling it a failed population, splitting and rerunning"
                    return 'fail'

        scores = []
        agent_fitness = []
        fails = 0
        excite_spike_count = [0 for i in range(len(connections))]
        excite_fail = 0
        inhib_spike_count = [0 for i in range(len(connections))]
        inhib_fail = 0
        for i in range(len(connections)):
            if i in failures:
                fails += 1
                scores.append([[max_fail_score], [max_fail_score], [max_fail_score], [max_fail_score]])
                agent_fitness.append(scores[i])
                excite_spike_count[i] -= max_fail_score
                inhib_spike_count[i] -= max_fail_score
                print "worst score for the failure"
            else:
                if i in excite_marker:
                    spikes = excite[i-excite_fail].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            excite_spike_count[i] += 1
                else:
                    excite_fail += 1
                if i in inhib_marker:
                    spikes = inhib[i-inhib_fail].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            inhib_spike_count[i] += 1
                else:
                    inhib_fail += 1
                scores.append(self.get_scores(game_pop=bandit[i - fails], simulator=simulator))
                # pop[i].stats = {'fitness': scores[i][len(scores[i]) - 1][0]}  # , 'steps': 0}
            if spike_f:
                agent_fitness.append([scores[i][len(scores[i]) - 1][0], excite_spike_count[i] + inhib_spike_count[i]])
            else:
                agent_fitness.append(scores[i][len(scores[i]) - 1][0])
            # print i, "| e:", excite_spike_count[i], "-i:", inhib_spike_count[i], "|\t", scores[i]
            e_string = "e: {}".format(excite_spike_count[i])
            i_string = "i: {}".format(inhib_spike_count[i])
            score_string = ""
            for j in range(len(scores[i])):
                score_string += "{:4},".format(scores[i][j][0])
            print "{:3} | {:8} {:8} - ".format(i, e_string, i_string), score_string
        p.end()

        return agent_fitness

    def read_fitnesses(self, config):
        fitnesses = []
        file_name = 'fitnesses {}.csv'.format(config)
        with open(file_name) as from_file:
            csvFile = csv.reader(from_file)
            for row in csvFile:
                metric = []
                for thing in row:
                    metric.append(literal_eval(thing))
                fitnesses.append(metric)
        return fitnesses

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