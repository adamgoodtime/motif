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
from bandit.spinn_bandit.python_models.bandit import Bandit
import math
import itertools
from copy import deepcopy
import operator
from spinn_front_end_common.utilities.globals_variables import get_simulator
import traceback
import math
from methods.networks import motif_population

class agent_pop(object):
    def __init__(self,
                 motif,
                 conn_weight=0.5,
                 # motif_weight=0.5,
                 crossover=0.5,
                 elitism=0.1,
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
                 similarity_threshold=0.4,
                 stagnation_age=25,
                 inputs=1,
                 outputs=2,
                 pop_size=100):

        self.motifs = motif
        self.pop_size = pop_size
        self.conn_weight = conn_weight
        self.motif_weight = 1 - conn_weight
        self.crossover = crossover
        self.elitism = elitism
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
        self.similarity_threshold = similarity_threshold
        self.stagnation_age = stagnation_age
        self.inputs = inputs
        self.outputs = outputs

        self.species = []
        self.agent_pop = []
        self.agent_nets = {}

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
            shaped_fitnesses = [0 for i in range(fitnesses[0])]
            indexed_fitness = []
            for i in range(len(fitnesses)):
                new_indexes = []
                for j in range(len(fitnesses[i])):
                    new_indexes.append([fitnesses[i][j], j])
                new_indexes.sort()
                indexed_fitness.append(new_indexes)
            for metric in indexed_fitness:
                for i in range(len(metric)):
                    shaped_fitnesses[metric[i][1]] += i  # maybe add some weighting here but I dunno
        else:
            shaped_fitnesses = [0 for i in range(len(fitnesses))]
            new_indexes = []
            for i in range(len(fitnesses)):
                new_indexes.append([fitnesses[i], i])
            new_indexes.sort()
            for i in range(len(fitnesses)):
                shaped_fitnesses[new_indexes[i][1]] += i
        return shaped_fitnesses

    def pass_fitnesses(self, fitnesses, fitness_shaping=True):
        if fitness_shaping:
            fitnesses = self.fitness_shape(fitnesses)
        for i in range(len(self.agent_pop)):
            self.agent_pop[i].append(fitnesses[i])


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
            self.agent_pop = self.generate_children(self.agent_pop, len(self.agent_pop))

    def mutate(self, parent, mutate_key={}):
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
            mutate_key['parent_f'] = parent[2]
            mutate_key['sex'] = 1
        motif_config = self.motifs.return_motif(parent[0])
        config_copy = deepcopy(motif_config)
        motif_size = len(config_copy['node'])
        for i in range(motif_size):
            if np.random.random() < self.motif_switch:
                config_copy['node'][i] = np.random.choice(self.motifs.motif_configs.keys())
                new_depth = self.motifs.motif_configs[config_copy['node'][i]]['depth']
                if new_depth >= config_copy['depth']:
                    config_copy['depth'] = new_depth + 1
                mutate_key['motif'] += 1
            elif np.random.random() < self.new_motif:
                config_copy['node'][i] = self.motifs.generate_motif()
                new_depth = self.motifs.motif_configs[config_copy['node'][i]]['depth']
                if new_depth >= config_copy['depth']:
                    config_copy['depth'] = new_depth + 1
                mutate_key['new'] += 1
            else:
                if np.random.random() < self.node_mutate:
                    mutate_key['node'] += 1
                    if config_copy['node'][i] == 'excitatory':
                        config_copy['node'][i] = 'inhibitory'
                    elif config_copy['node'][i] == 'inhibitory':
                        config_copy['node'][i] = 'excitatory'
                    else:
                        mutate_key['node'] -= 1
                if np.random.random() < self.io_mutate:
                    old_io = config_copy['io'][i]
                    while config_copy['io'][i] == old_io:
                        new_io = (np.random.choice((True, False)), np.random.choice((True, False)))
                        config_copy['io'][i] = new_io
                        mutate_key['io'] += 1
        if np.random.random() < self.conn_gone and len(config_copy['conn']) > 0:
            del config_copy['conn'][np.random.randint(len(config_copy['conn']))]
            mutate_key['c_gone'] += 1
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
        motif_id = self.motifs.insert_motif(config_copy)
        copy_copy = deepcopy(config_copy)
        node_count = 0
        for node in config_copy['node']:
            if node not in self.motifs.neuron_types:
                copy_copy['node'][node_count] = self.mutate([node], mutate_key)
            node_count += 1
        if copy_copy != config_copy:
            motif_id = self.motifs.insert_motif(copy_copy)
        return motif_id

    def mate(self, mum, dad):
        # maybe the crossover should be more than just random, incorporating depth or some other dad decision metric
            #
        child_id = mum[0]
        mum_motif = deepcopy(self.motifs.motif_configs[mum[0]])
        dad_list = self.motifs.list_motifs(dad[0])
        for i in range(len(mum_motif['node'])):
            if np.random.random() < self.crossover:
                mum_motif['node'][i] = np.random.choice(dad_list)
            elif mum_motif['node'][i] not in self.motifs.neuron_types:
                mum_motif['node'][i] = self.mate([mum_motif['node'][i]], dad)
        if self.motifs.motif_configs[mum[0]] != mum_motif:
            child_id = self.motifs.insert_motif(mum_motif)
        # for i in range(len(mum_motif['node'])):
        #     self.mate()
        return child_id

    # here is where the children are created for both a species and for the entire population if required
    def generate_children(self, pop, birthing, fitness_shaping=True):
        parents = deepcopy(pop)
        children = []
        elite = int(math.ceil(len(pop) * self.elitism))
        parents.sort(key=lambda x: x[2], reverse=True)
        for i in range(elite):
            children.append(parents[i][0])
        for i in range(elite, birthing):
            if np.random.random() < self.asexual:
                if fitness_shaping:
                    parent = parents[self.select_shaped(len(parents))]
                else:
                    print "use a function to determine the parent based on fitness"
                mutate_key = {}
                child = self.mutate(parent, mutate_key)
            else:
                if fitness_shaping:
                    mum = parents[self.select_shaped(len(parents))]
                    dad = parents[self.select_shaped(len(parents))]
                else:
                    print "use a function to determine the parent based on fitness"
                child = self.mate(mum, dad)
            children.append(child)
        return children

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

    def get_scores(self, game_pop, simulator):
        g_vertex = game_pop._vertex
        scores = g_vertex.get_data(
            'score', simulator.no_machine_time_steps, simulator.placements,
            simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
        return scores.tolist()

    def bandit_test(self, connections, arms, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01):
        max_attempts = 5
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
                        p.Population(len(arms), Bandit(arms, exposure_time, label='bandit_pop_{}-{}'.format(bandit_count, i))))
                    if e_size > 0:
                        excite_count += 1
                        excite.append(
                            p.Population(e_size, p.IF_cond_exp(), label='excite_pop_{}-{}'.format(excite_count, i)))
                        excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(excite_noise, excite[excite_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                        excite.record('spikes')
                        excite_marker.append(i)
                    if i_size > 0:
                        inhib_count += 1
                        inhib.append(p.Population(i_size, p.IF_cond_exp(), label='inhib_pop_{}-{}'.format(inhib_count, i)))
                        inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(inhib_noise, inhib[inhib_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                        inhib.record('spikes')
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

class agent_species(object):
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