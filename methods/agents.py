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
from methods.networks import motif_population

class agent_pop(object):
    def __init__(self,
                 motif,
                 conn_weight=0.5,
                 # motif_weight=0.5,
                 elitism=10, #%
                 asexual=0.5,
                 weight_mutate=0.8,
                 synapse_mutate=0.03,
                 node_mutate=0.03,
                 motif_mutate=0.03,
                 motif_switch=0.03,
                 similarity_threshold=0.4,
                 stagnation_age=25,
                 inputs=1,
                 outputs=2,
                 pop_size=100):

        self.motifs = motif
        self.pop_size = pop_size
        self.conn_weight = conn_weight
        self.motif_weight = 1 - conn_weight
        self.elitism = elitism
        self.asexual = asexual
        self.weight_mutate = weight_mutate
        self.synapse_mutate = synapse_mutate
        self.node_mutate = node_mutate
        self.motif_mutate = motif_mutate
        self.motif_switch = motif_switch
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

    def pass_fitnesses(self, fitnesses):
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

    def generate_species(self):
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

        self.species = filter(lambda s: s.no_improvement_age < self.stagnation_age or s.has_best, self.species)

        print "species are formed and quantified, now to add young and old age modifiers to quantify the amount of offspring generated"

    def evolve(self, species=True):
        if species:
            self.generate_species()


    def generate_children(self):
        print "here is where the children are created for both a species and for the entire population if required"

    def get_scores(self, game_pop, simulator):
        g_vertex = game_pop._vertex
        scores = g_vertex.get_data(
            'score', simulator.no_machine_time_steps, simulator.placements,
            simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
        return scores.tolist()

    def bandit_test(self, connections, arms, runtime=2000, exposure_time=200, noise_rate=50, noise_weight=1):
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
            # p.setup(timestep=1.0, min_delay=self.delay_range[0], max_delay=self.delay_range[1])
            p.setup(timestep=1.0, min_delay=1, max_delay=127)
            p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
            starting_pistol = p.Population(len(arms), p.SpikeSourceArray(spike_times=[0]))
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
                        p.Projection(starting_pistol, excite[excite_count], p.FromListConnector(in2e),
                                     receptor_type='excitatory')
                    if len(in2i) != 0:
                        p.Projection(bandit[bandit_count], inhib[inhib_count], p.FromListConnector(in2i),
                                     receptor_type='excitatory')
                        p.Projection(starting_pistol, inhib[inhib_count], p.FromListConnector(in2i),
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