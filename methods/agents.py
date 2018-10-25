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
                 elitism=10, #%
                 asexual=0.5,
                 weight_mutate=0.8,
                 synapse_mutate=0.03,
                 node_mutate=0.03,
                 motif_mutate=0.03,
                 motif_switch=0.03,
                 inputs=1,
                 outputs=2,
                 pop_size=100):

        self.motifs = motif
        self.pop_size = pop_size
        self.elitism = elitism
        self.asexual = asexual
        self.weight_mutate = weight_mutate
        self.synapse_mutate = synapse_mutate
        self.node_mutate = node_mutate
        self.motif_mutate = motif_mutate
        self.motif_switch = motif_switch
        self.inputs = inputs
        self.outputs = outputs

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

    def evolve(self):
        print "evolve them here"

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