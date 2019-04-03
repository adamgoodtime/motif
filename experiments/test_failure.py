from methods.networks import motif_population
from methods.agents import agent_population
import numpy as np
import traceback
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing
import pathos.multiprocessing
import csv
from ast import literal_eval
import itertools
import spynnaker8 as p
import time
import numpy as np
# from rank_inverted_pendulum.python_models.rank_pendulum import Rank_Pendulum
# from double_inverted_pendulum.python_models.double_pendulum import DoublePendulum
import spinn_gym as gym
import sys
from spinn_front_end_common.utilities.globals_variables import get_simulator
import traceback
import csv
from spinn_front_end_common.utilities import globals_variables
from ast import literal_eval

outputs = 2
config = ''
max_fail_score = -1000000
weight_max = 0.1

def test_failure(file_location, file):
    connections_setup = np.load(file_location+file)
    connections_setup = connections_setup.tolist()
    trace_error = connections_setup[len(connections_setup)-1]
    # print trace_error
    # del connections_setup[len(connections_setup)-1]
    pop_test(*connections_setup[0])

def split_ex_in(connections):
    excite = []
    inhib = []
    for conn in connections:
        if conn[2] > 0:
            excite.append(conn)
        else:
            inhib.append(conn)
    for conn in inhib:
        conn[2] *= -1
    return excite, inhib

def split_structural(connections):
    structural = []
    non_structural = []
    for conn in connections:
        if conn[5] == 'structural':
            structural.append([conn[0], conn[1], conn[2], conn[3], conn[4]])
        else:
            non_structural.append([conn[0], conn[1], conn[2], conn[3], conn[4]])
    return structural, non_structural

def split_plastic(connections):
    stdp = []
    structural = []
    non_plastic = []
    for conn in connections:
        if conn[4] == 'stdp':
            stdp.append([conn[0], conn[1], conn[2], conn[3]])
        elif conn[4] == 'structural':
            structural.append([conn[0], conn[1], conn[2], conn[3]])
        else:
            non_plastic.append([conn[0], conn[1], conn[2], conn[3]])
    return stdp, structural, non_plastic

def return_struct_model(size, stdp_model=None, scale=1):
    delays = [1, 15]
    default_parameters = {
        'stdp_model': stdp_model, 'f_rew': 10 ** (2*scale), 'weight': weight_max/3, 'delay': delays,
        's_max': 32, 'sigma_form_forward': 1, 'sigma_form_lateral': 1,
        'p_form_forward': 1., 'p_form_lateral': 1.,
        'p_elim_pot': 0, 'p_elim_dep': 2.450 * 10 ** (-2*scale),
        'grid': np.array([1, size]), 'lateral_inhibition': 0,
        'random_partner': False, 'is_distance_dependent': False}

    structure_model = p.StructuralMechanismSTDP(**default_parameters)
    #     #todo inhibitory or excitatory, is it jsut creating an edge?, limit to number of cons created
    #     stdp_model=stdp_model, #todo none=normal?
    #     weight=0,  # Use this weights when creating a new synapse todo, initial weight 0? can be an array?
    #     s_max=32,  # Maximum allowed fan-in per target-layer neuron todo target layer?
    #     grid=[np.sqrt(size), np.sqrt(size)],  # 2d spatial org of neurons todo what is this?
    #     # grid=[pop_size, 1], # 1d spatial org of neurons, uncomment this if wanted
    #     random_partner=True,  # Choose a partner neuron for formation at random, todo other options?
    #     # as opposed to selecting one of the last neurons to have spiked
    #     f_rew=size,  # 10, 000Hz todo only for all projections from pre to post or only from specified connections?
    #     sigma_form_forward=1.,  # spread of feed-forward connections todo what about feedback?
    #     delay=10  # Use this delay when creating a new synapse todo set from an array or random?
    # )
    # structure_model = p.StructuralMechanismStatic( #todo inhibitory or excitatory, is it jsut creating an edge?, limit to number of cons created
    #     weight=0,  # Use this weights when creating a new synapse todo, initial weight 0? can be an array?
    #     s_max=32,  # Maximum allowed fan-in per target-layer neuron todo target layer?
    #     grid=[np.sqrt(size), np.sqrt(size)],  # 2d spatial org of neurons todo what is this?
    #     # grid=[pop_size, 1], # 1d spatial org of neurons, uncomment this if wanted
    #     random_partner=True,  # Choose a partner neuron for formation at random, todo other options?
    #     # as opposed to selecting one of the last neurons to have spiked
    #     f_rew=size,  # Hz todo only for all projections from pre to post or only from specified connections?
    #     sigma_form_forward=1.,  # spread of feed-forward connections todo what about feedback?
    #     delay=10  # Use this delay when creating a new synapse todo set from an array or random?
    # )
    return structure_model

def get_scores(game_pop, simulator):
    g_vertex = game_pop._vertex
    scores = g_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
    return scores.tolist()

def pop_test(connections, test_data, split=4, runtime=41000, exposure_time=200, noise_rate=0, noise_weight=0.01,
                spike_f=False, make_action=True, exec_thing='arms', seed=0):
    np.random.seed(seed)
    sleep = 10 * np.random.random()
    # time.sleep(sleep)
    max_attempts = 2
    try_except = 0
    while try_except < max_attempts:
        input_pops = []
        model_count = -1
        input_arms = []
        excite = []
        excite_count = -1
        excite_marker = []
        inhib = []
        inhib_count = -1
        inhib_marker = []
        output_pop = []
        failures = []
        start = time.time()
        setup_retry_time = 60
        try_count = 0
        while time.time() - start < setup_retry_time:
            try:
                p.setup(timestep=1.0, min_delay=1, max_delay=127)
                neuron_type = p.IF_cond_exp
                p.set_number_of_neurons_per_core(neuron_type, 100)
                print "\nfinished setup seed = ", seed, "\n"
                print "test data = ", test_data
                break
            except:
                traceback.print_exc()
                sleep = 1 * np.random.random()
                time.sleep(sleep)
            print "\nsetup", try_count, " seed = ", seed, "\n", "\n"
            try_count += 1
        print "\nfinished setup seed = ", seed, "\n"
        for i in range(len(connections)):
            [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
             i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out, excite_params, inhib_params] = connections[i]
            if len(in2e) == 0 and len(in2i) == 0 and len(in2out) == 0:
                failures.append(i)
                print "agent {} was not properly connected to the game".format(i)
            else:
                model_count += 1
                input_model = gym.Bandit(arms=test_data,
                                         reward_delay=exposure_time,
                                         reward_based=0,
                                         rand_seed=[np.random.randint(0xffff) for j in range(4)],
                                         label='bandit_pop_{}-{}'.format(model_count, i))
                input_pop_size = input_model.neurons()
                input_pops.append(p.Population(input_pop_size, input_model))
                # added to ensure that the arms and bandit are connected to and from something
                null_pop = p.Population(1, neuron_type(), label='null{}'.format(i))
                p.Projection(input_pops[model_count], null_pop, p.AllToAllConnector(), p.StaticSynapse(delay=1))
                output_pop.append(p.Population(outputs, neuron_type(),
                                               label='output_pop_{}-{}'.format(model_count, i)))
                if spike_f == 'out' or make_action or exec_thing == 'mnist':
                    output_pop[model_count].record('spikes')
                if exec_thing != 'mnist':
                    p.Projection(output_pop[model_count], input_pops[model_count], p.AllToAllConnector())
                if e_size > 0:
                    excite_count += 1
                    excite.append(
                        p.Population(e_size, neuron_type(**excite_params),
                                     label='excite_pop_{}-{}'.format(excite_count, i)))
                    if noise_rate:
                        excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(excite_noise, excite[excite_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if spike_f:
                        excite[excite_count].record('spikes')
                    excite_marker.append(i)
                if i_size > 0:
                    inhib_count += 1
                    inhib.append(p.Population(i_size, neuron_type(**inhib_params),
                                              label='inhib_pop_{}-{}'.format(inhib_count, i)))
                    if noise_rate:
                        inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(inhib_noise, inhib[inhib_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if spike_f:
                        inhib[inhib_count].record('spikes')
                    inhib_marker.append(i)
                stdp_model = p.STDPMechanism(
                    timing_dependence=p.SpikePairRule(
                        tau_plus=20., tau_minus=20.0, A_plus=0.02, A_minus=0.02),
                    weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=weight_max))
                if len(in2e) != 0:
                    [in_ex, in_in] = split_ex_in(in2e)
                    if len(in_ex) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_ex)
                        if len(stdp) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='excitatory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count],
                                         p.FromListConnector(structural),
                                         receptor_type='excitatory',
                                         synapse_type=return_struct_model(e_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_in)
                        if len(stdp) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='inhibitory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count],
                                         p.FromListConnector(structural),
                                         receptor_type='inhibitory',
                                         synapse_type=return_struct_model(e_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(in2i) != 0:
                    [in_ex, in_in] = split_ex_in(in2i)
                    if len(in_ex) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_ex)
                        if len(stdp) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='excitatory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count],
                                         p.FromListConnector(structural),
                                         receptor_type='excitatory',
                                         synapse_type=return_struct_model(i_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_in)
                        if len(stdp) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='inhibitory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count],
                                         p.FromListConnector(structural),
                                         receptor_type='inhibitory',
                                         synapse_type=return_struct_model(i_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(in2out) != 0:
                    [in_ex, in_in] = split_ex_in(in2out)
                    if len(in_ex) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_ex)
                        if len(stdp) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='excitatory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count],
                                         p.FromListConnector(structural),
                                         receptor_type='excitatory',
                                         synapse_type=return_struct_model(outputs, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_in)
                        if len(stdp) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='inhibitory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count],
                                         p.FromListConnector(structural),
                                         receptor_type='inhibitory',
                                         synapse_type=return_struct_model(outputs, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(e2e) != 0:
                    [stdp, structural, non_plastic] = split_plastic(e2e)
                    if len(stdp) != 0:
                        p.Projection(excite[excite_count], excite[excite_count],
                                     p.FromListConnector(stdp),
                                     receptor_type='excitatory',
                                     synapse_type=stdp_model)
                    if len(structural) != 0:
                        p.Projection(excite[excite_count], excite[excite_count],
                                     p.FromListConnector(structural),
                                     receptor_type='excitatory',
                                     synapse_type=return_struct_model(e_size, stdp_model))
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], excite[excite_count],
                                     p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(e2i) != 0:
                    [stdp, structural, non_plastic] = split_plastic(e2i)
                    if len(stdp) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count],
                                     p.FromListConnector(stdp),
                                     receptor_type='excitatory',
                                     synapse_type=stdp_model)
                    if len(structural) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count],
                                     p.FromListConnector(structural),
                                     receptor_type='excitatory',
                                     synapse_type=return_struct_model(i_size, stdp_model))
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count],
                                     p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(i2e) != 0:
                    [stdp, structural, non_plastic] = split_plastic(i2e)
                    if len(stdp) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count],
                                     p.FromListConnector(stdp),
                                     receptor_type='inhibitory',
                                     synapse_type=stdp_model)
                    if len(structural) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count],
                                     p.FromListConnector(structural),
                                     receptor_type='inhibitory',
                                     synapse_type=return_struct_model(e_size, stdp_model))
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count],
                                     p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(i2i) != 0:
                    [stdp, structural, non_plastic] = split_plastic(i2i)
                    if len(stdp) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count],
                                     p.FromListConnector(stdp),
                                     receptor_type='inhibitory',
                                     synapse_type=stdp_model)
                    if len(structural) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count],
                                     p.FromListConnector(structural),
                                     receptor_type='inhibitory',
                                     synapse_type=return_struct_model(i_size, stdp_model))
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count],
                                     p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(e2out) != 0:
                    [stdp, structural, non_plastic] = split_plastic(e2out)
                    if len(stdp) != 0:
                        p.Projection(excite[excite_count], output_pop[model_count],
                                     p.FromListConnector(stdp),
                                     receptor_type='excitatory',
                                     synapse_type=stdp_model)
                    if len(structural) != 0:
                        p.Projection(excite[excite_count], output_pop[model_count],
                                     p.FromListConnector(structural),
                                     receptor_type='excitatory',
                                     synapse_type=return_struct_model(outputs, stdp_model))
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], output_pop[model_count],
                                     p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(i2out) != 0:
                    [stdp, structural, non_plastic] = split_plastic(i2out)
                    if len(stdp) != 0:
                        p.Projection(inhib[inhib_count], output_pop[model_count],
                                     p.FromListConnector(stdp),
                                     receptor_type='inhibitory',
                                     synapse_type=stdp_model)
                    if len(structural) != 0:
                        p.Projection(inhib[inhib_count], output_pop[model_count],
                                     p.FromListConnector(structural),
                                     receptor_type='inhibitory',
                                     synapse_type=return_struct_model(outputs, stdp_model))
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], output_pop[model_count],
                                     p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(out2e) != 0:
                    [in_ex, in_in] = split_ex_in(out2e)
                    if len(in_ex) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_ex)
                        if len(stdp) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='excitatory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count],
                                         p.FromListConnector(structural),
                                         receptor_type='excitatory',
                                         synapse_type=return_struct_model(e_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_in)
                        if len(stdp) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='inhibitory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count],
                                         p.FromListConnector(structural),
                                         receptor_type='inhibitory',
                                         synapse_type=return_struct_model(e_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(out2i) != 0:
                    [in_ex, in_in] = split_ex_in(out2i)
                    if len(in_ex) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_ex)
                        if len(stdp) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='excitatory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count],
                                         p.FromListConnector(structural),
                                         receptor_type='excitatory',
                                         synapse_type=return_struct_model(i_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_in)
                        if len(stdp) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='inhibitory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count],
                                         p.FromListConnector(structural),
                                         receptor_type='inhibitory',
                                         synapse_type=return_struct_model(i_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(out2in) != 0 and exec_thing != 'mnist':
                    [in_ex, in_in] = split_ex_in(out2in)
                    if len(in_ex) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_ex)
                        if len(stdp) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='excitatory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count],
                                         p.FromListConnector(structural),
                                         receptor_type='excitatory',
                                         synapse_type=return_struct_model(input_pop_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_in)
                        if len(stdp) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='inhibitory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count],
                                         p.FromListConnector(structural),
                                         receptor_type='inhibitory',
                                         synapse_type=return_struct_model(input_pop_size, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(out2out) != 0:
                    [in_ex, in_in] = split_ex_in(out2out)
                    if len(in_ex) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_ex)
                        if len(stdp) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='excitatory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count],
                                         p.FromListConnector(structural),
                                         receptor_type='excitatory',
                                         synapse_type=return_struct_model(outputs, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [stdp, structural, non_plastic] = split_plastic(in_in)
                        if len(stdp) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count],
                                         p.FromListConnector(stdp),
                                         receptor_type='inhibitory',
                                         synapse_type=stdp_model)
                        if len(structural) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count],
                                         p.FromListConnector(structural),
                                         receptor_type='inhibitory',
                                         synapse_type=return_struct_model(outputs, stdp_model))
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count],
                                         p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
        print "\nfinished connections seed = ", seed, "\n"
        simulator = get_simulator()
        try:
            print "\nrun seed = ", seed, "\n"
            if len(connections) == len(failures):
                p.end()
                print "nothing to run so ending and returning fail"
                return 'fail'
            p.run(runtime)
            try_except = max_attempts
            break
        except:
            traceback.print_exc()
            try:
                print "\nrun 2 seed = ", seed, "\n"
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
        # p.run(runtime)
        print "\nfinished run seed = ", seed, "\n"

    scores = []
    agent_fitness = []
    fails = 0
    excite_spike_count = [0 for i in range(len(connections))]
    excite_fail = 0
    inhib_spike_count = [0 for i in range(len(connections))]
    inhib_fail = 0
    output_spike_count = [0 for i in range(len(connections))]
    print "reading the spikes of ", config, '\n', seed
    for i in range(len(connections)):
        print "started processing fitness of: ", i, '/', len(connections), "seed", seed
        if i in failures:
            print "worst score for the failure"
            fails += 1
            scores.append([[max_fail_score], [max_fail_score], [max_fail_score], [max_fail_score]])
            # agent_fitness.append(scores[i])
            excite_spike_count[i] -= max_fail_score
            inhib_spike_count[i] -= max_fail_score
        else:
            if spike_f or make_action or exec_thing == 'mnist':
                made_action = False
                if spike_f == 'out' or make_action:
                    spikes = output_pop[i - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            output_spike_count[i] += 1
                            make_action = True
                            break
                        if make_action:
                            break
                if i in excite_marker and spike_f:
                    # print "counting excite spikes"
                    spikes = excite[i - excite_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            excite_spike_count[i] += 1
                else:
                    excite_fail += 1
                    # print "had an excite failure"
                if i in inhib_marker and spike_f:
                    # print "counting inhib spikes"
                    spikes = inhib[i - inhib_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            inhib_spike_count[i] += 1
                else:
                    inhib_fail += 1
                    # print "had an inhib failure"
            if exec_thing == 'recall':
                score = get_scores(game_pop=input_pops[i - fails], simulator=simulator)
                accuracy = float(score[len(score) - 2][0]) / float(score[len(score) - 1][0])
                scores.append([[accuracy]])
            else:
                scores.append(get_scores(game_pop=input_pops[i - fails], simulator=simulator))
            # pop[i].stats = {'fitness': scores[i][len(scores[i]) - 1][0]}  # , 'steps': 0}
        # print "\nfinished spikes", seed
        if spike_f or make_action:
            agent_fitness.append([scores[i][len(scores[i]) - 1][0], excite_spike_count[i] + inhib_spike_count[i], output_spike_count[i]])
        else:
            agent_fitness.append(scores[i][len(scores[i]) - 1][0])
        # print i, "| e:", excite_spike_count[i], "-i:", inhib_spike_count[i], "|\t", scores[i]
    print seed, "\nThe scores for this run of {} agents are:".format(len(connections))
    for i in range(len(connections)):
        print "c:{}, s:{}, si:{}, si0:{}".format(len(connections), len(scores), len(scores[i]), len(scores[i][0]))
        e_string = "e: {}".format(excite_spike_count[i])
        i_string = "i: {}".format(inhib_spike_count[i])
        score_string = ""
        # if reward == 0:
        #     for j in range(len(scores[i])):
        #         score_string += "{:4},".format(scores[i][j][0])
        # else:
        #     score_string += "{:4},".format(scores[i][len(scores[i])-1][0])
        score_string += "{:4},".format(scores[i][len(scores[i])-1][0])
        print "{:3} | {:8} {:8} - ".format(i, e_string, i_string), score_string
    print "before end = ", seed
    p.end()
    print "\nafter end = ", seed, "\n"
    print config
    return agent_fitness

file_loc = '/home/adampcloth/Documents/Simulations/Motif/failed agents/errors in master/'
file_name = 'failed agent 0.87180560467 bandit-0.4-10-0 pl ave action shape_f multate dev_n stdev_n 5000 max_d-[3, 10], w_max-0.1, rents-0.2, elite-0.2, psize-100, bins-[10, 375], io-0.95 read-5.npy'
test_failure(file_loc, file_name)
