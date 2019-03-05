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
from python_models.pendulum import Pendulum
from rank_inverted_pendulum.python_models.rank_pendulum import Rank_Pendulum
from double_inverted_pendulum.python_models.double_pendulum import DoublePendulum
import spinn_gym as gym
from spinn_arm.python_models.arm import Arm
# from spinn_breakout import Breakout
import math
import itertools
import sys
import termios
import contextlib
from copy import deepcopy
import operator
from spinn_front_end_common.utilities.globals_variables import get_simulator
import traceback
import math
from methods.networks import motif_population
import traceback
import csv
import threading
import subprocess
import pathos.multiprocessing
from spinn_front_end_common.utilities import globals_variables
import argparse
from ast import literal_eval


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

def split_plastic(connections):
    plastic = []
    non_plastic = []
    for conn in connections:
        if conn[4] == 'plastic':
            plastic.append([conn[0], conn[1], conn[2], conn[3]])
        else:
            non_plastic.append([conn[0], conn[1], conn[2], conn[3]])
    return plastic, non_plastic

def connect_to_arms(pre_pop, from_list, arms, r_type, plastic, stdp_model):
    arm_conn_list = []
    for i in range(len(arms)):
        arm_conn_list.append([])
    for conn in from_list:
        arm_conn_list[conn[1]].append((conn[0], 0, conn[2], conn[3]))
        # print "out:", conn[1]
        # if conn[1] == 2:
        #     print '\nit is possible\n'
    for i in range(len(arms)):
        if len(arm_conn_list[i]) != 0:
            if plastic:
                p.Projection(pre_pop, arms[i], p.FromListConnector(arm_conn_list[i]),
                             receptor_type=r_type, synapse_type=stdp_model)
            else:
                p.Projection(pre_pop, arms[i], p.FromListConnector(arm_conn_list[i]),
                             receptor_type=r_type)

def get_scores(game_pop, simulator):
    g_vertex = game_pop._vertex
    scores = g_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
    return scores.tolist()

def pop_test(connections, test_data, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                reward=0, spike_f=False, make_action=True, exec_thing='bout', seed=0):
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
                p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
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
        print config
        for i in range(len(connections)):
            [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
             i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out, excite_params, inhib_params] = connections[i]
            if len(in2e) == 0 and len(in2i) == 0 and len(in2out) == 0:
                failures.append(i)
                print "agent {} was not properly connected to the game".format(i)
            else:
                model_count += 1
                if exec_thing == 'pen':
                    input_model = gym.Pendulum(encoding=encoding,
                                               time_increment=time_increment,
                                               pole_length=pole_length,
                                               pole_angle=test_data[0],
                                               reward_based=reward_based,
                                               force_increments=force_increments,
                                               max_firing_rate=max_firing_rate,
                                               number_of_bins=number_of_bins,
                                               central=central,
                                               bin_overlap=bin_overlap,
                                               tau_force=tau_force,
                                               rand_seed=[np.random.randint(0xffff) for j in range(4)],
                                               label='pendulum_pop_{}-{}'.format(model_count, i))
                elif exec_thing == 'rank pen':
                    input_model = Rank_Pendulum(encoding=encoding,
                                                time_increment=time_increment,
                                                pole_length=pole_length,
                                                pole_angle=test_data[0],
                                                reward_based=reward_based,
                                                force_increments=force_increments,
                                                max_firing_rate=max_firing_rate,
                                                number_of_bins=number_of_bins,
                                                central=central,
                                                bin_overlap=bin_overlap,
                                                tau_force=tau_force,
                                                rand_seed=[np.random.randint(0xffff) for j in range(4)],
                                                label='rank_pendulum_pop_{}-{}'.format(model_count, i))
                elif exec_thing == 'double pen':
                    input_model = DoublePendulum(encoding=encoding,
                                                 time_increment=time_increment,
                                                 pole_length=pole_length,
                                                 pole_angle=test_data[0],
                                                 pole2_length=pole2_length,
                                                 pole2_angle=0,  # -test_data[0],
                                                 reward_based=reward_based,
                                                 force_increments=force_increments,
                                                 max_firing_rate=max_firing_rate,
                                                 number_of_bins=number_of_bins,
                                                 central=central,
                                                 bin_overlap=bin_overlap,
                                                 tau_force=tau_force,
                                                 rand_seed=[np.random.randint(0xffff) for j in range(4)],
                                                 label='double_pendulum_pop_{}-{}'.format(model_count, i))
                elif exec_thing == 'bout':
                    input_model = gym.Breakout(x_factor=x_factor,
                                               y_factor=y_factor,
                                               bricking=bricking,
                                               random_seed=[np.random.randint(0xffff) for j in range(4)],
                                               label='breakout_pop_{}-{}'.format(model_count, i))
                else:
                    input_model = gym.Bandit(arms=test_data,
                                             reward_delay=exposure_time,
                                             reward_based=reward,
                                             rand_seed=[np.random.randint(0xffff) for j in range(4)],
                                             label='bandit_pop_{}-{}'.format(model_count, i))
                input_pop_size = input_model.neurons()
                input_pops.append(p.Population(input_pop_size, input_model))
                # added to ensure that the arms and bandit are connected to and from something
                null_pop = p.Population(1, p.IF_cond_exp(), label='null{}'.format(i))
                p.Projection(input_pops[model_count], null_pop, p.AllToAllConnector())
                if fast_membrane:
                    output_pop.append(p.Population(outputs, p.IF_cond_exp(tau_m=0.5,  # parameters for a fast membrane
                                                                          tau_refrac=0,
                                                                          v_thresh=-64,
                                                                          tau_syn_E=0.5,
                                                                          tau_syn_I=0.5),
                                                   label='output_pop_{}-{}'.format(model_count, i)))
                else:
                    output_pop.append(p.Population(outputs, p.IF_cond_exp(),
                                                   label='output_pop_{}-{}'.format(model_count, i)))
                if spike_f == 'out' or make_action:
                    output_pop[model_count].record('spikes')
                p.Projection(output_pop[model_count], input_pops[model_count], p.AllToAllConnector())
                if e_size > 0:
                    excite_count += 1
                    v_rest = excite_params['v_rest']  # Resting membrane potential in mV.
                    cm = excite_params['cm']   # Capacity of the membrane in nF
                    tau_m = excite_params['tau_m']   # Membrane time constant in ms.
                    tau_refrac = excite_params['tau_refrac']   # Duration of refractory period in ms.
                    tau_syn_E = excite_params['tau_syn_E']  # Rise time of the excitatory synaptic alpha function in ms.
                    tau_syn_I = excite_params['tau_syn_I']  # Rise time of the inhibitory synaptic alpha function in ms.
                    e_rev_E = excite_params['e_rev_E']  # Reversal potential for excitatory input in mV
                    e_rev_I = excite_params['e_rev_I']   # Reversal potential for inhibitory input in mV
                    v_thresh = excite_params['v_thresh']  # Spike threshold in mV.
                    v_reset = excite_params['v_reset']   # Reset potential after a spike in mV.
                    i_offset = excite_params['i_offset']   # Offset current in nA
                    excite.append(
                        p.Population(e_size, p.IF_cond_exp(**excite_params
                                                            # v_rest=v_rest,
                                                            #                            cm=cm,
                                                            #                            tau_m=tau_m,
                                                            #                            tau_refrac=tau_refrac,
                                                            #                            tau_syn_E=tau_syn_E,
                                                            #                            tau_syn_I=tau_syn_I,
                                                            #                            e_rev_E=e_rev_E,
                                                            #                            e_rev_I=e_rev_I,
                                                            #                            v_thresh=v_thresh,
                                                            #                            v_reset=v_reset,
                                                            #                            i_offset=i_offset
                                                           ),
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
                    inhib.append(p.Population(i_size, p.IF_cond_exp(**inhib_params),
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
                    weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=0.1))
                if len(in2e) != 0:
                    [in_ex, in_in] = split_ex_in(in2e)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], excite[excite_count], p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(in2i) != 0:
                    [in_ex, in_in] = split_ex_in(in2i)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(in2out) != 0:
                    [in_ex, in_in] = split_ex_in(in2out)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(input_pops[model_count], output_pop[model_count], p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(e2e) != 0:
                    [plastic, non_plastic] = split_plastic(e2e)
                    if len(plastic) != 0:
                        p.Projection(excite[excite_count], excite[excite_count], p.FromListConnector(plastic),
                                     receptor_type='excitatory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], excite[excite_count], p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(e2i) != 0:
                    [plastic, non_plastic] = split_plastic(e2i)
                    if len(plastic) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count], p.FromListConnector(plastic),
                                     receptor_type='excitatory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(i2e) != 0:
                    [plastic, non_plastic] = split_plastic(i2e)
                    if len(plastic) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count], p.FromListConnector(plastic),
                                     receptor_type='inhibitory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count], p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(i2i) != 0:
                    [plastic, non_plastic] = split_plastic(i2i)
                    if len(plastic) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count], p.FromListConnector(plastic),
                                     receptor_type='inhibitory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(e2out) != 0:
                    [plastic, non_plastic] = split_plastic(e2out)
                    if len(plastic) != 0:
                        p.Projection(excite[excite_count], output_pop[model_count], p.FromListConnector(plastic),
                                     receptor_type='excitatory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], output_pop[model_count], p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(i2out) != 0:
                    [plastic, non_plastic] = split_plastic(i2out)
                    if len(plastic) != 0:
                        p.Projection(inhib[inhib_count], output_pop[model_count], p.FromListConnector(plastic),
                                     receptor_type='inhibitory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], output_pop[model_count], p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(out2e) != 0:
                    [in_ex, in_in] = split_ex_in(out2e)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], excite[excite_count], p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(out2i) != 0:
                    [in_ex, in_in] = split_ex_in(out2i)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(out2in) != 0:
                    [in_ex, in_in] = split_ex_in(out2in)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], input_pops[model_count], p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(out2out) != 0:
                    [in_ex, in_in] = split_ex_in(out2out)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(output_pop[model_count], output_pop[model_count], p.FromListConnector(non_plastic),
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
        print "started processing fitness of: ", i, '/', len(connections)
        if i in failures:
            print "worst score for the failure"
            fails += 1
            scores.append([[max_fail_score], [max_fail_score], [max_fail_score], [max_fail_score]])
            # agent_fitness.append(scores[i])
            excite_spike_count[i] -= max_fail_score
            inhib_spike_count[i] -= max_fail_score
        else:
            if spike_f or make_action:
                if spike_f == 'out' or make_action:
                    spikes = output_pop[i - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            output_spike_count[i] += 1
                if i in excite_marker and spike_f:
                    print "counting excite spikes"
                    spikes = excite[i - excite_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            excite_spike_count[i] += 1
                else:
                    excite_fail += 1
                    print "had an excite failure"
                if i in inhib_marker and spike_f:
                    print "counting inhib spikes"
                    spikes = inhib[i - inhib_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            inhib_spike_count[i] += 1
                else:
                    inhib_fail += 1
                    print "had an inhib failure"
            scores.append(get_scores(game_pop=input_pops[i - fails], simulator=simulator))
            # pop[i].stats = {'fitness': scores[i][len(scores[i]) - 1][0]}  # , 'steps': 0}
        print "\nfinished spikes", seed
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
        if reward == 0:
            for j in range(len(scores[i])):
                score_string += "{:4},".format(scores[i][j][0])
        else:
            score_string += "{:4},".format(scores[i][len(scores[i])-1][0])
        print "{:3} | {:8} {:8} - ".format(i, e_string, i_string), score_string
    print "before end = ", seed
    p.end()
    print "\nafter end = ", seed, "\n"
    print config
    return agent_fitness

def print_fitnesses(fitnesses):
    # with open('fitnesses {} {}.csv'.format(config, test_id), 'w') as file:
    #     writer = csv.writer(file, delimiter=',', lineterminator='\n')
    #     for fitness in fitnesses:
    #         writer.writerow(fitness)
    #     file.close()
    np.save('fitnesses {} {}.npy'.format(config, test_id), fitnesses)

def read_globals(config):
    file_name = 'globals {}.csv'.format(config)
    with open(file_name) as from_file:
        csvFile = csv.reader(from_file)
        for row in csvFile:
            try:
                globals()[row[0]] = literal_eval(row[1])
            except:
                print "",
                # try:
                #     globals()[row[0]] = row[1]
                # except:
                #     print ""
                # traceback.print_exc()
                # break

print "thing"
# parser = argparse.ArgumentParser(
#     description='just trying to pass a single number into here',
# formatter_class=argparse.RawTextHelpFormatter)
# args = parser.parse_args()
config = sys.argv[1] #literal_eval(args.config)
test_id = sys.argv[2]#literal_eval(args.test_id)
file_name = 'data {} {}.npy'.format(config, test_id)
connections_and_config = np.load(file_name)

read_globals(config)

# fitnesses = pop_test(connections, test_data_set, split, runtime, exposure_time, noise_rate, noise_weight,
#                                reward, size_f, spike_f, True)
fitnesses = pop_test(*connections_and_config)

print_fitnesses(fitnesses)