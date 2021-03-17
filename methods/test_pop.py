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
# from poisson.poisson_tools import *

mnist_pointer = '../../NE16/poisson'

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
    if constant_delays:
        delays = constant_delays
    else:
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

def connect_2_pops(connections, input_pop, output_pop, receptor_type, pre_pop_size, stdp_model, max_conn=252):
    [stdp, structural, non_plastic] = split_plastic(connections)
    if len(stdp) != 0:
        from_list_segments = [stdp[x:x+max_conn] for x in range(0, len(stdp), max_conn)]
        for connection in from_list_segments:
            p.Projection(input_pop, output_pop,
                         p.FromListConnector(connection),
                         receptor_type=receptor_type,
                         synapse_type=stdp_model)
    if len(structural) != 0:
        from_list_segments = [structural[x:x+max_conn] for x in range(0, len(structural), max_conn)]
        for connection in from_list_segments:
            p.Projection(input_pop, output_pop,
                         p.FromListConnector(connection),
                         receptor_type=receptor_type,
                         synapse_type=stdp_model)
    if len(non_plastic) != 0:
        from_list_segments = [non_plastic[x:x+max_conn] for x in range(0, len(non_plastic), max_conn)]
        for connection in from_list_segments:
            p.Projection(input_pop, output_pop,
                         p.FromListConnector(connection),
                         receptor_type=receptor_type,
                         synapse_type=stdp_model)

def get_scores(game_pop, simulator):
    g_vertex = game_pop._vertex
    try:
        scores = g_vertex.get_data(
            'score', simulator.no_machine_time_steps, simulator.placements,
            simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
        return scores.tolist()
    except:
        return [[max_fail_score], [max_fail_score], [max_fail_score], [max_fail_score]]

def return_chip_list(machine):
    chip_list = []
    for i in range(8):
        for j in range(8):
            chip_list.append([i, j])
    to_be_removed = []
    for chip in chip_list:
        if chip[0] > 4 and chip[1] < 3 and chip[0] - 4 > chip[1]:
            to_be_removed.append(chip)
        elif chip[0] < 4 and chip[1] > 3 and chip[0] + 3 < chip[1]:
            to_be_removed.append(chip)

    for removal in to_be_removed:
        del chip_list[chip_list.index(removal)]
    full_chip_list = []
    for i, ethernet in enumerate(machine.ethernet_connected_chips):
        print("i:", i, "- chip:", ethernet.x, "/", ethernet.y)
        chips_from_board = 0
        for chip in chip_list: #machine.BOARD_48_CHIPS:
            x = chip[0] + ethernet.x
            y = chip[1] + ethernet.y
            if machine.is_chip_at(x, y):
                if [ethernet.x, ethernet.y] != [0, 12] and [ethernet.x, ethernet.y] != [0, 24] and [ethernet.x, ethernet.y] != [0, 36] and [x, y] != [105, 49]:
                    full_chip_list.append([x, y])
                    chips_from_board += 1
                else:
                    print("removing bad board covering", x, "/", y)
            else:
                print("chip", x, "/", y, "does not exist")
            if chips_from_board >= max_chips_per_board:
                break
    return full_chip_list

def pop_test(connections, test_data, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                spike_f=False, make_action=True, exec_thing='bout', seed=0):
    np.random.seed(seed)
    sleep = 10 * np.random.random()
    # time.sleep(sleep)
    max_attempts = 2
    try_except = 0
    if isinstance(test_data[0], list):
        if exec_thing == 'arms':
            if len(test_data) >= 2:
                test_data_set = test_data
            else:
                test_data_set = [test_data]
        else:
            test_data_set = test_data
    else:
        test_data_set = [test_data]
    if max_chips_per_board:
        number_of_chips = round(len(test_data_set) * len(connections) * 1.2 * (48 / max_chips_per_board))
        placement = True
    else:
        placement = False
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
                if placement:
                    p.setup(timestep=1.0, min_delay=1, max_delay=127, n_chips_required=number_of_chips)
                else:
                    p.setup(timestep=1.0, min_delay=1, max_delay=127)
                if neuron_choice == 'IF_cond_exp':
                    neuron_type = p.IF_cond_exp
                elif neuron_choice == 'IF_curr_exp':
                    neuron_type = p.IF_curr_exp
                elif neuron_choice == 'IF_curr_alpha':
                    neuron_type = p.IF_curr_alpha
                elif neuron_choice == 'calcium':
                    neuron_type = p.extra_models.IFCurrExpCa2Adaptive
                else:
                    print("incorrect neuron type")
                    raise Exception
                p.set_number_of_neurons_per_core(neuron_type, 32)
                print("\nfinished setup seed = ", seed, "\n")
                print("test data = ", test_data)
                break
            except:
                traceback.print_exc()
                sleep = 1 * np.random.random()
                time.sleep(sleep)
            print("\nsetup", try_count, " seed = ", seed, "\n", "\n")
            try_count += 1
        print("\nfinished setup seed = ", seed, "\n")
        print(config)
        if placement:
            machine = p.get_machine()
            chip_list = return_chip_list(machine)
        if exec_thing == 'mnist':
            [data, labels] = get_train_data(mnist_pointer)
            starting_point = np.random.randint(60000 - data_size)
            sub_data = data[starting_point: starting_point + data_size]
            sub_labels = labels[starting_point: starting_point + data_size]
            mnist_spikes = mnist_poisson_gen(sub_data, 28, 28, max_freq, on_duration, off_duration)
            input_model = p.Population(28*28, p.SpikeSourceArray(spike_times=mnist_spikes), label='MNIST_input_pop')
        for test_data in test_data_set:
            for i in range(len(connections)):
                [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size, # turned off connections to inputs except output
                 i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out, excite_params, inhib_params] = connections[i]
                if len(in2e) == 0 and len(in2i) == 0 and len(in2out) == 0:
                    failures.append(i)
                    print("agent {} was not properly connected to the game".format(i))
                else:
                    if placement:
                        chip_x = chip_list[0][0]
                        chip_y = chip_list[0][1]
                        del chip_list[0]
                        while not machine.is_chip_at(chip_x, chip_y):
                            print(" 2nd chip", chip_x, "/", chip_y, "does not exist")
                            chip_x = chip_list[0][0]
                            chip_y = chip_list[0][1]
                            del chip_list[0]
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
                        input_model = gym.DoublePendulum(encoding=encoding,
                                                         time_increment=time_increment,
                                                         pole_length=pole_length,
                                                         pole_angle=test_data[0],
                                                         pole2_length=pole2_length,
                                                         pole2_angle=pole2_angle,
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
                    elif exec_thing == 'logic':
                        input_model = gym.Logic(truth_table=truth_table,
                                                input_sequence=test_data,
                                                stochastic=stochastic,
                                                score_delay=score_delay,
                                                rate_on=rate_on,
                                                rate_off=rate_off,
                                                rand_seed=[np.random.randint(0xffff) for j in range(4)],
                                                label='logic_pop_{}-{}'.format(model_count, i))
                    elif exec_thing == 'arms':
                        input_model = gym.Bandit(arms=test_data,
                                                 reward_delay=exposure_time,
                                                 reward_based=arms_reward,
                                                 stochastic=stochastic,
                                                 constant_input=constant_input,
                                                 rate_on=rate_on,
                                                 rate_off=rate_off,
                                                 rand_seed=[np.random.randint(0xffff) for j in range(4)],
                                                 label='bandit_pop_{}-{}'.format(model_count, i))
                    elif exec_thing == 'recall':
                        input_model = gym.Recall(rate_on=rate_on,
                                                 rate_off=rate_off,
                                                 pop_size=recall_pop_size,
                                                 prob_command=prob_command,
                                                 prob_in_change=prob_in_change,
                                                 time_period=time_period,
                                                 stochastic=stochastic,
                                                 reward=recall_reward,
                                                 rand_seed=[np.random.randint(0xffff) for j in range(4)],
                                                 label='recall_pop_{}-{}'.format(model_count, i))
                    elif exec_thing == 'mnist':
                        # shared population already created
                        None
                    elif exec_thing == 'neuron':
                        input_model = p.IF_cond_exp()
                    else:
                        print("Incorrect input model selected")
                        raise Exception
                    if exec_thing != 'mnist':
                        if exec_thing != 'neuron':
                            input_pop_size = input_model.neurons()
                        else:
                            input_pop_size = neuron_pop_size
                        input_pops.append(p.Population(input_pop_size, input_model))
                    else:
                        input_pops.append(input_model)
                    if placement:
                        input_pops[model_count].add_placement_constraint(x=chip_x, y=chip_y)
                    # added to ensure that the arms and bandit are connected to and from something
                    # null_pop = p.Population(1, neuron_type(), label='null{}'.format(i))
                    # p.Projection(input_pops[model_count], null_pop, p.AllToAllConnector(), p.StaticSynapse(delay=1))
                    if fast_membrane:
                        output_pop.append(p.Population(outputs, neuron_type(tau_m=0.5,  # parameters for a fast membrane
                                                                            tau_refrac=0,
                                                                            v_thresh=-64,
                                                                            tau_syn_E=0.5,
                                                                            tau_syn_I=0.5),
                                                       label='output_pop_{}-{}'.format(model_count, i)))
                    else:
                        output_pop.append(p.Population(outputs, neuron_type(),
                                                       label='output_pop_{}-{}'.format(model_count, i)))
                    if placement:
                        output_pop[model_count].add_placement_constraint(x=chip_x, y=chip_y)
                    if spike_f == 'out' or make_action or exec_thing == 'mnist':
                        output_pop[model_count].record('spikes')
                    if exec_thing != 'mnist':
                        p.Projection(output_pop[model_count], input_pops[model_count], p.OneToOneConnector())
                    if e_size > 0:
                        excite_count += 1
                        excite.append(
                            p.Population(e_size, neuron_type(**excite_params),
                                         label='excite_pop_{}-{}'.format(excite_count, i)))
                        if noise_rate:
                            excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
                            if placement:
                                excite_noise.add_placement_constraint(x=chip_x, y=chip_y)
                            p.Projection(excite_noise, excite[excite_count], p.OneToOneConnector(),
                                         p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                        if spike_f:
                            excite[excite_count].record('spikes')
                        if placement:
                            excite[excite_count].add_placement_constraint(x=chip_x, y=chip_y)
                        excite_marker.append(i)
                    if i_size > 0:
                        inhib_count += 1
                        inhib.append(p.Population(i_size, neuron_type(**inhib_params),
                                                  label='inhib_pop_{}-{}'.format(inhib_count, i)))
                        if noise_rate:
                            inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
                            if placement:
                                inhib_noise.add_placement_constraint(x=chip_x, y=chip_y)
                            p.Projection(inhib_noise, inhib[inhib_count], p.OneToOneConnector(),
                                         p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                        if spike_f:
                            inhib[inhib_count].record('spikes')
                        if placement:
                            inhib[inhib_count].add_placement_constraint(x=chip_x, y=chip_y)
                        inhib_marker.append(i)
                    stdp_model = p.STDPMechanism(
                        timing_dependence=p.SpikePairRule(
                            tau_plus=20., tau_minus=20.0, A_plus=0.02, A_minus=0.02),
                        weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=weight_max))
                    if len(in2e) != 0:
                        [in_ex, in_in] = split_ex_in(in2e)
                        if len(in_ex) != 0:
                            connect_2_pops(in_ex, input_pops[model_count], excite[excite_count],
                                           'excitatory', e_size, stdp_model)
                        if len(in_in) != 0:
                            connect_2_pops(in_in, input_pops[model_count], excite[excite_count],
                                           'inhibitory', e_size, stdp_model)
                    if len(in2i) != 0:
                        [in_ex, in_in] = split_ex_in(in2i)
                        if len(in_ex) != 0:
                            connect_2_pops(in_ex, input_pops[model_count], inhib[inhib_count],
                                           'excitatory', i_size, stdp_model)
                        if len(in_in) != 0:
                            connect_2_pops(in_in, input_pops[model_count], inhib[inhib_count],
                                           'inhibitory', i_size, stdp_model)
                    if len(in2out) != 0:
                        [in_ex, in_in] = split_ex_in(in2out)
                        if len(in_ex) != 0:
                            connect_2_pops(in_ex, input_pops[model_count], output_pop[model_count],
                                           'excitatory', outputs, stdp_model)
                        if len(in_in) != 0:
                            connect_2_pops(in_in, input_pops[model_count], output_pop[model_count],
                                           'inhibitory', outputs, stdp_model)
                    if len(e2e) != 0:
                        connect_2_pops(e2e, excite[excite_count], excite[excite_count],
                                       'excitatory', e_size, stdp_model)
                    if len(e2i) != 0:
                        connect_2_pops(e2i, excite[excite_count], inhib[inhib_count],
                                       'excitatory', i_size, stdp_model)
                    if len(i2e) != 0:
                        connect_2_pops(i2e, inhib[inhib_count], excite[excite_count],
                                       'inhibitory', e_size, stdp_model)
                    if len(i2i) != 0:
                        connect_2_pops(i2i, inhib[inhib_count], inhib[inhib_count],
                                       'inhibitory', i_size, stdp_model)
                    if len(e2out) != 0:
                        connect_2_pops(e2out, excite[excite_count], output_pop[model_count],
                                       'excitatory', outputs, stdp_model)
                    if len(i2out) != 0:
                        connect_2_pops(i2out, inhib[inhib_count], output_pop[model_count],
                                       'inhibitory', outputs, stdp_model)
                    if len(out2e) != 0:
                        [in_ex, in_in] = split_ex_in(out2e)
                        if len(in_ex) != 0:
                            connect_2_pops(in_ex, output_pop[model_count], excite[excite_count],
                                           'excitatory', e_size, stdp_model)
                        if len(in_in) != 0:
                            connect_2_pops(in_in, output_pop[model_count], excite[excite_count],
                                           'inhibitory', e_size, stdp_model)
                    if len(out2i) != 0:
                        [in_ex, in_in] = split_ex_in(out2i)
                        if len(in_ex) != 0:
                            connect_2_pops(in_ex, output_pop[model_count], inhib[inhib_count],
                                           'excitatory', i_size, stdp_model)
                        if len(in_in) != 0:
                            connect_2_pops(in_in, output_pop[model_count], inhib[inhib_count],
                                           'inhibitory', i_size, stdp_model)
                    if len(out2in) != 0 and exec_thing != 'mnist':
                        [in_ex, in_in] = split_ex_in(out2in)
                        if len(in_ex) != 0:
                            connect_2_pops(in_ex, output_pop[model_count], input_pops[model_count],
                                           'excitatory', input_pop_size, stdp_model)
                        if len(in_in) != 0:
                            connect_2_pops(in_in, output_pop[model_count], input_pops[model_count],
                                           'inhibitory', input_pop_size, stdp_model)
                    if len(out2out) != 0:
                        [in_ex, in_in] = split_ex_in(out2out)
                        if len(in_ex) != 0:
                            connect_2_pops(in_ex, output_pop[model_count], output_pop[model_count],
                                           'excitatory', outputs, stdp_model)
                        if len(in_in) != 0:
                            connect_2_pops(in_in, output_pop[model_count], output_pop[model_count],
                                           'inhibitory', outputs, stdp_model)
        print("\nfinished connections seed = ", seed, "\n")
        simulator = get_simulator()
        try:
            print("\nrun seed = ", seed, "\n")
            if len(connections) == len(failures):
                p.end()
                print("nothing to run so ending and returning fail")
                return ['fail', 'fail']
            p.run(runtime)
            try_except = max_attempts
            break
        except:
            failure = traceback.format_exc()
            traceback.print_exc()
            try:
                print("\nrun 2 seed = ", seed, "\n")
                globals_variables.unset_simulator()
                print("end was necessary")
            except:
                traceback.print_exc()
                print("end wasn't necessary")
            try_except += 1
            print("failed to run on attempt ", try_except, "\n")  # . total fails: ", all_fails, "\n"
            if try_except >= max_attempts:
                print("calling it a failed population, splitting and rerunning")
                return ['fail', failure]
        # p.run(runtime)
        print("\nfinished run seed = ", seed, "\n")

    scores = []
    agent_fitness = []
    fails = 0
    excite_spike_count = [0 for i in range(len(output_pop))]
    excite_fail = 0
    inhib_spike_count = [0 for i in range(len(output_pop))]
    inhib_fail = 0
    output_spike_count = [0 for i in range(len(output_pop))]
    print("reading the spikes of ", config, '\n', seed)
    for i in range(len(output_pop)):
        print("started processing fitness of: ", i, '/', len(output_pop), "seed", seed)
        if i in failures:
            print("worst score for the failure")
            fails += 1
            scores.append([[max_fail_score], [max_fail_score], [max_fail_score], [max_fail_score]])
            # agent_fitness.append(scores[i])
            excite_spike_count[i] -= max_fail_score
            inhib_spike_count[i] -= max_fail_score
        else:
            if spike_f or make_action or exec_thing == 'mnist':
                if exec_thing == 'mnist':
                    choices = [[0 for j in range(10)] for k in range(data_size)]
                    spikes = output_pop[i - fails].get_data('spikes').segments[0].spiketrains
                    neuron_id = 0
                    for neuron in spikes:
                        for spike_time in neuron:
                            time_segment = 0
                            while float(spike_time) > (time_segment + 1) * (on_duration + off_duration):
                                time_segment += 1
                            choices[time_segment][neuron_id] += 1
                        neuron_id += 1
                if spike_f == 'out' or make_action:
                    spikes = output_pop[i - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        output_spike_count[i] += neuron.size
                        if output_spike_count[i] != 0:
                            break
                if isinstance(spike_f, float):
                    if spike_f > 0:
                        output_spike_count[i] = 0
                        spikes = output_pop[i - fails].get_data('spikes').segments[0].spiketrains
                        for neuron in spikes:
                            output_spike_count[i] += neuron.size
                if i in excite_marker and spike_f:
                    # print "counting excite spikes"
                    spikes = excite[i - excite_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        excite_spike_count[i] += neuron.size
                else:
                    excite_fail += 1
                    # print "had an excite failure"
                if i in inhib_marker and spike_f:
                    # print "counting inhib spikes"
                    spikes = inhib[i - inhib_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        inhib_spike_count[i] += neuron.size
                else:
                    inhib_fail += 1
                    # print "had an inhib failure"
            if exec_thing == 'mnist':
                score = 0
                for j in range(len(choices)):
                    choice = choices[j].index(np.max(choices[j]))
                    if choice == sub_labels[j]:
                        score += 1
                scores.append([[score]])
            elif exec_thing == 'recall':
                score = get_scores(game_pop=input_pops[i - fails], simulator=simulator)
                # correct_recalls = float(score[len(score) - 2][0])
                correct_recalls_0 = float(score[len(score) - 3][0])
                correct_recalls_1 = float(score[len(score) - 2][0])
                correct_recalls = correct_recalls_0 + correct_recalls_1
                number_of_trials = float(score[len(score) - 1][0])
                accuracy = correct_recalls / number_of_trials
                # confidence - if both -ve keep negative
                if correct_recalls == 0:
                    confidence = 0
                elif prob_command > 0.5:
                    confidence = 1
                else:
                    confidence = (abs(correct_recalls_0) * abs(correct_recalls_1)) / ((correct_recalls / 2)**2)
                accuracy *= confidence
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
    print(seed, "\nThe scores for this run of {} agents are:".format(len(connections)))
    for i in range(len(output_pop)):
        print("c:{}, s:{}, si:{}, si0:{}".format(len(connections), len(scores), len(scores[i]), len(scores[i][0])))
        e_string = "e: {}".format(excite_spike_count[i])
        i_string = "i: {}".format(inhib_spike_count[i])
        score_string = ""
        # if reward == 0:
        #     for j in range(len(scores[i])):
        #         score_string += "{:4},".format(scores[i][j][0])
        # else:
        #     score_string += "{:4},".format(scores[i][len(scores[i])-1][0])
        score_string += "{:4},".format(scores[i][len(scores[i])-1][0])
        print("{:3} | {:8} {:8} - ".format(i, e_string, i_string), score_string)
    print("before end = ", seed)
    p.end()
    print("\nafter end = ", seed, "\n")
    print(config)
    return [agent_fitness, 'complete']

def print_fitnesses(fitnesses):
    # with open('fitnesses {} {}.csv'.format(config, test_id), 'w') as file:
    #     writer = csv.writer(file, delimiter=',', lineterminator='\n')
    #     for fitness in fitnesses:
    #         writer.writerow(fitness)
    #     file.close()
    np.save('fitnesses {} {}.npy'.format(config, test_id), fitnesses)

def read_globals(config):
    # read_globals = np.load('globals {}.npy'.format(config))
    # for thing in read_globals:
    #     globals()[thing] = read_globals[thing]
    file_name = 'globals {}.csv'.format(config)
    with open(file_name) as from_file:
        csvFile = csv.reader(from_file)
        for row in csvFile:
            try:
                globals()[row[0]] = literal_eval(row[1])
            except:
                print("", end=' ')
                # try:
                #     globals()[row[0]] = row[1]
                # except:
                #     print ""
                # traceback.print_exc()
                # break

print("thing")
# parser = argparse.ArgumentParser(
#     description='just trying to pass a single number into here',
# formatter_class=argparse.RawTextHelpFormatter)
# args = parser.parse_args()
config = sys.argv[1] #literal_eval(args.config)
test_id = sys.argv[2]#literal_eval(args.test_id)
neuron_choice = sys.argv[3]
file_name = 'data {} {}.npy'.format(config, test_id)
connections_and_config = np.load(file_name)

read_globals(config)

# fitnesses = pop_test(connections, test_data_set, split, runtime, exposure_time, noise_rate, noise_weight,
#                                reward, size_f, spike_f, True)
fitnesses = pop_test(*connections_and_config)

print_fitnesses(fitnesses)