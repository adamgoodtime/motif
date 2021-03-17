from methods.networks import motif_population
from methods.agents import agent_population
from methods.neurons import neuron_population
import numpy as np
# import sys
# import itertools
# import traceback
# import threading
# from multiprocessing.pool import ThreadPool
# import multiprocessing
# import pathos.multiprocessing
from experiments.evolution_config import *
from experiments.test_config import *
from experiments.test_tf_xor import test_xor
from experiments.test_tf_pen import test_pen


if __name__ == '__main__':
    print("starting")
    # global connections, arms, max_fail_score, pole_angle, inputs, outputs, config, test_data_set, encoding, runtime, maximum_depth, make_action, constant_delays, weight_max, input_current_stdev, stochastic, rate_on, rate_off

    fitness_weighting = [1, 1]
    # for i in range(number_of_tests):
    #     fitness_weighting.append(1)
    if spike_f:
        fitness_weighting.append(spike_f)

    neurons = neuron_population(inputs=inputs,
                                outputs=outputs,
                                pop_size=inputs+outputs+200+agent_pop_size*3,
                                io_prob=io_prob,
                                input_current_stdev=input_current_stdev,
                                read_population=read_neurons,
                                neuron_type=neuron_type,
                                default=not stdev_neurons,
                                calcium_tau=calcium_tau,
                                calcium_i_alpha=calcium_i_alpha)

    if neuron_type == 'IF_cond_exp':
        weight_max = 0.1
        config += 'cond '
    elif neuron_type == 'IF_curr_exp':
        weight_max = 4.8
        config += 'cure '
    elif neuron_type == 'IF_curr_alpha':
        weight_max = 23.5
        config += 'cura '
    elif neuron_type == 'calcium':
        weight_max = 4.8
        config += 'calc-{}-{} '.format(calcium_tau, calcium_i_alpha)
    elif neuron_type == 'tf_basic':
        weight_max = 5.05
        config += "tfb "
    elif neuron_type == 'tf_LIF':
        weight_max = 13
        config += "tf "
    else:
        print("incorrect neuron type")
        raise Exception
    if weight_scale:
        weight_max /= float(weight_scale)
        config += 'ws-{} '.format(weight_scale)
    motifs = motif_population(neurons,
                              max_motif_size=4,#maximum_depth[0],
                              no_weight_bins=no_bins,
                              no_delay_bins=no_bins,
                              weight_range=(0.005, weight_max),
                              constant_delays=constant_delays,
                              delay_range=(1., max_delay),
                              neuron_types=(['excitatory', 'inhibitory']),
                              global_io=('highest', 'seeded', 'in'),
                              read_entire_population=read_motifs,
                              keep_reading=keep_reading,
                              viable_parents=viable_parents,
                              plasticity=plasticity,
                              structural=structural,
                              population_size=agent_pop_size*3+inputs+outputs)

    # todo :add number of different motifs to the fitness function to promote regularity

    agents = agent_population(motifs,
                              pop_size=agent_pop_size,
                              inputs=inputs,
                              outputs=outputs,
                              elitism=elitism,
                              sexuality=[7./20., 8./20., 3./20., 2./20.],
                              # sexuality=[7./20., 9./20., 4./20., 0],
                              base_mutate=base_mutate,
                              multiple_mutates=multiple_mutates,
                              allow_i2o=allow_i2o,
                              # input_shift=0,
                              # output_shift=0,
                              maximum_depth=maximum_depth,
                              viable_parents=viable_parents)

    # config += "max_d-{}, rents-{}, elite-{}, psize-{}, bins-{}".format(
    #     maximum_depth, viable_parents, elitism, agent_pop_size, no_bins)

    config += "max_d-{}, bins-{}".format(
        maximum_depth, no_bins)

    if io_prob:
        config += ", io-{}".format(io_prob)
    else:
        config += " {}".format(motifs.global_io[1])
    if read_motifs and read_neurons:
        config += ' readmn-{}'.format(keep_reading)
    elif read_motifs:
        config += ' readm-{}'.format(keep_reading)
    elif read_neurons:
        config += ' readn-{}'.format(keep_reading)

    print(config)

    best_performance_score = []
    best_performance_fitness = []

    for gen in range(generations):

        print(config)

        if gen == 0:
            connections, spinn_conn = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0])
        elif reset_pop:
            if not gen % reset_pop:
                connections, spinn_conn = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0], create='reset')
        else:
            connections, spinn_conn = agents.generate_spinn_nets(input=inputs, output=outputs, max_depth=maximum_depth[0], create=False)

        # if gen != 0: # to check repeatability
        #     for i in range(repeat_best_amount):
        #         connections.append(best_score_connections)
        #     for i in range(repeat_best_amount):
        #         connections.append(best_fitness_connections)

        # config = 'test'
        if config != 'test':
            # globals()['exec_thing'] = exec_thing
            # exec(compile(open("../methods/exec_subprocess.py", "rb").read(), "../methods/exec_subprocess.py", 'exec'), globals())
            if exec_thing == 'xor':
                fitnesses = test_xor(connections)
            elif exec_thing == 'pen':
                fitnesses = test_pen(connections)

        print("returned")

        # fitnesses = agents.read_fitnesses(config, max_fail_score, make_action)

        print("1", motifs.total_weight)

        # agent_spikes = []
        # for k in range(len(connections)):
        #     spike_total = 0
        #     for j in range(number_of_tests):
        #         if isinstance(fitnesses[j][k], list):
        #             spike_total -= fitnesses[j][k][1] + fitnesses[j][k][2]
        #             fitnesses[j][k] = fitnesses[j][k][0]
        #         else:
        #             spike_total -= 1000000
        #     agent_spikes.append(spike_total)
        # if spike_f:
        #     fitnesses.append(agent_spikes)

        # if gen != 0:
        #     best_agent_repeat_score = []
        #     best_agent_repeat_fitness = []
        #     for i in range(len(fitnesses)):
        #         print(fitnesses[i][agent_pop_size:agent_pop_size+repeat_best_amount])
        #         best_agent_repeat_score.append(fitnesses[i][agent_pop_size:agent_pop_size+repeat_best_amount])
        #         best_agent_repeat_fitness.append(fitnesses[i][agent_pop_size+repeat_best_amount:agent_pop_size+(2*repeat_best_amount)])
        #         fitnesses[i] = fitnesses[i][0:agent_pop_size]
        #     test_scores = [0 for i in range(repeat_best_amount)]
        #     test_fitness = [0 for i in range(repeat_best_amount)]
        #     for i in range(number_of_tests):
        #         for j in range(repeat_best_amount):
        #             test_scores[j] += best_agent_repeat_score[i][j]
        #             test_fitness[j] += best_agent_repeat_fitness[i][j]
        #     best_performance_score.append(round(np.average(test_scores), 2))
        #     best_performance_fitness.append(round(np.average(test_fitness), 2))

        agents.pass_fitnesses(fitnesses, max_fail_score, fitness_weighting, fitness_shaping=shape_fitness)

        # [best_score_connections, best_fitness_connections] = agents.status_update(fitnesses, gen, config, number_of_tests, connections, best_performance_score, best_performance_fitness)
        [best_score_connections, best_fitness_connections] = agents.new_status_update(fitnesses, gen, config, spinn_conn, best_performance_score, best_performance_fitness)

        print("\nFINISHED - ", gen, "\n\n")

        # if gen != 0:
        #     print("The performance of the best agent from last generation was:")
        #     print("Score:")
        #     for i in range(number_of_tests):
        #         print(best_agent_repeat_score[i])
        #     print("Fitness:")
        #     for i in range(number_of_tests):
        #         print(best_agent_repeat_fitness[i])

        print("\nconfig: ", config, "\n")

        print("2", motifs.total_weight)

        if gen % 10 == 0:
            print("stop")

        motifs.adjust_weights(agents.agent_pop, reward_shape=reward_shape, iteration=gen, average=averaging_weights, develop_neurons=develop_neurons)

        print("3", motifs.total_weight)

        if config != 'test' and save_data:
            motifs.save_motifs(gen, config)
            agents.save_agents(gen, config)
            if stdev_neurons:
                neurons.save_neurons(gen, config)

        print("4", motifs.total_weight)

        motifs.set_delay_bins(no_bins, gen, generations)
        motifs.set_weight_bins(no_bins, gen, generations)
        agents.set_max_d(maximum_depth, gen, generations)

        agents.evolve(species=False)

        print("finished", gen, motifs.total_weight)

    print("done")


# adjust population weights and clean up unused motifs

# generate offspring
    # mutate the individual and translate it to a new motif
    # connection change
    # swap motif

# bandit(generations)

# if __name__ == '__main__':
#     main()

#ToDo
'''
anneal the amount of IO weight (unnecessary when added population of neurons)
add species to agents
keep a log of seperate motif components to affect the chance of certain nodes and connects being chosen randomly through mutation
    a separate population of weights and nodes (neurons and synapes)
variability in the plasticity rule
crossover not just random   
    select motifs based on IO ratio
    select based on weight
figure out the disparity between expected possible combinations and actual
'''