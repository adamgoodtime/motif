from methods.networks import motif_population
from methods.agents import agent_population
import numpy as np
import itertools
import traceback
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing
import pathos.multiprocessing

connections = []

def bandit(generations):
    print "starting"
    global connections

    weight_max = 0.1

    arm1 = 0.9
    arm2 = 0.1
    arm3 = 0.1
    arm_len = 1
    arms = []
    for i in range(arm_len):
        # arms.append([arm1, arm2])
        # arms.append([arm2, arm1])
        for arm in list(itertools.permutations([arm1, arm2, arm3])):
            arms.append(list(arm))
    # arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
    '''arms = [[0.1, 0.2, 0.9, 0.3, 0.2, 0.1, 0.2, 0.1], [0.9, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.2],
            [0.3, 0.9, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1], [0.2, 0.1, 0.1, 0.9, 0.2, 0.3, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.2, 0.9, 0.2, 0.3, 0.2], [0.1, 0.2, 0.1, 0.2, 0.2, 0.9, 0.1, 0.3],
            [0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.9, 0.2], [0.1, 0.3, 0.2, 0.2, 0.1, 0.2, 0.1, 0.9]]
    # '''
    if isinstance(arms[0], list):
        number_of_arms = len(arms[0])
    else:
        number_of_arms = len(arms)

    split = 1

    agent_pop_size = 200
    reward_shape = False
    reward = 0
    noise_rate = 0
    noise_weight = 0.01
    maximum_depth = 10
    size_fitness = False
    spikes_fitness = False
    random_arms = 0
    viable_parents = 0.3
    elitism = 0.3
    runtime = 41000
    exposure_time = 200

    # check max motif count
    motifs = motif_population(max_motif_size=3,
                              no_weight_bins=5,
                              no_delay_bins=5,
                              weight_range=(0.005, weight_max),
                              # delay_range=(1, 25),
                              neuron_types=(['excitatory', 'inhibitory']),
                              io_weight=[2, number_of_arms, 1],
                              # read_entire_population='motif population 0: conf.csv',
                              population_size=agent_pop_size+200)

    # todo :add number of different motifs to the fitness function to promote regularity
    # config = "bandit reward_shape:{}, reward:{}, noise r-w:{}-{}, arms:{}-{}-{}, max_d{}, size:{}, spikes:{}, w_max{}".format(
    #     reward_shape, reward, noise_rate, noise_weight, arms[0], len(arms), random_arms, maximum_depth, size_fitness, spikes_fitness, weight_max)

    agents = agent_population(motifs,
                              pop_size=agent_pop_size,
                              inputs=2,
                              outputs=number_of_arms,
                              elitism=elitism,
                              maximum_depth=maximum_depth,
                              viable_parents=viable_parents)

    config = "bandit reward_shape:{}, reward:{}, noise r-w:{}-{}, arms:{}-{}-{}, max_d:{}, size:{}, spikes:{}, " \
             "w_max:{}, rents:{}, elitism:{}, pop_size:{}".format(reward_shape, reward, noise_rate, noise_weight,
                                                                  arms[0], len(arms), random_arms, maximum_depth,
                                                                  size_fitness, spikes_fitness,weight_max,
                                                                  viable_parents, elitism, agent_pop_size)

    globals()['pop_size'] = agent_pop_size
    globals()['config'] = config
    # globals()['connections'] = connections
    globals()['arms'] = arms
    globals()['split'] = split
    globals()['runtime'] = runtime
    globals()['reward'] = reward
    globals()['noise_rate'] = noise_rate
    globals()['noise_weight'] = noise_weight
    globals()['size_f'] = size_fitness
    globals()['spike_f'] = spikes_fitness
    globals()['exposure_time'] = exposure_time
    max_fail_score = -int(runtime / exposure_time)

    for i in range(generations):

        if random_arms:
            arms = []
            for k in range(random_arms):
                total = 1
                arm = []
                for j in range(number_of_arms - 1):
                    arm.append(np.random.uniform(0, total))
                    total -= arm[j]
                arm.append(total)
                arms.append(arm)

        if i == 0:
            connections = agents.generate_spinn_nets(input=2, output=number_of_arms, max_depth=3)
        else:
            connections = agents.generate_spinn_nets(input=2, output=number_of_arms, max_depth=3, create=False)

        # fitnesses = agents.thread_bandit(connections, arms, split=16, runtime=21000, exposure_time=200, reward=reward, noise_rate=noise_rate, noise_weight=noise_weight, size_f=size_fitness, spike_f=spikes_fitness)

        # config = 'test'
        if config != 'test':
            # arms = [0.1, 0.9, 0.2]
            # agents.bandit_test(connections, arms)
            execfile("../methods/exec_bandit.py", globals())

        fitnesses = agents.read_fitnesses(config, max_fail_score)

        print "1", motifs.total_weight

        if spikes_fitness:
            agent_spikes = []
            for k in range(agent_pop_size):
                spike_total = 0
                for j in range(len(arms)):
                    if isinstance(fitnesses[j][k], list):
                        spike_total -= fitnesses[j][k][1]
                        fitnesses[j][k] = fitnesses[j][k][0]
                    else:
                        spike_total -= 1000000
                agent_spikes.append(spike_total)
            fitnesses.append(agent_spikes)

        agents.pass_fitnesses(fitnesses)

        if config != 'test':
            agents.status_update(fitnesses, i, config, len(arms))

        print "\nconfig: ", config, "\n"

        print "2", motifs.total_weight

        motifs.adjust_weights(agents.agent_pop, reward_shape=reward_shape)

        print "3", motifs.total_weight

        if config != 'test':
            motifs.save_motifs(i, config)
            agents.save_agents(i, config)

        print "4", motifs.total_weight

        agents.evolve(species=False)

        print "finished", i, motifs.total_weight


# adjust population weights and clean up unused motifs

# generate offspring
    # mutate the individual and translate it to a new motif
    # connection change
    # swap motif

bandit(1000)

print "done"

# if __name__ == '__main__':
#     main()

#ToDo
'''
create a motif for the input/output population that is connected to the reservoir network
shifting of upper reference needs to be careful of layers larger than 10
figure out mapping to inputs
    have a fixed network but synaptic plasticity on IO
    have a IO metric attached to each motif
    connect in some fashion inputs/outputs to nodes with no inputs/outputs
        how to select order IO is chosen
        the more outgoing/incoming the better
    force a motif which represents the io 'substrate'
figure out the disparity between expected possible combinations and actual
'''