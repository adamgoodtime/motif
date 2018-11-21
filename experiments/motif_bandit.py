from methods.networks import motif_population
from methods.agents import agent_population
import numpy as np
import traceback
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing
import pathos.multiprocessing

def bandit():
    print "starting"

    # check max motif count
    motifs = motif_population(max_motif_size=3,
                              no_weight_bins=5,
                              no_delay_bins=5,
                              weight_range=(0.005, 0.05),
                              # delay_range=(),
                              # read_entire_population='motif population 0: conf.csv',
                              population_size=200)

    # todo :add number of different motifs to the fitness function to promote regularity

    agent_pop_size = 100
    # arms = [[0.1, 0.9], [0.9, 0.1]]
    # arms = [[0.2, 0.8], [0.8, 0.2]]
    # arms = [[0.4, 0.6], [0.6, 0.4]]
    arms = [[0.4, 0.6], [0.6, 0.4], [0.1, 0.9], [0.9, 0.1]]
    number_of_arms = 2
    split = 1

    reward_shape = True
    reward = 0
    noise_rate = 100
    noise_weight = 0.01
    maximum_depth = 10
    size_fitness = True
    spikes_fitness = True
    random_arms = 0

    config = "bandit reward_shape:{}, reward:{}, noise r-w:{}-{}, arms:{}-{}-{}, max_d{}, size:{}, spikes:{}".format(
        reward_shape, reward, noise_rate, noise_weight, arms[0][1], len(arms), random_arms, maximum_depth, size_fitness, spikes_fitness)

    agents = agent_population(motifs, pop_size=agent_pop_size, inputs=2, outputs=number_of_arms, maximum_depth=maximum_depth)

    for i in range(1000):

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

        # fitnesses = agents.thread_bandit(connections, arms, split=split, runtime=21000, exposure_time=200, reward=reward, noise_rate=noise_rate, noise_weight=noise_weight, size_f=size_fitness, spike_f=spikes_fitness)

        globals()['pop_size'] = agent_pop_size
        globals()['config'] = config
        globals()['connections'] = connections
        globals()['arms'] = arms
        globals()['split'] = split
        globals()['runtime'] = 21000
        globals()['reward'] = reward
        globals()['noise_rate'] = noise_rate
        globals()['noise_weight'] = noise_weight
        globals()['size_f'] = size_fitness
        globals()['spike_f'] = spikes_fitness
        globals()['exposure_time'] = 200
        execfile("../methods/exec_bandit.py", globals())

        fitnesses = agents.read_fitnesses(config)

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

        agents.status_update(fitnesses, i, config, len(arms))

        print "\nconfig: ", config, "\n"

        print "2", motifs.total_weight

        motifs.adjust_weights(agents.agent_pop, reward_shape=reward_shape)

        print "3", motifs.total_weight

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

bandit()

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