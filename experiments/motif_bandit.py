from methods.networks import motif_population
from methods.agents import agent_population
import numpy as np
import traceback
import threading
from multiprocessing.pool import ThreadPool
# import multiprocessing
import pathos.multiprocessing

def main():
    print "starting"

    # check max motif count
    motifs = motif_population(max_motif_size=3,
                              no_weight_bins=5,
                              no_delay_bins=5,
                              weight_range=(0.005, 0.05),
                              # delay_range=(),
                              # read_entire_population='motif population 0: conf.csv',
                              population_size=200)

    # motifs.generate_agents(max_depth=3, pop_size=100)

    # todo :print best agent, add spikes to the fitness function - weight the spikes prob, add noise?

    arms = [0.1, 0.9]

    reward_shape = True
    reward = 1

    config = "bandit reward_shape:{}, reward:{}".format(reward_shape, reward)

    # convert motifs to networks
    # agent_pop_conn = motifs.convert_population(inputs=1, outputs=len(arms)) # [in2e, in2i, e2e, e2i, i2e, i2i, out2e, out2i]

    agents = agent_population(motifs, pop_size=100)

    def helper(args):
        return agents.thread_bandit_test(*args)

    for i in range(1000):
        if i == 0:
            connections = agents.generate_spinn_nets(input=1, output=len(arms), max_depth=3)
        else:
            connections = agents.generate_spinn_nets(input=1, output=len(arms), max_depth=3, create=False)

        # evaluate
            # pass the agent pop connections into a fucntion which tests the networks and returns fitnesses
        arms = [0.1, 0.9]
        step_size = len(connections) / 4
        connection_threads = [[connections[x:x + step_size], arms, 4, 2000, 200, 100, 0.01, 0] for x in xrange(0, len(connections), step_size)]
        # pool = ThreadPool(processes=len(connection_threads))
        pool = pathos.multiprocessing.Pool(processes=len(connection_threads))
        results = []
        values = []
        # for segment in connection_threads:
            # result = pool.apply_async(func=agents.thread_bandit_test, args=(segment, arms, 4, 2000, 200, 100, 0.01, 0))

        # result = pool.map(func=agents.thread_bandit_test, iterable=connection_threads)

        result = pool.map(func=helper, iterable=connection_threads)
            # results.append(result)
            # t = threading.Thread(target=agents.thread_bandit_test, args=[segment, arms, 4, 2000, 200, 100, 0.01, 0])
            # t.start()
            # todo figure out returning values back from threads
        # for res in results:
        #     values.append(res.get())
        # the_result = result.get()

        fitnesses = agents.bandit_test(connections, arms, runtime=21000, reward=reward)
        arms = [0.9, 0.1]
        fitnesses2 = agents.bandit_test(connections, arms, runtime=21000, reward=reward)
        best_score = -1000000
        best_agent = 'they suck'
        combined_fitnesses = [fitnesses, fitnesses2]
        # for j in range(len(fitnesses)):
        #     combined_fitnesses.append(fitnesses[j] + fitnesses2[j])
        #     print j, "|", combined_fitnesses[j]
        #     if combined_fitnesses[j] > best_score:
        #         best_score = combined_fitnesses[j]
        #         best_agent = j
        # print "best fitness was ", best_score, " by agent:", best_agent, "with a score of ", fitnesses[best_agent], "and", fitnesses2[best_agent]

        # fitnesses = np.random.randint(0, 100, len(agents.agent_pop))

        print "1", motifs.total_weight

        agents.pass_fitnesses(combined_fitnesses)

        print "2", motifs.total_weight

        motifs.adjust_weights(agents.agent_pop, reward_shape=reward_shape)

        print "3", motifs.total_weight

        # for parent in agents.agent_pop:
        #     list_child = []
        #     list_child = motifs.list_motifs(parent[0], list_child)
        #     for item in list_child:
        #         try:
        #             motif = motifs.motif_configs[item]
        #         except:
        #             traceback.print_exc()
        #             print "weights killed it 1"

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

print "done"

if __name__ == '__main__':
    main()

#ToDo
'''
complete checks for infinite loops, in mutate mainly
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