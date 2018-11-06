from methods.networks import motif_population
from methods.agents import agent_pop
import numpy as np

print "starting"

#check max motif count
motif_pop = motif_population(max_motif_size=3,
                             no_weight_bins=3,
                             no_delay_bins=3,
                             # read_entire_population='motif population 0: conf.csv',
                             population_size=200)

# motif_pop.generate_agents(max_depth=3, pop_size=100)

arms = [0.1, 0.9]

# convert motifs to networks
# agent_pop_conn = motif_pop.convert_population(inputs=1, outputs=len(arms)) # [in2e, in2i, e2e, e2i, i2e, i2i, out2e, out2i]

agents = agent_pop(motif_pop, pop_size=100)

for i in range(1000):
    if i == 0:
        connections = agents.generate_spinn_nets(input=1, output=len(arms), max_depth=3)
    else:
        connections = agents.generate_spinn_nets(input=1, output=len(arms), max_depth=3, create=False)

    # evaluate
        # pass the agent pop connections into a fucntion which tests the networks and returns fitnesses
    # fitnesses = agents.bandit_test(connections, arms, runtime=21000)
    fitnesses = np.random.randint(0, 100, len(agents.agent_pop))

    agents.pass_fitnesses(fitnesses)

    motif_pop.adjust_weights(agents.agent_pop)

    motif_pop.save_motifs(i, 'conf')

    agents.evolve(species=False)

    print "finished", i

# adjust population weights and clean up unused motifs

# generate offspring
    # mutate the individual and translate it to a new motif
    # connection change
    # swap motif

print "done"

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