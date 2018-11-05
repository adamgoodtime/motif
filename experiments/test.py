from methods.networks import motif_population
from methods.agents import agent_pop
import numpy as np

print "starting"

#check max motif count
motif_pop = motif_population(max_motif_size=3,
                             no_weight_bins=3,
                             no_delay_bins=3,
                             population_size=100)

# motif_pop.generate_agents(max_depth=3, pop_size=100)

arms = [0.1, 0.9]

# convert motifs to networks
# agent_pop_conn = motif_pop.convert_population(inputs=1, outputs=len(arms)) # [in2e, in2i, e2e, e2i, i2e, i2i, out2e, out2i]

agents = agent_pop(motif_pop)

connections = agents.generate_spinn_nets(input=1, output=len(arms), max_depth=3)

agents.motifs.reward_shape()

# evaluate
    # pass the agent pop connections into a fucntion which tests the networks and returns fitnesses
# fitnesses = agents.bandit_test(connections, arms, runtime=21000)
fitnesses = np.random.randint(0, 100, len(agents.agent_pop))

agents.pass_fitnesses(fitnesses)

motif_pop.adjust_weights(agents.agent_pop)

agents.evolve(species=False)

# adjust population weights and clean up unused motifs

# generate offspring
    # mutate the individual and translate it to a new motif
    # connection change
    # swap motif

print "done"

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
should a motif of motifs become a motif in it's own right - doesn't because weight = 1
    store original motif and child motifs in agent if successful create new motif, label level it was useful?
    store new motifs in a separate population and transfer if useful
create a function to generate random motifs from initial settings
figure out the disparity between expected possible combinations and actual
'''