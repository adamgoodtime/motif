from methods.networks import motif_population

print "starting"

#check max motif count
motif_pop = motif_population(max_motif_size=2,
                             no_weight_bins=2,
                             no_delay_bins=2,
                             population_size=50000)

motif_pop.generate_agents(inputs=6, outputs=2)

# convert motifs to networks
agent_pop = motif_pop.convert_population()

# evaluate

# adjust population weights and clean up unused motifs

# generate offspring
    # mutate the individual and translate it to a new motif
    # connection change
    # swap motif

print "done"

#ToDo
'''
figure out mapping to inputs
    have a fixed network but synaptic plasticity on IO
    have a IO metric attached to each motif
    connect in some fashion inputs/outputs to nodes with no inputs/outputs
        how to select order IO is chosen
        the more outgoing/incoming the better
    force a motif which represents the io 'substrate'
should a motif of motifs become a motif in it's own right
    store original motif and child motifs in agent if successful create new motif, label level it was useful?
figure out the disparity between expected possible combinations and actual
'''