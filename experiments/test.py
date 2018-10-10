from methods.networks import motif_population

print "starting"

#check max motif count
motif_pop = motif_population(max_motif_size=3,
                             no_weight_bins=6,
                             no_delay_bins=6,
                             motif_d=((-1, 1), (-1, 1)),
                             population_size=500)

motif_pop.generate_agents(inputs=(range(5),), outputs=range(2))

# convert motifs to networks
agent_pop = motif_pop.convert_population()

# evaluate

# adjust population weights and clean up unused motifs

# generate offspring
    # mutate the individual and translate it to a new motif
    # connection change
    # swap motif
    # create a mirror copy

print "done"