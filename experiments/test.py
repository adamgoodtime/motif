from methods.networks import motif_population

print "starting"

#check max motif count
motif_pop = motif_population(max_motif_size=3,
                             no_weight_bins=6,
                             no_delay_bins=6,
                             population_size=500)

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