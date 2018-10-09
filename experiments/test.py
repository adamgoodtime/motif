from methods.networks import motif_population

print "starting"

#check max motif count
motif_pop = motif_population(max_motif_size=4,
                             no_weight_bins=7,
                             no_delay_bins=7,
                             population_size=200)

agent_pop = motif_pop.generate_agents()

print "done"