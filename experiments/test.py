from methods.networks import motif_population

print "starting"

#check max motif count
motif_pop = motif_population(population_size=200)

agent_pop = motif_pop.generate_agents()

print "done"