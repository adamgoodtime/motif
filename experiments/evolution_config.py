connections = []
weight_max = 0.1

agent_pop_size = 100
generations = 2000
reward_shape = False
averaging_weights = True
noise_rate = 0
noise_weight = 0.01
fast_membrane = False
exec_thing = 'pen'

threading_tests = True
split = 1
new_split = 4  # agent_pop_size
max_chips_per_board = 0

#motif params
maximum_depth = [3, 7]
no_bins = [5, 100]
reset_pop = 0
size_f = False
spike_f = 0#.1#'out'
repeat_best_amount = 2
# depth fitness
make_action = True
shape_fitness = False
viable_parents = 0.2
elitism = 0.2
io_prob = 0.75  # 1.0 - (1.0 / 11.0)
read_motifs = 0
# read_motifs = 'Dirty place/Motif pop xor pl 200 stdev_n.npy'
# read_motifs = 'Dirty place/Motif pop xor pl 5000 stdev_n.npy'
read_neurons = 0
# read_neurons = 'Dirty place/Neuron pop xor pl 200 stdev_n.npy'
# read_neurons = 'Dirty place/Neuron pop xor pl 5000 stdev_n.npy'
keep_reading = 5
constant_delays = 0
max_delay = 20
base_mutate = 0
multiple_mutates = True
plasticity = False
structural = False
develop_neurons = True
stdev_neurons = False
neuron_type = 'tf_LIF'
input_current_stdev = 0.3
calcium_tau = 1000
calcium_i_alpha = 3
allow_i2o = False
weight_scale = 1.5
free_label = '--'#'{}'.format(sys.argv[1])
parallel = False
save_data = False