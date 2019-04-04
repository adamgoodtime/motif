from methods.networks import motif_population
from methods.agents import agent_population
import numpy as np
import traceback
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing
import pathos.multiprocessing
import csv
from ast import literal_eval
import itertools

#todo here will be where an agent and a motif population will be read in to examine what is present in a well performing agent

#todo could also do motif analysis and progression throught the simulation

# read motif populations over time
# see how each changes with time and how much weight it has

def motif_tracking(file_location, config):
    create_files = True
    motifs_over_time = []
    weights_over_time = {}
    print '{}/Motif population {}: {}'.format(file_location, 0, config)
    for i in range(3000):
        try:
            with open('{}/Motif population {}: {}.csv'.format(file_location, i, config)) as motif_file:
                total_weight = 0
                weight_dist = []
                csvFile = csv.reader(motif_file)
                motif = False
                for row in csvFile:
                    temp = row
                    if temp[0] == 'node':
                        if motif:
                            try:
                                weights_over_time[motif['id']].append(motif['weight'])
                            except:
                                empty_slots = []
                                for j in range(i):
                                    empty_slots.append(0)
                                empty_slots.append(motif['weight'])
                                weights_over_time[motif['id']] = empty_slots
                            weight_dist.append([motif['id'], motif['weight']])
                            # self.insert_motif(deepcopy(motif), weight=motif['weight'], read=True)
                        motif = {}
                    atribute = temp[0]
                    del temp[0]
                    if atribute == 'depth':
                        temp = int(temp[0])
                    elif atribute == 'weight':
                        temp = literal_eval(temp[0])
                        total_weight += temp
                    elif atribute == 'conn' or atribute == 'io':
                        for j in range(len(temp)):
                            temp[j] = literal_eval(temp[j])
                    elif atribute == 'id':
                        temp = temp[0]
                    motif['{}'.format(atribute)] = temp
            for key in weights_over_time:
                if len(weights_over_time[key]) <= i:
                    weights_over_time[key].append(0)
            for j in range(len(weight_dist)):
                weight_dist[j][1] /= float(total_weight)
            weight_dist.sort(key=lambda x: x[1], reverse=True)
            motifs_over_time.append(weight_dist)
            print "finished ", i
        except:
            if i == 0:
                create_files = False
            print "no file to read"
            break

    if create_files:
        with open('{}/Motifs over time: {}.csv'.format(file_location, config), 'w') as weight_file:
            writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
            for iteration in motifs_over_time:
                writer.writerow(iteration)
                # for motif in iteration:
                #     writer.writerow(motif)
            weight_file.close()
        print "finished 0"

        with open('{}/meta data/5 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
            writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
            for motif in weights_over_time:
                data = [motif]
                entries = 0
                for weight in weights_over_time[motif]:
                    if weight != 0:
                        entries += 1
                    data.append(weight)
                if entries > 5:
                    writer.writerow(data)
                # for motif in iteration:
                #     writer.writerow(motif)
            weight_file.close()
        print "finished 5"

        with open('{}/meta data/10 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
            writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
            for motif in weights_over_time:
                data = [motif]
                entries = 0
                for weight in weights_over_time[motif]:
                    if weight != 0:
                        entries += 1
                    data.append(weight)
                if entries > 10:
                    writer.writerow(data)
                # for motif in iteration:
                #     writer.writerow(motif)
            weight_file.close()
        print "finished 10"

        with open('{}/meta data/1 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
            writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
            for motif in weights_over_time:
                data = [motif]
                entries = 0
                for weight in weights_over_time[motif]:
                    if weight != 0:
                        entries += 1
                    data.append(weight)
                if entries > 1:
                    writer.writerow(data)
                # for motif in iteration:
                #     writer.writerow(motif)
            weight_file.close()
        print "finished 1"

        with open('{}/meta data/2 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
            writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
            for motif in weights_over_time:
                data = [motif]
                entries = 0
                for weight in weights_over_time[motif]:
                    if weight != 0:
                        entries += 1
                    data.append(weight)
                if entries > 2:
                    writer.writerow(data)
                # for motif in iteration:
                #     writer.writerow(motif)
            weight_file.close()
        print "finished 2"

        with open('{}/meta data/20 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
            writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
            for motif in weights_over_time:
                data = [motif]
                entries = 0
                for weight in weights_over_time[motif]:
                    if weight != 0:
                        entries += 1
                    data.append(weight)
                if entries > 20:
                    writer.writerow(data)
                # for motif in iteration:
                #     writer.writerow(motif)
            weight_file.close()
        print "finished 20"

        with open('{}/meta data/50 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
            writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
            for motif in weights_over_time:
                data = [motif]
                entries = 0
                for weight in weights_over_time[motif]:
                    if weight != 0:
                        entries += 1
                    data.append(weight)
                if entries > 50:
                    writer.writerow(data)
                # for motif in iteration:
                #     writer.writerow(motif)
            weight_file.close()
        print "finished 50"

        with open('{}/meta data/100 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
            writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
            for motif in weights_over_time:
                data = [motif]
                entries = 0
                for weight in weights_over_time[motif]:
                    if weight != 0:
                        entries += 1
                    data.append(weight)
                if entries > 100:
                    writer.writerow(data)
                # for motif in iteration:
                #     writer.writerow(motif)
            weight_file.close()
        print "finished 100"

def test_failure(file_location):
    global connections
    connections = np.load(file_location)
    execfile("../methods/exec_subprocess.py", globals())

def read_motif(motif_id, iteration, file_location, config):
    motifs = motif_population(max_motif_size=3,
                              no_weight_bins=5,
                              no_delay_bins=5,
                              weight_range=(0.005, weight_max),
                              # delay_range=(2, 2.00001),
                              read_entire_population='{}/Motif population {}: {}.csv'.format(file_location, iteration, config),
                              population_size=100)

    motif = motifs.motif_configs[motif_id]
    motif_struct = motifs.read_motif(motif_id)
    [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
     i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out] = motifs.convert_individual(motif_id)
    connections = [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
     i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out]
    # motif1 = motifs.motif_configs[motif_id-1]
    # motif_struct1 = motifs.read_motif(motif_id-1)
    # motif2 = motifs.motif_configs[motif_id-2]
    # motif_struct2 = motifs.read_motif(motif_id-2)
    print motif_struct

def mutate_anal(file_location, config, until):
    print "attempting file - {}/mutate keys for {}: {}.csv".format(file_location, 0, config)
    mutate_track = {}
    mutate_track['motif'] = []
    mutate_track['new'] = []
    mutate_track['node'] = []
    mutate_track['io'] = []
    mutate_track['m_add'] = []
    mutate_track['m_gone'] = []
    mutate_track['c_add'] = []
    mutate_track['c_gone'] = []
    mutate_track['param_w'] = []
    mutate_track['param_d'] = []
    mutate_track['in_shift'] = []
    mutate_track['out_shift'] = []
    mutate_track['plasticity'] = []
    mutate_track['sex'] = []
    mutate_track['asex'] = []
    mutate_track['both'] = []
    mutate_track['fresh'] = []
    for i in range(until):
        try:
            with open('{}/mutate keys for {}: {}.csv'.format(file_location, i, config)) as mutate_file:
                csvFile = csv.reader(mutate_file)
                print "reading iteration", i
                for row in csvFile:
                    temp = row
                    if temp[0] == 'motif':
                        motif = literal_eval(temp[1])
                    elif temp[0] == 'new':
                        new = literal_eval(temp[1])
                    elif temp[0] == 'node':
                        node = literal_eval(temp[1])
                    elif temp[0] == 'io':
                        io = literal_eval(temp[1])
                    elif temp[0] == 'm_add':
                        m_add = literal_eval(temp[1])
                    elif temp[0] == 'm_gone':
                        m_gone = literal_eval(temp[1])
                    elif temp[0] == 'c_add':
                        c_add = literal_eval(temp[1])
                    elif temp[0] == 'c_gone':
                        c_gone = literal_eval(temp[1])
                    elif temp[0] == 'param_w':
                        param_w = literal_eval(temp[1])
                    elif temp[0] == 'param_d':
                        param_d = literal_eval(temp[1])
                    elif temp[0] == 'in_shift':
                        in_shift = literal_eval(temp[1])
                    elif temp[0] == 'out_shift':
                        out_shift = literal_eval(temp[1])
                    elif temp[0] == 'plasticity':
                        plasticity = literal_eval(temp[1])
                    elif temp[0] == 'mum':
                        mum = literal_eval(temp[1])
                        mum_fitness = mum[0]
                        mum_score = mum[1]
                    elif temp[0] == 'dad':
                        dad = literal_eval(temp[1])
                        dad_fitness = dad[0]
                        dad_score = dad[1]
                    elif temp[0] == 'sex':
                        sex = literal_eval(temp[1])
                    else:
                        if i > 1:
                            mum_fitness_change = agent_fitness - mum_fitness
                            dad_fitness_change = agent_fitness - dad_fitness
                            ave_fitness_change = (mum_fitness_change + dad_fitness_change) / 2
                            mum_score_change = agent_score - mum_score
                            dad_score_change = agent_score - dad_score
                            ave_score_change = (mum_score_change + dad_score_change) / 2
                            # if motif:
                            #     mutate_track['motif'].append([ave_fitness_change * motif, ave_score_change * motif])
                            # if new:
                            #     mutate_track['new'].append([ave_fitness_change * new, ave_score_change * new])
                            # if node:
                            #     mutate_track['node'].append([ave_fitness_change * node, ave_score_change * node])
                            # if io:
                            #     mutate_track['io'].append([ave_fitness_change * io, ave_score_change * io])
                            # if m_add:
                            #     mutate_track['m_add'].append([ave_fitness_change * m_add, ave_score_change * m_add])
                            # if m_gone:
                            #     mutate_track['m_gone'].append([ave_fitness_change * m_gone, ave_score_change * m_gone])
                            # if c_add:
                            #     mutate_track['c_add'].append([ave_fitness_change * c_add, ave_score_change * c_add])
                            # if c_gone:
                            #     mutate_track['c_gone'].append([ave_fitness_change * c_gone, ave_score_change * c_gone])
                            # if param_w:
                            #     mutate_track['param_w'].append([ave_fitness_change * param_w, ave_score_change * param_w])
                            # if param_d:
                            #     mutate_track['param_d'].append([ave_fitness_change * param_d, ave_score_change * param_d])
                            # if sex:
                            #     mutate_track['sex'].append([ave_fitness_change, ave_score_change])
                            # if not sex:
                            #     mutate_track['asex'].append([ave_fitness_change, ave_score_change])
                            if motif:
                                mutate_track['motif'].append(ave_score_change * motif)
                            if new:
                                mutate_track['new'].append(ave_score_change * new)
                            if node:
                                mutate_track['node'].append(ave_score_change * node)
                            if io:
                                mutate_track['io'].append(ave_score_change * io)
                            if m_add:
                                mutate_track['m_add'].append(ave_score_change * m_add)
                            if m_gone:
                                mutate_track['m_gone'].append(ave_score_change * m_gone)
                            if c_add:
                                mutate_track['c_add'].append(ave_score_change * c_add)
                            if c_gone:
                                mutate_track['c_gone'].append(ave_score_change * c_gone)
                            if param_w:
                                mutate_track['param_w'].append(ave_score_change * param_w)
                            if param_d:
                                mutate_track['param_d'].append(ave_score_change * param_d)
                            if param_w:
                                mutate_track['in_shift'].append(ave_score_change * in_shift)
                            if param_d:
                                mutate_track['out_shift'].append(ave_score_change * out_shift)
                            if plasticity:
                                mutate_track['plasticity'].append(ave_score_change * plasticity)
                            if sex == 1:
                                mutate_track['sex'].append(ave_score_change)
                            if sex == 0:
                                mutate_track['asex'].append(ave_score_change)
                            if sex == 2:
                                mutate_track['both'].append(ave_score_change)
                            if sex == 3:
                                mutate_track['fresh'].append(ave_score_change)
                            if motif:
                                mutate_track['motif'].append(ave_fitness_change * motif)
                            if new:
                                mutate_track['new'].append(ave_fitness_change * new)
                            if node:
                                mutate_track['node'].append(ave_fitness_change * node)
                            if io:
                                mutate_track['io'].append(ave_fitness_change * io)
                            if m_add:
                                mutate_track['m_add'].append(ave_fitness_change * m_add)
                            if m_gone:
                                mutate_track['m_gone'].append(ave_fitness_change * m_gone)
                            if c_add:
                                mutate_track['c_add'].append(ave_fitness_change * c_add)
                            if c_gone:
                                mutate_track['c_gone'].append(ave_fitness_change * c_gone)
                            if param_w:
                                mutate_track['param_w'].append(ave_fitness_change * param_w)
                            if param_d:
                                mutate_track['param_d'].append(ave_fitness_change * param_d)
                            if param_w:
                                mutate_track['in_shift'].append(ave_fitness_change * in_shift)
                            if param_d:
                                mutate_track['out_shift'].append(ave_fitness_change * out_shift)
                            if plasticity:
                                mutate_track['plasticity'].append(ave_fitness_change * plasticity)
                            if sex == 1:
                                mutate_track['sex'].append(ave_fitness_change)
                            if sex == 0:
                                mutate_track['asex'].append(ave_fitness_change)
                            if sex == 2:
                                mutate_track['both'].append(ave_fitness_change)
                            if sex == 3:
                                mutate_track['fresh'].append(ave_fitness_change)
                            motif = 0
                            nen = 0
                            io = 0
                            m_add = 0
                            m_gone = 0
                            c_add = 0
                            c_gone = 0
                            param_w = 0
                            param_d = 0
                            mum_fitness = 0
                            mum_score = 0
                            dad_fitness = 0
                            dad_score = 0
                            sex = 0
                        agent_fitness = literal_eval(temp[1])
                        agent_score = literal_eval(temp[2])
        except:
            traceback.print_exc()
            print "no more files or bad config"jhvjhgchgc
            if i > 10:
                with open('{}/meta data/Mutate over time until {}: {}.csv'.format(file_location, until, config), 'w') as mutate_file:
                    writer = csv.writer(mutate_file, delimiter=',', lineterminator='\n')
                    row = []
                    for operator in mutate_track:
                        row.append('{} scr'.format(operator))
                        row.append('{} fit'.format(operator))
                    writer.writerow(row)
                    row = []
                    i = 0
                    while True:
                        row = []
                        all_done = True
                        for operator in mutate_track:
                            if len(mutate_track[operator]) > i:
                                all_done = False
                                row.append(mutate_track[operator][i])
                                row.append(mutate_track[operator][i+1])
                            else:
                                row.append('')
                        writer.writerow(row)
                        if all_done:
                            break
                        i += 2
                    mutate_file.close()
            break
    if i == until - 1:
        with open('{}/meta data/Mutate over time until {}: {}.csv'.format(file_location, until, config),
                  'w') as mutate_file:
            writer = csv.writer(mutate_file, delimiter=',', lineterminator='\n')
            row = []
            for operator in mutate_track:
                row.append('{} scr'.format(operator))
                row.append('{} fit'.format(operator))
            writer.writerow(row)
            row = []
            i = 0
            while True:
                row = []
                all_done = True
                for operator in mutate_track:
                    if len(mutate_track[operator]) > i:
                        all_done = False
                        row.append(mutate_track[operator][i])
                        row.append(mutate_track[operator][i + 1])
                    else:
                        row.append('')
                writer.writerow(row)
                if all_done:
                    break
                i += 2
            mutate_file.close()


file_location = 'runtime data/high io, good pendulum'

connections = []

weight_max = 0.1

agent_pop_size = 100
reward_shape = False
averaging_weights = True
noise_rate = 0
noise_weight = 0.01
fast_membrane = False

threading_tests = True
split = 1
new_split = 4  # agent_pop_size

#motif params
maximum_depth = [3, 15]
no_bins = [10, 375]
reset_pop = 0
size_f = False
spike_f = False#'out'
make_action = True
shape_fitness = True
random_arms = 0
viable_parents = 0.2
elitism = 0.2
exposure_time = 200
io_prob = 0.75  # 1.0 - (1.0 / 11.0)
read_pop = 0
# read_pop = 'Dirty place/good pendulum with plastic and high bins.csv'
keep_reading = 5
constant_delays = 0
base_mutate = 0
multiple_mutates = True
exec_thing = 'arms'
plasticity = True
structural = True
develop_neurons = True
stdev_neurons = True
free_label = 0

#arms params
arms_runtime = 41000
arm1 = 0.8
arm2 = 0.1
arm3 = 0.1
arm_len = 1
arms = []
arms_reward = 1
for i in range(arm_len):
    # arms.append([arm1, arm2])
    # arms.append([arm2, arm1])
    for arm in list(itertools.permutations([arm1, arm2, arm3])):
        arms.append(list(arm))
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1], [0, 1], [1, 0]]
'''top_prob = 1
low_prob = 0
med_prob = 0.1
hii_prob = 0.2
arms = [[low_prob, med_prob, top_prob, hii_prob, med_prob, low_prob, med_prob, low_prob], [top_prob, low_prob, low_prob, med_prob, hii_prob, med_prob, low_prob, med_prob],
        [hii_prob, top_prob, med_prob, low_prob, low_prob, med_prob, med_prob, low_prob], [med_prob, low_prob, low_prob, top_prob, med_prob, hii_prob, low_prob, med_prob],
        [low_prob, low_prob, low_prob, med_prob, top_prob, med_prob, hii_prob, med_prob], [low_prob, med_prob, low_prob, med_prob, med_prob, top_prob, low_prob, hii_prob],
        [med_prob, low_prob, hii_prob, low_prob, med_prob, low_prob, top_prob, med_prob], [low_prob, hii_prob, med_prob, med_prob, low_prob, med_prob, low_prob, top_prob]]
# '''

#pendulum params
pendulum_runtime = 181000
double_pen_runtime = 60000
max_fail_score = 0
no_v = False
encoding = 0
time_increment = 20
pole_length = 1
pole2_length = 0.1
pole_angle = [[0.1], [0.2], [-0.1], [-0.2]]
reward_based = 1
force_increments = 20
max_firing_rate = 1000
number_of_bins = 6
central = 1
bin_overlap = 2
tau_force = 0

#logic params
logic_runtime = 5000
score_delay = 200
stochastic = 1
truth_table = [0, 1, 1, 0]
# truth_table = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
input_sequence = []
segment = [0 for j in range(int(np.log2(len(truth_table))))]
input_sequence.append(segment)
for i in range(1, len(truth_table)):
    current_value = i
    segment = [0 for j in range(int(np.log2(len(truth_table))))]
    while current_value != 0:
        highest_power = int(np.log2(current_value))
        segment[highest_power] = 1
        current_value -= 2**highest_power
    input_sequence.append(segment)

#Recall params
recall_runtime = 60000
rate_on = 50
rate_off = 0
recall_pop_size = 1
prob_command = 1./6.
prob_in_change = 1./2.
time_period = 200
stochastic = 1
recall_reward = 0
recall_parallel_runs = 2

#MNIST
max_freq = 5000
on_duration = 1000
off_duration = 1000
data_size = 200
mnist_parallel_runs = 2
mnist_runtime = data_size * (on_duration + off_duration)

#erbp params
erbp_runtime = 20
erbp_max_depth = [5, 100]

#breakout params
breakout_runtime = 181000
x_factor = 8
y_factor = 8
bricking = 0

inputs = 0
outputs = 0
test_data_set = []
config = ''
runtime = 0

if exec_thing == 'br':
    runtime = arms_runtime
    inputs = (160 / x_factor) * (128 / y_factor)
    outputs = 2
    config = 'bout {}-{}-{} '.format(x_factor, y_factor, bricking)
    test_data_set = 'something'
    number_of_tests = 'something'
elif exec_thing == 'pen':
    runtime = pendulum_runtime
    encoding = 1
    inputs = 4
    if encoding != 0:
        inputs *= number_of_bins
    if no_v:
        inputs /= 2
    outputs = 2
    config = 'pend-an{}-{}-F{}-R{}-B{}-O{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap)
    test_data_set = pole_angle
    number_of_tests = len(pole_angle)
elif exec_thing == 'rank pen':
    runtime = pendulum_runtime
    inputs = 4 * number_of_bins
    if no_v:
        inputs /= 2
    outputs = force_increments
    config = 'rank-pend-an{}-{}-F{}-R{}-B{}-O{}-E{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap, encoding)
    test_data_set = pole_angle
    number_of_tests = len(pole_angle)
elif exec_thing == 'double pen':
    runtime = double_pen_runtime
    inputs = 6 * number_of_bins
    if no_v:
        inputs /= 2
    outputs = force_increments
    config = 'double-pend-an{}-{}-pl{}-{}-F{}-R{}-B{}-O{} '.format(pole_angle[0], len(pole_angle), pole_length, pole2_length, force_increments, max_firing_rate, number_of_bins, bin_overlap)
    test_data_set = pole_angle
    number_of_tests = len(pole_angle)
elif exec_thing == 'arms':
    if isinstance(arms[0], list):
        number_of_arms = len(arms[0])
    else:
        number_of_arms = len(arms)
    runtime = arms_runtime
    test_data_set = arms
    inputs = 2
    outputs = number_of_arms
    config = 'bandit-{}-{}-{} '.format(arms[0][0], len(arms), random_arms)
    number_of_tests = len(arms)
elif exec_thing == 'logic':
    runtime = logic_runtime
    test_data_set = input_sequence
    number_of_tests = len(input_sequence)
    inputs = len(input_sequence[0])
    outputs = 2
    if stochastic:
        config = 'logic-stoc-{}-run{}-sample{} '.format(truth_table, runtime, score_delay)
    else:
        config = 'logic-{}-run{}-sample{} '.format(truth_table, runtime, score_delay)
elif exec_thing == 'recall':
    runtime = recall_runtime
    for j in range(recall_parallel_runs):
        test_data_set.append([j])
    number_of_tests = recall_parallel_runs
    inputs = 4 * recall_pop_size
    outputs = 2
    if stochastic:
        config = 'recall-stoc-pop_s{}-run{}-in_p{}-r_on{} '.format(recall_pop_size, runtime, prob_in_change, rate_on)
    else:
        config = 'recall-pop_s{}-run{}-in_p{}-r_on{} '.format(recall_pop_size, runtime, prob_in_change, rate_on)
elif exec_thing == 'mnist':
    runtime = mnist_runtime
    for j in range(mnist_parallel_runs):
        test_data_set.append([j])
    number_of_tests = mnist_parallel_runs
    inputs = 28*28
    outputs = 10
    config = 'mnist-freq-{}-on-{}-off-{}-size-{} '.format(max_freq, on_duration, off_duration, data_size)
elif exec_thing == 'erbp':
    maximum_depth = erbp_max_depth
    make_action = False
    runtime = erbp_runtime
    inputs = 0
    outputs = 0
    test_data_set = [[0], [1]]
    number_of_tests = len(test_data_set)
    config = 'erbp {} {} '.format(runtime, maximum_depth)
else:
    print "\nNot a correct test setting\n"
    raise Exception
if plasticity:
    config += 'pl '
if averaging_weights:
    config += 'ave '
if make_action:
    config += 'action '
if spike_f:
    if spike_f == 'out':
        config += 'out-spikes '
    else:
        config += 'spikes '
if size_f:
    config += 'size '
if reward_shape:
    config += 'shape_r '
if shape_fitness:
    config += 'shape_f '
if reset_pop:
    config += 'reset-{} '.format(reset_pop)
if base_mutate:
    config += 'mute-{} '.format(base_mutate)
if multiple_mutates:
    config += 'multate '
if noise_rate:
    config += 'n r-w-{}-{} '.format(noise_rate, noise_weight)
if constant_delays:
    config += 'delay-{} '.format(constant_delays)
if fast_membrane:
    config += 'fast_mem '
if no_v:
    config += 'no_v '
if develop_neurons:
    config += 'dev_n '
if stdev_neurons:
    config += 'stdev_n '
if free_label:
    config += '{} '.format(free_label)

config += "max_d-{}, w_max-{}, rents-{}, elite-{}, psize-{}, bins-{}".format(
    maximum_depth, weight_max, viable_parents, elitism, agent_pop_size, no_bins)

if io_prob:
    config += ", io-{}".format(io_prob)
else:
    config += " {}".format(motifs.global_io[1])
if read_pop:
    config += ' read-{}'.format(keep_reading)

# motif_tracking(file_location, config)
# read_motif('12202', 30, file_location, config)
mutate_anal(file_location, config, 50)