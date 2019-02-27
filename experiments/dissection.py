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

def read_motif(motif_id, iteration, file_location, config):
    motifs = motif_population(max_motif_size=3,
                              no_weight_bins=5,
                              no_delay_bins=5,
                              weight_range=(0.005, weight_max),
                              # delay_range=(2, 2.00001),
                              io_weight=[inputs, outputs, io_weight],
                              read_entire_population='{}/Motif population {}: {}.csv'.format(file_location, iteration, config),
                              population_size=100)

    motif = motifs.motif_configs[motif_id]
    motif_struct = motifs.read_motif(motif_id)
    [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
     i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out] = motifs.convert_individual(motif_id, inputs, outputs)
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
            print "no more files or bad config"
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

io_weight = 1

weight_max = 0.1

arm1 = 1
arm2 = 0
# arm3 = 0.1
arm_len = 1
arms = []
for i in range(arm_len):
    arms.append([arm1, arm2])
    arms.append([arm2, arm1])
    # for arm in list(itertools.permutations([arm1, arm2, arm3])):
    #     arms.append(list(arm))
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1], [0, 1], [1, 0]]
'''top_prob = 1
0.1 = base prob 1
0.2 equals base prob 2
etc
split node and share inputs but half outputs
arms = [[0.1, 0.2, top_prob, 0.3, 0.2, 0.1, 0.2, 0.1], [top_prob, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.2],
        [0.3, top_prob, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1], [0.2, 0.1, 0.1, top_prob, 0.2, 0.3, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.2, top_prob, 0.2, 0.3, 0.2], [0.1, 0.2, 0.1, 0.2, 0.2, top_prob, 0.1, 0.3],
        [0.2, 0.1, 0.3, 0.1, 0.2, 0.1, top_prob, 0.2], [0.1, 0.3, 0.2, 0.2, 0.1, 0.2, 0.1, top_prob]]
# '''
if isinstance(arms[0], list):
    number_of_arms = len(arms[0])
else:
    number_of_arms = len(arms)

agent_pop_size = 100
reward_shape = False
averaging_weights = True
reward = 1
noise_rate = 0
noise_weight = 0.01
fast_membrane = False

threading_tests = True
split = 1
new_split = agent_pop_size

maximum_depth = [4, 10]
no_bins = [10, 75]
reset_pop = 0
size_f = False
spike_f = False # 'out'
shape_fitness = True
random_arms = 0
viable_parents = 0.2
elitism = 0.2
runtime = 41000
exposure_time = 200
io_weighting = 20
read_pop = 0  # 'new_io_motif_easy_3.csv'
keep_reading = 5
constant_delays = 1
base_mutate = 0
multiple_mutates = True
exec_thing = 'pen'
plasticity = False
free_label = 0

max_fail_score = 0

no_v = False
encoding = 1
time_increment = 20
pole_length = 1
pole_angle = [[0.1], [0.2], [-0.1], [-0.2]]
reward_based = 1
force_increments = 20
max_firing_rate = 1000
number_of_bins = 6
central = 1
bin_overlap = 2
tau_force = 0

x_factor = 8
y_factor = 8
bricking = 0

inputs = 0
outputs = 0
test_data_set = []
config = ''

if exec_thing == 'br':
    inputs = (160 / x_factor) * (128 / y_factor)
    outputs = 2
    config = 'bout {}-{}-{} '.format(x_factor, y_factor, bricking)
    test_data_set = 'something'
    number_of_tests = 'something'
elif exec_thing == 'xor':
    arms = [[0, 0], [0, 1], [1, 0], [1, 1]]
    config = 'xor '
    inputs = 2
    if reward == 1:
        outputs = 2
    else:
        outputs = 1
    max_fail_score = -1
    test_data_set = arms
    number_of_tests = len(arms)
elif exec_thing == 'pen':
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
    inputs = 4 * number_of_bins
    if no_v:
        inputs /= 2
    outputs = force_increments
    config = 'rank-pend-an{}-{}-F{}-R{}-B{}-O{}-E{} '.format(pole_angle[0], len(pole_angle), force_increments, max_firing_rate, number_of_bins, bin_overlap, encoding)
    test_data_set = pole_angle
    number_of_tests = len(pole_angle)
else:
    test_data_set = arms
    inputs = 2
    outputs = number_of_arms
    config = 'bandit-{}-{}-{} '.format(arms[0][0], len(arms), random_arms)
    number_of_tests = len(arms)
if plasticity:
    config += 'pl '
if averaging_weights:
    config += 'ave '
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
    config += 'no_v'
if free_label:
    config += '{} '.format(free_label)

config += "ex-{}, reward-{}, max_d-{}, w_max-{}, rents-{}, elite-{}, psize-{}, bins-{}".format(
    exec_thing, reward, maximum_depth, weight_max, viable_parents, elitism, agent_pop_size,
    no_bins)

if io_weighting:
    config += ", io-{}".format(io_weighting)
else:
    config += " {}".format(io_weight)
if read_pop:
    config += ' read-{}'.format(keep_reading)
# config = "bandit reward_shape:{}, reward:{}, noise r-w:{}-{}, arms:{}-{}-{}, max_d{}, size:{}, spikes:{}, w_max{}".format(
#     reward_shape, reward, noise_rate, noise_weight, arms[0], len(arms), random_arms, maximum_depth, size_fitness, spikes_fitness, weight_max)

# motif_tracking(file_location, config)
# read_motif('12202', 30, file_location, config)
mutate_anal(file_location, config, 50)